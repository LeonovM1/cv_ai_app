import os
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io

app = Flask(__name__)
CORS(app)

# ---------- Регистрация шрифта для кириллицы ----------
# Пытаемся загрузить DejaVuSans.ttf из папки проекта
FONT_NAME = 'Helvetica'  # fallback
try:
    pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
    FONT_NAME = 'DejaVuSans'
    print("Шрифт DejaVuSans загружен")
except:
    try:
        # Попробуем системный Arial (Windows)
        pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
        FONT_NAME = 'Arial'
        print("Шрифт Arial загружен")
    except:
        print("Шрифт для кириллицы не найден, используется Helvetica (только латиница)")

# ---------- Модель ----------
model = YOLO('yolov8n.pt')

# ---------- База данных ----------
DB_NAME = 'history.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            total_people INTEGER,
            on_crosswalk INTEGER,
            classes_detected TEXT,
            result_image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_to_db(filename, total, on_crosswalk, class_list, result_path):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO requests (timestamp, filename, total_people, on_crosswalk, classes_detected, result_image_path)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), filename, total, on_crosswalk, json.dumps(class_list), result_path))
    conn.commit()
    conn.close()

def get_all_history():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT id, timestamp, filename, total_people, on_crosswalk, classes_detected, result_image_path FROM requests ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    return rows

def get_record_by_id(record_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT id, timestamp, filename, total_people, on_crosswalk, classes_detected, result_image_path FROM requests WHERE id = ?', (record_id,))
    row = c.fetchone()
    conn.close()
    return row

def point_in_polygon(x, y, poly):
    poly_pts = np.array(poly, dtype=np.int32)
    return cv2.pointPolygonTest(poly_pts, (x, y), False) >= 0

# ---------- Универсальная функция построения PDF ----------
def build_pdf(records, title="Отчёт по подсчёту пешеходов", include_stats=True):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    
    # Стили с использованием зарегистрированного шрифта
    styles = getSampleStyleSheet()
    # Создаём свои стили с нужным шрифтом
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontName=FONT_NAME,
        fontSize=18,
        alignment=1,  # center
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName=FONT_NAME,
        fontSize=14,
        spaceAfter=6
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=10
    )
    
    elements = []
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Дата генерации: {datetime.now().strftime('%d.%m.%Y %H:%M')}", normal_style))
    elements.append(Spacer(1, 24))
    
    if include_stats and records:
        total_entries = len(records)
        total_people_all = sum(row[3] for row in records)  # total_people
        total_on_cross = sum(row[4] for row in records)    # on_crosswalk
        avg_on_cross = total_on_cross / total_entries if total_entries else 0
        
        elements.append(Paragraph("Общая статистика:", heading_style))
        stats_text = f"""
        <b>Всего обработано изображений:</b> {total_entries}<br/>
        <b>Всего обнаружено людей:</b> {total_people_all}<br/>
        <b>Всего людей на переходе:</b> {total_on_cross}<br/>
        <b>Среднее количество людей на переходе на изображение:</b> {avg_on_cross:.2f}
        """
        elements.append(Paragraph(stats_text, normal_style))
        elements.append(Spacer(1, 24))
    
    if records:
        elements.append(Paragraph("Детали по каждому изображению:", heading_style))
        elements.append(Spacer(1, 12))
        
        table_data = [["№", "Время", "Файл", "Всего людей", "На переходе"]]
        for idx, row in enumerate(records, start=1):
            timestamp = row[1][:19]
            fname = row[2][:30]
            total = str(row[3])
            on_cross = str(row[4])
            table_data.append([str(idx), timestamp, fname, total, on_cross])
        
        table = Table(table_data, colWidths=[0.5*inch, 1.5*inch, 2.0*inch, 1.0*inch, 1.2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), FONT_NAME),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(table)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------- Маршруты ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = file.filename
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    polygon_str = request.form.get('polygon')
    if not polygon_str:
        return jsonify({'error': 'Не задана область перехода (полигон)'}), 400

    try:
        polygon_rel = json.loads(polygon_str)
    except:
        return jsonify({'error': 'Неверный формат полигона'}), 400

    if len(polygon_rel) < 3:
        return jsonify({'error': 'Полигон должен содержать минимум 3 точки'}), 400

    polygon = [[int(x * w), int(y * h)] for x, y in polygon_rel]

    results = model(img)
    boxes = results[0].boxes

    all_classes = []
    for box in boxes:
        cls = int(box.cls.cpu().numpy()[0])
        all_classes.append(model.names[cls])

    person_boxes = [box for box in boxes if int(box.cls.cpu().numpy()[0]) == 0]
    total_people = len(person_boxes)

    # Комбинированная проверка
    on_crosswalk = 0
    for box in person_boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
        width = x2 - x1
        cx = int((x1 + x2) / 2)
        cy_center = int((y1 + y2) / 2)
        points = [
            (int(x1 + 0.2 * width), int(y2)),
            (cx, int(y2)),
            (int(x2 - 0.2 * width), int(y2))
        ]
        points.append((cx, cy_center))
        if any(point_in_polygon(px, py, polygon) for px, py in points):
            on_crosswalk += 1

    # Визуализация
    img_copy = img.copy()
    pts = np.array(polygon, dtype=np.int32)
    cv2.polylines(img_copy, [pts], True, (0, 255, 0), 3)

    for box in person_boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
        width = x2 - x1
        cx = int((x1 + x2) / 2)
        cy_center = int((y1 + y2) / 2)
        points = [
            (int(x1 + 0.2 * width), int(y2)),
            (cx, int(y2)),
            (int(x2 - 0.2 * width), int(y2))
        ]
        points.append((cx, cy_center))
        inside = any(point_in_polygon(px, py, polygon) for px, py in points)
        color = (0, 255, 0) if inside else (0, 0, 255)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        label = "on crosswalk" if inside else "outside"
        cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    result_path = 'static/result.jpg'
    cv2.imwrite(result_path, img_copy)

    unique_classes = list(set(all_classes))
    # Сохраняем в БД и получаем ID записи
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO requests (timestamp, filename, total_people, on_crosswalk, classes_detected, result_image_path)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), filename, total_people, on_crosswalk, json.dumps(unique_classes), result_path))
    record_id = c.lastrowid
    conn.commit()
    conn.close()

    return jsonify({
        'total_people': total_people,
        'on_crosswalk': on_crosswalk,
        'classes': unique_classes,
        'result_url': '/static/result.jpg',
        'record_id': record_id   # возвращаем ID для отчёта
    })

@app.route('/report')
def generate_report_all():
    records = get_all_history()
    buffer = build_pdf(records, "Отчёт по всем обработанным изображениям")
    return send_file(buffer, as_attachment=True, download_name='report_all.pdf', mimetype='application/pdf')

@app.route('/report/<int:record_id>')
def generate_report_single(record_id):
    record = get_record_by_id(record_id)
    if not record:
        return jsonify({'error': 'Запись не найдена'}), 404
    buffer = build_pdf([record], f"Отчёт по изображению: {record[2]}", include_stats=False)
    return send_file(buffer, as_attachment=True, download_name=f'report_{record_id}.pdf', mimetype='application/pdf')

# ---------- Запуск ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)