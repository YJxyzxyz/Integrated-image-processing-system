from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import pywt
import os
import io
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件被上传！'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': '没有选择文件！'}), 400

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        conversion_type = request.form.get('conversion_type')
        enhancement_type = request.form.get('enhancement_type')

        if conversion_type in ['edge_detection', 'grayscale']:
            result_image = convert_image(file_path, conversion_type)
        elif conversion_type == 'enhancement':
            result_image = enhance_image(file_path, enhancement_type)

        img_io = io.BytesIO()
        result_image.save(img_io, format='PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png', download_name='result.png')

def convert_image(image_path, conversion_type):
    img = Image.open(image_path)

    if conversion_type == 'edge_detection':
        img = img.filter(ImageFilter.FIND_EDGES)
    elif conversion_type == 'grayscale':
        img = img.convert('L')

    return img

def enhance_image(image_path, enhancement_type):
    img = Image.open(image_path)

    if enhancement_type == 'histogram_equalization':
        img = img.convert('L')
        img_array = np.array(img)
        img_eq = ImageOps.equalize(Image.fromarray(img_array))
        return img_eq
    elif enhancement_type == 'contrast_stretching':
        img_array = np.array(img)
        p2, p98 = np.percentile(img_array, (2, 98))
        img_cs = np.clip((img_array - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_cs)
    elif enhancement_type == 'gamma_correction':
        gamma = 2.2  # 伽马值
        img_array = np.array(img) / 255.0
        img_corrected = np.power(img_array, gamma) * 255
        return Image.fromarray(img_corrected.astype(np.uint8))
    elif enhancement_type == 'color_enhancement':
        img_array = np.array(img)
        img_enhanced = np.clip(img_array * 1.2, 0, 255).astype(np.uint8)  # 色彩增强
        return Image.fromarray(img_enhanced)
    elif enhancement_type == 'sharpening':
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img_array = np.array(img)
        img_sharpened = Image.fromarray(cv2.filter2D(img_array, -1, kernel))  # 使用OpenCV进行锐化
        return img_sharpened
    elif enhancement_type == 'high_pass_filter':
        img_array = np.array(img)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img_filtered = Image.fromarray(cv2.filter2D(img_array, -1, kernel))  # 高通滤波
        return img_filtered
    elif enhancement_type == 'low_pass_filter':
        img_array = np.array(img)
        kernel = np.ones((5, 5), np.float32) / 25  # 简单的平均滤波器
        img_filtered = Image.fromarray(cv2.filter2D(img_array, -1, kernel))  # 低通滤波
        return img_filtered
    elif enhancement_type == 'wavelet_transform':
        img_array = np.array(img)
        coeffs = pywt.wavedec2(img_array, 'haar')
        img_wavelet = pywt.waverec2(coeffs, 'haar')
        return Image.fromarray(np.clip(img_wavelet, 0, 255).astype(np.uint8))

if __name__ == '__main__':
    app.run(debug=True)
