from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image
import os
import io

app = Flask(__name__)

# 设置上传文件的保存路径
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

        # 获取转换选项
        conversion_type = request.form.get('conversion_type')

        # 进行相应的图像转换处理
        result_image = convert_image(file_path, conversion_type)

        # 将转换后的图像保存到BytesIO中，以便直接返回给用户
        img_io = io.BytesIO()
        result_image.save(img_io, format='PNG')
        img_io.seek(0)

        # 生成可访问的图片URL
        return send_file(img_io, mimetype='image/png')


def convert_image(image_path, conversion_type):
    img = Image.open(image_path)

    if conversion_type == 'pixelate':
        # 像素化处理
        img = img.resize((16, 16), Image.NEAREST)
        img = img.resize(img.size, Image.NEAREST)
    elif conversion_type == 'grayscale':
        # 灰度处理
        img = img.convert('L')

    return img


if __name__ == '__main__':
    app.run(debug=True)
