from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image, ImageDraw
import numpy as np
import os
import io
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 加载 YOLOv5 模型
def load_yolo_model(model_path):
    from ultralytics import YOLO
    model = YOLO(model_path).cuda()
    return model

# 加载 Faster R-CNN 模型
def load_faster_rcnn_model(model_path):
    model = fasterrcnn_resnet50_fpn(weights=None)  # 初始化未加载权重的模型
    state_dict = torch.load(model_path)  # 加载权重文件
    model.load_state_dict(state_dict)  # 加载权重到模型
    model.eval()  # 设置为推理模式
    return model

# 使用预下载的模型路径
yolo_model_path = './models/yolov5su.pt'
faster_rcnn_model_path = './models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'

yolo_model = load_yolo_model(yolo_model_path)
faster_rcnn_model = load_faster_rcnn_model(faster_rcnn_model_path)

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
        method_type = request.form.get('method_type')

        try:
            if conversion_type == 'grayscale':
                result_image = convert_image(file_path, method_type)
            elif conversion_type == 'enhancement':
                result_image = enhance_image(file_path, method_type)
            elif conversion_type == 'segmentation':
                result_image = segment_image(file_path, method_type)
            elif conversion_type == 'object_detection':
                if method_type == 'yolo':
                    result_image = apply_yolo_detection(file_path)
                elif method_type == 'faster_rcnn':
                    result_image = apply_faster_rcnn_detection(file_path)
                else:
                    return jsonify({'error': '无效的目标检测方法！'}), 400
            else:
                return jsonify({'error': '无效的转换类型！'}), 400

            img_io = io.BytesIO()
            result_image.save(img_io, format='PNG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/png', download_name='result.png')
        except Exception as e:
            return jsonify({'error': f'处理失败: {str(e)}'}), 500

# 应用 YOLOv5 目标检测
def apply_yolo_detection(image_path):
    results = yolo_model(image_path)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for result in results[0].boxes.data:
        x1, y1, x2, y2, confidence, cls_id = result[:6]
        cls_name = yolo_model.names[int(cls_id)]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{cls_name} {confidence:.2f}", fill="red")

    return img

# 应用 Faster R-CNN 目标检测
def apply_faster_rcnn_detection(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0)  # 转换为 PyTorch 张量，增加批次维度

    with torch.no_grad():
        predictions = faster_rcnn_model(img_tensor)  # 推理

    # 提取检测结果
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    # 绘制边界框
    draw = ImageDraw.Draw(img)
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:  # 置信度阈值
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            draw.text((x1, y1), f"Class {label}: {score:.2f}", fill="blue")

    return img

if __name__ == '__main__':
    app.run(debug=True)
