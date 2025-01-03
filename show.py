from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import os
import io
import cv2
import math
import pywt
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.autograd import Variable
from model import AODnet  # 导入 AOD-Net 模型定义
from celery import Celery
import logging

# Flask 应用初始化
app = Flask(__name__)

# 配置上传和结果文件夹
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Celery 配置
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# 初始化 Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载 AOD-Net 模型
def load_aod_net(model_path):
    net = torch.load(model_path)
    return net

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
aod_net_model_path = './models/AOD_net_epoch_relu_best.pth'

yolo_model = load_yolo_model(yolo_model_path)
faster_rcnn_model = load_faster_rcnn_model(faster_rcnn_model_path)
# aod_net_model = load_aod_net(aod_net_model_path)

# 图像处理函数
def convert_image(image_path, grayscale_type):
    img = Image.open(image_path)
    img_array = np.array(img)
    if grayscale_type == 'luminosity':
        gray = 0.21 * img_array[:, :, 0] + 0.72 * img_array[:, :, 1] + 0.07 * img_array[:, :, 2]
    elif grayscale_type == 'average':
        gray = img_array.mean(axis=2)
    elif grayscale_type == 'desaturation':
        gray = (img_array.max(axis=2) + img_array.min(axis=2)) / 2
    elif grayscale_type == 'max_decomposition':
        gray = img_array.max(axis=2)
    elif grayscale_type == 'min_decomposition':
        gray = img_array.min(axis=2)
    elif grayscale_type == 'custom_weights':
        gray = 0.5 * img_array[:, :, 0] + 0.3 * img_array[:, :, 1] + 0.2 * img_array[:, :, 2]
    else:
        raise ValueError("无效的灰度化方法")
    return Image.fromarray(gray.astype(np.uint8))

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
        gamma = 2.2
        img_array = np.array(img) / 255.0
        img_corrected = np.power(img_array, gamma) * 255
        return Image.fromarray(img_corrected.astype(np.uint8))
    elif enhancement_type == 'color_enhancement':
        img_array = np.array(img)
        img_enhanced = np.clip(img_array * 1.2, 0, 255).astype(np.uint8)
        return Image.fromarray(img_enhanced)
    elif enhancement_type == 'sharpening':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img_array = np.array(img)
        img_sharpened = Image.fromarray(cv2.filter2D(img_array, -1, kernel))
        return img_sharpened
    elif enhancement_type == 'high_pass_filter':
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        img_array = np.array(img.convert('L'))
        img_filtered = cv2.filter2D(img_array, -1, kernel)
        img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
        return Image.fromarray(img_filtered)
    elif enhancement_type == 'low_pass_filter':
        kernel = np.ones((5, 5), np.float32) / 25
        img_array = np.array(img.convert('L'))
        img_filtered = cv2.filter2D(img_array, -1, kernel)
        img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
        return Image.fromarray(img_filtered)
    elif enhancement_type == 'wavelet_transform':
        img_array = np.array(img.convert('L'))
        coeffs = pywt.wavedec2(img_array, 'haar', level=2)
        coeffs[0] = np.zeros_like(coeffs[0])
        img_wavelet = pywt.waverec2(coeffs, 'haar')
        img_wavelet = np.clip(img_wavelet, 0, 255).astype(np.uint8)
        return Image.fromarray(img_wavelet)
    else:
        raise ValueError("无效的增强方法")

def segment_image(image_path, segmentation_type):
    img = Image.open(image_path)
    img_array = np.array(img.convert('L'))
    if segmentation_type == 'thresholding':
        _, segmented = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)
        img = Image.fromarray(segmented)
    elif segmentation_type == 'region_growing':
        def region_grow(img, seed, threshold=30):
            visited = np.zeros_like(img, dtype=bool)
            region = np.zeros_like(img, dtype=np.uint8)
            stack = [seed]
            seed_value = img[seed]
            while stack:
                x, y = stack.pop()
                if not visited[x, y] and abs(int(img[x, y]) - int(seed_value)) <= threshold:
                    visited[x, y] = True
                    region[x, y] = 255
                    for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                        if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1] and not visited[nx, ny]:
                            stack.append((nx, ny))
            return region
        img_smoothed = cv2.GaussianBlur(img_array, (5, 5), 0)
        seed = (img_smoothed.shape[0] // 2, img_smoothed.shape[1] // 2)
        segmented = region_grow(img_smoothed, seed, threshold=30)
    elif segmentation_type == 'kmeans':
        img_array = img_array.reshape(-1, 1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(img_array, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        segmented = centers[labels.flatten()].reshape(img.size[::-1]).astype(np.uint8)
        img = Image.fromarray(segmented)
    elif segmentation_type == 'sobel':
        sobel_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edge = (edge / edge.max() * 255).astype(np.uint8)
        img = Image.fromarray(edge)
    elif segmentation_type == 'laplacian':
        edge = cv2.Laplacian(img_array, cv2.CV_64F, ksize=3)
        edge = cv2.convertScaleAbs(edge)
        edge = cv2.equalizeHist(edge)
        img = Image.fromarray(edge)
    elif segmentation_type == 'canny':
        edge = cv2.Canny(img_array, 100, 200)
        img = Image.fromarray(edge)
    elif segmentation_type == 'prewitt':
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewitt_x = cv2.filter2D(img_array, -1, kernel_x)
        prewitt_y = cv2.filter2D(img_array, -1, kernel_y)
        edge = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)
        edge = (edge / edge.max() * 255).astype(np.uint8)
        img = Image.fromarray(edge)
    else:
        raise ValueError("无效的分割方法")
    return img

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

def apply_faster_rcnn_detection(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0)
    with torch.no_grad():
        predictions = faster_rcnn_model(img_tensor)
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    draw = ImageDraw.Draw(img)
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            draw.text((x1, y1), f"Class {label}: {score:.2f}", fill="blue")
    return img

def dehaze_dark_channel(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_channel, kernel)
    h, w = dark_channel.shape
    num_pixels = int(0.001 * h * w)
    indices = np.unravel_index(np.argsort(dark_channel.ravel())[-num_pixels:], dark_channel.shape)
    atmospheric_light = np.mean(image[indices], axis=0)
    norm_image = image / atmospheric_light
    transmission = 1 - 0.95 * np.min(norm_image, axis=2)
    transmission = np.clip(transmission, 0.1, 1)
    recovered_image = (image - atmospheric_light) / transmission[:, :, np.newaxis] + atmospheric_light
    recovered_image = np.clip(recovered_image, 0, 255).astype(np.uint8)
    return Image.fromarray(recovered_image)

def dehaze_aod_net(image_path):
    net = load_aod_net(aod_net_model_path)
    net = net.cuda()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze_(0)
    val_img = Variable(img_tensor)
    if torch.cuda.is_available():
        val_img = val_img.cuda()
    prediction = net(val_img)
    prediction = prediction.data.cpu().numpy().squeeze().transpose((1, 2, 0))
    prediction = (prediction * 255.0).astype(np.uint8)
    return Image.fromarray(prediction)

# Celery 任务
@celery.task
def process_image_async(file_path, conversion_type, method_type):
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
                raise ValueError("无效的目标检测方法")
        elif conversion_type == 'dehaze':
            if method_type == 'dark_channel':
                result_image = dehaze_dark_channel(file_path)
            elif method_type == 'aod_net':
                result_image = dehaze_aod_net(file_path)
            else:
                raise ValueError("无效的去雾方法")
        else:
            raise ValueError("无效的转换类型")

        result_path = os.path.join(RESULT_FOLDER, os.path.basename(file_path))
        result_image.save(result_path)
        return result_path
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        raise

# 路由
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

        # 异步处理图像
        task = process_image_async.delay(file_path, conversion_type, method_type)
        return jsonify({'task_id': task.id}), 202

@app.route('/status/<task_id>')
def task_status(task_id):
    task = process_image_async.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {'state': task.state, 'status': '处理中...'}
    elif task.state == 'SUCCESS':
        response = {'state': task.state, 'result': task.result}
    else:
        response = {'state': task.state, 'status': str(task.info)}
    return jsonify(response)

@app.route('/result/<filename>')
def get_result(filename):
    result_path = os.path.join(RESULT_FOLDER, filename)
    if os.path.exists(result_path):
        return send_file(result_path, mimetype='image/png')
    else:
        return jsonify({'error': '结果文件不存在！'}), 404

if __name__ == '__main__':
    app.run(debug=True)