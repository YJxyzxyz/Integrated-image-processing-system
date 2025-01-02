import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 读取图像并转换为灰度图
image_path = './uploads/test.jpg'  # 图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"图像文件 {image_path} 未找到，请检查路径是否正确。")

# 初始化参数
initial_threshold = 100  # 初始边缘强度阈值
initial_blur = 5  # 初始模糊核大小

# 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(left=0.1, bottom=0.3)  # 调整布局，为滑块留出空间

# 显示原始图像
ax1.set_title('Original Image')
im1 = ax1.imshow(image, cmap='gray')

# 显示边缘检测结果
ax2.set_title('Edge Detection')
edges = np.zeros_like(image, dtype=np.uint8)
im2 = ax2.imshow(edges, cmap='gray', vmin=0, vmax=255)  # 设置显示范围


# 更新函数，用于根据参数重新计算边缘检测结果
def update(val):
    # 获取当前滑块的值
    threshold = slider_threshold.val
    blur_size = int(slider_blur.val)

    # 对图像进行高斯模糊
    blurred_image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)

    # 使用Sobel算子计算梯度
    gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算边缘强度
    edges = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # 根据阈值进行二值化
    edges = np.uint8(edges > threshold) * 255

    # 打印调试信息
    print(f"Threshold: {threshold}, Blur Size: {blur_size}")
    print(f"Edge intensity range: {edges.min()} to {edges.max()}")

    # 更新边缘检测结果
    im2.set_data(edges)
    fig.canvas.draw_idle()


# 创建阈值滑块
ax_threshold = plt.axes([0.2, 0.15, 0.6, 0.03])  # 滑块位置和大小
slider_threshold = Slider(ax_threshold, 'Threshold', 0, 255, valinit=initial_threshold)
slider_threshold.on_changed(update)

# 创建模糊核大小滑块
ax_blur = plt.axes([0.2, 0.1, 0.6, 0.03])  # 滑块位置和大小
slider_blur = Slider(ax_blur, 'Blur Size', 1, 15, valinit=initial_blur, valstep=2)
slider_blur.on_changed(update)

# 初始更新
update(None)

# 显示界面
plt.show()