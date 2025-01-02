import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# 读取图像并转换为灰度图
image_path = './uploads/test.jpg'  # 图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"图像文件 {image_path} 未找到，请检查路径是否正确。")

# 初始化参数
initial_operator = 'Sobel'  # 初始边缘检测算子
initial_params = {
    'Sobel': {'threshold': 100, 'blur_size': 5},
    'Canny': {'threshold': 100, 'blur_size': 5},
    'Prewitt': {'threshold': 100, 'blur_size': 5},
    'Laplacian': {'threshold': 10, 'blur_size': 5}
}

# 参数范围配置
param_ranges = {
    'Sobel': {'threshold': (0, 255, 1), 'blur_size': (1, 15, 2)},
    'Canny': {'threshold': (0, 255, 1), 'blur_size': (1, 15, 2)},
    'Prewitt': {'threshold': (0, 255, 1), 'blur_size': (1, 15, 2)},
    'Laplacian': {'threshold': (0, 50, 1), 'blur_size': (1, 15, 2)}
}

# 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(left=0.1, bottom=0.4)  # 调整布局，为滑块和下拉菜单留出空间

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
    operator = radio.value_selected  # 获取当前选择的算子

    # 对图像进行高斯模糊
    blurred_image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)

    # 根据选择的算子计算边缘
    if operator == 'Sobel':
        gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        edges = np.uint8(edges > threshold) * 255
    elif operator == 'Canny':
        edges = cv2.Canny(blurred_image, threshold, threshold * 2)
    elif operator == 'Prewitt':
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        gradient_x = cv2.filter2D(blurred_image, cv2.CV_64F, kernel_x)
        gradient_y = cv2.filter2D(blurred_image, cv2.CV_64F, kernel_y)
        edges = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        edges = np.uint8(edges > threshold) * 255
    elif operator == 'Laplacian':
        edges = cv2.Laplacian(blurred_image, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges) > threshold) * 255  # 取绝对值并二值化

    # 更新边缘检测结果
    im2.set_data(edges)
    fig.canvas.draw_idle()


# 创建阈值滑块
ax_threshold = plt.axes([0.2, 0.25, 0.6, 0.03])  # 滑块位置和大小
slider_threshold = Slider(
    ax_threshold, 'Threshold',
    param_ranges[initial_operator]['threshold'][0],  # 最小值
    param_ranges[initial_operator]['threshold'][1],  # 最大值
    valinit=initial_params[initial_operator]['threshold'],  # 初始值
    valstep=param_ranges[initial_operator]['threshold'][2]  # 步长
)
slider_threshold.on_changed(update)

# 创建模糊核大小滑块
ax_blur = plt.axes([0.2, 0.2, 0.6, 0.03])  # 滑块位置和大小
slider_blur = Slider(
    ax_blur, 'Blur Size',
    param_ranges[initial_operator]['blur_size'][0],  # 最小值
    param_ranges[initial_operator]['blur_size'][1],  # 最大值
    valinit=initial_params[initial_operator]['blur_size'],  # 初始值
    valstep=param_ranges[initial_operator]['blur_size'][2]  # 步长
)
slider_blur.on_changed(update)

# 创建算子选择下拉菜单
ax_radio = plt.axes([0.2, 0.05, 0.6, 0.1])  # 下拉菜单位置和大小
radio = RadioButtons(ax_radio, ('Sobel', 'Canny', 'Prewitt', 'Laplacian'), active=0)


# 切换算子时更新滑块范围和初始值
def update_sliders(label):
    # 更新滑块范围
    slider_threshold.set_val(initial_params[label]['threshold'])
    slider_threshold.set_min(param_ranges[label]['threshold'][0])
    slider_threshold.set_max(param_ranges[label]['threshold'][1])
    slider_threshold.set_valstep(param_ranges[label]['threshold'][2])

    slider_blur.set_val(initial_params[label]['blur_size'])
    slider_blur.set_min(param_ranges[label]['blur_size'][0])
    slider_blur.set_max(param_ranges[label]['blur_size'][1])
    slider_blur.set_valstep(param_ranges[label]['blur_size'][2])

    # 更新边缘检测结果
    update(None)


radio.on_clicked(update_sliders)

# 初始更新
update(None)

# 显示界面
plt.show()