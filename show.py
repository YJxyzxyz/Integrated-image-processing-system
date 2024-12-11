import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 读取图像
image = cv2.imread('./results/1.png', cv2.IMREAD_GRAYSCALE)  # 请替换为你的图片路径
image = cv2.resize(image, (400, 400))  # 调整图像大小以便展示

# Sobel算子
sobel_x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
sobel_y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

# 创建动画
fig, ax = plt.subplots(figsize=(6, 6))

# 初始化动画帧
def update(frame):
    ax.clear()

    # 当前帧的进度
    progress = frame / 30

    # 使用Sobel算子逐步处理图像
    if frame < 15:
        # 逐步应用X方向的Sobel算子
        sobel_image = cv2.filter2D(image, -1, sobel_x_kernel)
        current_image = cv2.convertScaleAbs(sobel_image * (progress / 1.0))
        ax.imshow(current_image, cmap='gray')
        ax.set_title('逐步应用Sobel X方向 (进度 {:.2f})'.format(progress))
    elif frame < 30:
        # 逐步应用Y方向的Sobel算子
        sobel_image = cv2.filter2D(image, -1, sobel_y_kernel)
        current_image = cv2.convertScaleAbs(sobel_image * ((progress - 0.5) / 0.5))
        ax.imshow(current_image, cmap='gray')
        ax.set_title('逐步应用Sobel Y方向 (进度 {:.2f})'.format(progress - 0.5))
    else:
        # 最终合成图像
        sobel_x = cv2.filter2D(image.astype(np.float32), -1, sobel_x_kernel)
        sobel_y = cv2.filter2D(image.astype(np.float32), -1, sobel_y_kernel)

        # 确保sobel_x和sobel_y的尺寸和类型一致
        if sobel_x.shape != sobel_y.shape:
            raise ValueError("sobel_x和sobel_y的尺寸不匹配")

        combined_image = cv2.magnitude(sobel_x, sobel_y)
        current_image = cv2.convertScaleAbs(combined_image)
        ax.imshow(current_image, cmap='gray')
        ax.set_title('结合的Sobel边缘')

    ax.axis('off')

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=60, repeat=True, interval=100)

plt.tight_layout()
plt.show()
