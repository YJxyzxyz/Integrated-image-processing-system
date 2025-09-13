import cv2
import numpy as np
import matplotlib.pyplot as plt


def select_corresponding_points(img1, img2, num_points=8):
    """
    在两幅图像上手动选择对应点

    参数:
        img1: 左图 (numpy array)
        img2: 右图 (numpy array)
        num_points: 需要选择的点对数，默认8

    返回:
        pts1: (num_points, 2) 左图坐标
        pts2: (num_points, 2) 右图坐标
    """
    pts1, pts2 = [], []

    # 先在左图选择 num_points 个点
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap="gray")
    plt.title(f"Left Image: Select {num_points} points")
    plt.axis("on")
    pts1 = plt.ginput(num_points, timeout=0)  # 点击左图
    pts1 = np.array(pts1, dtype=np.float32)

    # 再在右图选择对应点
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap="gray")
    plt.title(f"Right Image: Select {num_points} corresponding points")
    plt.axis("on")
    pts2 = plt.ginput(num_points, timeout=0)  # 点击右图
    pts2 = np.array(pts2, dtype=np.float32)

    plt.close()

    return pts1, pts2


# =================== 使用示例 ===================
if __name__ == "__main__":
    # 读取图像
    img1 = cv2.imread("GT1.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("GT2.jpg", cv2.IMREAD_GRAYSCALE)

    pts1, pts2 = select_corresponding_points(img1, img2, num_points=8)
    print("左图点坐标：\n", pts1)
    print("右图点坐标：\n", pts2)
