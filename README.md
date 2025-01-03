# Integrated image processing system

欢迎访问我的综合图像处理系统项目！该项目旨在系统展示多种经典和深度学习图像处理算法的应用与效果。用户可以通过友好的界面深入体验图像增强、滤波、边缘检测、特征提取等算法以及去雾，图像风格迁移等AI模型的实际效果。无论您是图像处理领域的研究者、开发者，还是技术爱好者，本项目都为您提供了一个探索和实践的平台。😺😺😺

该项目正在开发中，敬请期待

2024.12.11

实现灰度化，简单边缘检测

实现图像增强（直方图均衡化、对比度拉伸、伽马矫正、色彩增强、锐化、高通滤波、低通滤波、小波变换）



2024.12.12

实现多种灰度化方法

- `luminosity`：基于加权平均。
- `average`：简单平均。
- `desaturation`：最大和最小值的平均。
- `max_decomposition`：最大值作为灰度值。
- `min_decomposition`：最小值作为灰度值。
- `custom_weights`：基于自定义的权重（50%红色、30%绿色、20%蓝色）

实现多种边缘检测方法

- Sobel算子
- Laplacian算子
- Canny边缘检测
- Prewitt算子

初步优化前端形式，重新调整图像分割类别，新增区域生长 Kmeans分类 阈值分割

新增深度学习方法--YOLOv5目标检测(yolov5su.pt)（[Releases · ultralytics/yolov5](https://github.com/ultralytics/yolov5/releases)）



2024.12.13

新增深度学习方法--Fast R-CNN目标检测模型(fasterrcnn_resnet50_fpn_coco-258fb6c6.pth)（https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth）

2024.12.16

新增去雾 暗通道 AOD

2024.12.17

AOD由于原作者的版本为3.6，我使用的python版本是3.9.11，需重构加载函数，并对模型输出的图片重新调整过一化处理过程，原处理方式输出图片带有灰蒙蒙一片，待解决

2024.12.23

AOD反归一化问题已解决，目前可以正常实现去雾，使用的原论文预训练模型AOD_net_epoch_relu_best.pth（[AODnet-by-pytorch/model_pretrained at master · weberwcwei/AODnet-by-pytorch](https://github.com/weberwcwei/AODnet-by-pytorch/tree/master/model_pretrained)）

2024.12.31 → 2025.1.1

Happy 2025

2025.1.2

在test.py中尝试对边缘检测的部分添加滑块，用户调整滑块可以实现不同的效果

添加滑块动态参数范围

2025.1.3 

在test.py中加入供用户选择文件保存路径的功能

在show.py中进行Redis分布式文件管理系统尝试

初步优化前端界面：

- 加载动画：在表单提交时显示一个旋转的加载动画，处理完成后隐藏
- 卡片悬停效果：当鼠标悬停在卡片上时，卡片会轻微上移，增加互动感
- 按钮动画：按钮在悬停时会放大，点击时会缩小，增加点击反馈
- 结果渐显效果：转换结果会以渐显的方式显示，增加视觉吸引力
- 图片悬停效果：转换后的图片在悬停时会轻微放大，增加互动性

修改前后端逻辑，现在用户在选择第一次方法后可以重新选择方法
