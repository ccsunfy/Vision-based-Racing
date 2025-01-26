import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 加载深度图像
depth_image_path = "utils/depth_image/depth_image_0.jpg"
depth_image = np.load(depth_image_path)

# 将深度图像归一化到0-255范围
depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

# 将深度图像转换为8位无符号整数类型
depth_image_8bit = depth_image_normalized.astype(np.uint8)

# 将深度图像转换为伪彩色图像
pseudo_color_image = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)

# 显示伪彩色图像
plt.imshow(pseudo_color_image)
plt.title("Pseudo-color Depth Image")
plt.axis('off')
plt.show()

# 使用YOLO进行检测（假设已经加载了YOLO模型）
# yolo_model.detect(pseudo_color_image)