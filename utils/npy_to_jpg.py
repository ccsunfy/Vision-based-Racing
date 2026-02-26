import os
import numpy as np
from PIL import Image

def npy_to_image(npy_file, output_folder):
    # 加载 .npy 文件
    data = np.load(npy_file)
    
    # 将数据归一化并转换为8位灰度图像
    image_data = (data / np.max(data) * 255).astype(np.uint8)
    
    # 去除多余的维度
    image_data = np.squeeze(image_data)
    
    # 确保数组形状为(height, width)
    if image_data.ndim == 3:
        image_data = image_data[0]
    
    # 创建PIL图像对象
    image = Image.fromarray(image_data)
    
    # 构建输出文件路径
    base_name = os.path.basename(npy_file).replace('.npy', '.jpg')
    output_path = os.path.join(output_folder, base_name)
    
    # 保存为JPEG格式
    image.save(output_path)
    print(f"Saved image to {output_path}")

if __name__ == "__main__":
    npy_file = "utils/depth_image/depth_image_0.npy"  # 替换为您的 .npy 文件路径
    output_folder = "utils/depth_image"  # 替换为您的输出文件夹路径
    npy_to_image(npy_file, output_folder)