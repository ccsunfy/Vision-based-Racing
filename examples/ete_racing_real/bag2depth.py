#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from tqdm import tqdm

def convert_depth_images(bag_file, topic_name, output_dir, min_depth, max_depth, scaling_factor):
    """
    从ROS bag文件提取深度图并保存为PNG图片
    
    参数:
        bag_file: ROS bag文件路径
        topic_name: 要提取的深度图话题名称
        output_dir: 输出目录
        min_depth: 最小有效深度值(m)
        max_depth: 最大有效深度值(m)
        scaling_factor: 深度值缩放因子
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化OpenCV转换桥
    bridge = CvBridge()
    
    # 打开bag文件
    bag = rosbag.Bag(bag_file, "r")
    
    # 获取话题消息总数
    msg_count = bag.get_message_count(topic_filters=[topic_name])
    
    print(f"正在处理文件: {bag_file}")
    print(f"提取话题: {topic_name}")
    print(f"找到 {msg_count} 条深度图消息")
    print(f"输出目录: {output_dir}")
    print(f"深度范围: {min_depth}-{max_depth} meters")
    
    try:
        # 使用tqdm创建进度条
        with tqdm(total=msg_count, unit="frame") as pbar:
            # 遍历所有消息
            for topic, msg, timestamp in bag.read_messages(topics=[topic_name]):
                try:
                    # 将ROS图像消息转换为OpenCV图像
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                    
                    # 深度图处理
                    if msg.encoding == "16UC1":
                        # 处理16位无符号整数深度图
                        depth_image = cv_image.astype(np.float32) * scaling_factor
                    elif msg.encoding == "32FC1":
                        # 处理32位浮点深度图
                        depth_image = cv_image
                    else:
                        print(f"警告: 不支持的深度图格式 {msg.encoding}")
                        continue
                    
                    # 应用深度范围过滤
                    depth_image[depth_image < min_depth] = 0
                    depth_image[depth_image > max_depth] = max_depth
                    
                    # 归一化深度值到0-65535范围 (16位)
                    depth_normalized = ((depth_image - min_depth) / 
                                      (max_depth - min_depth) * 65535).astype(np.uint16)
                    
                    # 生成文件名
                    timestamp_ns = timestamp.to_nsec()
                    filename = os.path.join(output_dir, f"depth_{timestamp_ns}.png")
                    
                    # 保存为16位PNG图像 (保留完整深度信息)
                    cv2.imwrite(filename, depth_normalized)
                    
                    # 更新进度条
                    pbar.update(1)
                    
                except CvBridgeError as e:
                    print(f"转换错误: {e}")
                    
    finally:
        bag.close()
        print("\n处理完成!")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='从ROS bag提取深度图并保存为PNG')
    parser.add_argument('bag_file', type=str, help='ROS bag文件路径')
    parser.add_argument('topic', type=str, help='深度图话题名称')
    parser.add_argument('output_dir', type=str, help='输出目录')
    parser.add_argument('--min_depth', type=float, default=0.1, 
                        help='最小有效深度值(m), 默认0.1')
    parser.add_argument('--max_depth', type=float, default=10.0, 
                        help='最大有效深度值(m), 默认10.0')
    parser.add_argument('--scaling', type=float, default=0.001, 
                        help='深度值缩放因子 (16UC1格式使用), 默认0.001')
    
    args = parser.parse_args()
    
    # 运行转换函数
    convert_depth_images(
        args.bag_file,
        args.topic,
        args.output_dir,
        args.min_depth,
        args.max_depth,
        args.scaling
    )