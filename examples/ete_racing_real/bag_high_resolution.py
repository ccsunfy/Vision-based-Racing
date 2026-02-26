#!/usr/bin/env python
import rosbag
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import time

# 配置参数
bag_path = "/home/suncc/demo2_fast_6_4_2.bag"
depth_topic = "/visfly/depth"
output_dir = "depth_output"
os.makedirs(output_dir, exist_ok=True)

# 高清视频参数
output_video = os.path.join(output_dir, "depth_video_hd.mp4")
frame_rate = 30  # 帧率
resolution = (1280, 720)  # 目标分辨率
quality = "high"  # 可选: "medium", "high", "lossless"

# 初始化
bridge = CvBridge()
bag = rosbag.Bag(bag_path)
start_time = time.time()

# 方法1: 保存原始深度图序列 (最高质量)
def save_depth_sequence():
    """保存原始深度图序列为PNG文件"""
    frame_count = 0
    for topic, msg, t in bag.read_messages(topics=[depth_topic]):
        try:
            depth_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            # 保存原始深度图 (16位PNG)
            frame_path = os.path.join(output_dir, f"depth_frame_{frame_count:06d}.png")
            cv2.imwrite(frame_path, depth_img)
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"已保存 {frame_count} 帧...")
                
        except Exception as e:
            print(f"处理帧错误: {e}")
    
    print(f"已保存 {frame_count} 帧原始深度图到 {output_dir}")
    return frame_count

# 方法2: 直接生成高质量视频
def create_hd_video():
    """直接生成高质量深度视频"""
    # 配置视频编码器
    if quality == "lossless":
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # 无损编码
    elif quality == "high":
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # 高质量H.264
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 中等质量
    
    video_writer = None
    frame_count = 0
    
    for topic, msg, t in bag.read_messages(topics=[depth_topic]):
        try:
            depth_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            # 处理无效深度值
            valid_mask = (depth_img != 0).astype(np.uint8)
            
            # 计算有效深度范围
            if np.any(valid_mask):
                min_val = np.min(depth_img[depth_img != 0])
                max_val = np.max(depth_img[depth_img != 0])
            else:
                min_val, max_val = 0, 1
            
            # 归一化到0-255
            depth_normalized = np.zeros_like(depth_img, dtype=np.uint8)
            if max_val > min_val:
                valid_pixels = depth_img[valid_mask == 1]
                normalized_pixels = 255 * (valid_pixels - min_val) / (max_val - min_val)
                depth_normalized[valid_mask == 1] = normalized_pixels.astype(np.uint8)
            
            # 转换为三通道灰度图像
            depth_bgr = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
            
            # 调整分辨率 (如果需要)
            if resolution:
                depth_bgr = cv2.resize(depth_bgr, resolution, interpolation=cv2.INTER_LANCZOS4)
            
            # 初始化视频写入器
            if video_writer is None:
                h, w = depth_bgr.shape[:2]
                video_writer = cv2.VideoWriter(
                    output_video, 
                    fourcc, 
                    frame_rate, 
                    (w, h)
                )
            
            video_writer.write(depth_bgr)
            frame_count += 1
            
            # 进度显示
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"已处理 {frame_count} 帧 | 耗时: {elapsed:.1f}秒")
        
        except Exception as e:
            print(f"处理帧错误: {e}")
    
    if video_writer:
        video_writer.release()
    
    print(f"已生成高清视频: {output_video} ({frame_count}帧)")
    return frame_count

# 主程序
if __name__ == "__main__":
    print("开始处理深度图...")
    
    # 选择处理方法 (推荐使用序列保存方法)
    method = input("选择处理方法: (1) 保存原始序列 (最高质量) (2) 生成高清视频 [1/2]: ") or "1"
    
    if method == "1":
        frame_count = save_depth_sequence()
        
        # 提供FFmpeg命令用于转换序列为视频
        ffmpeg_cmd = f"ffmpeg -framerate {frame_rate} -i {output_dir}/depth_frame_%06d.png " \
                     f"-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p " \
                     f"{output_dir}/depth_video_h264_lossless.mp4"
        
        print("\n已保存原始深度图序列。可以使用以下命令生成最高质量视频:")
        print(ffmpeg_cmd)
        
    else:
        frame_count = create_hd_video()
    
    # 处理统计
    elapsed = time.time() - start_time
    print(f"\n处理完成! 总帧数: {frame_count} | 总耗时: {elapsed:.1f}秒")
    print(f"平均帧率: {frame_count/elapsed:.1f} FPS")
    
    bag.close()