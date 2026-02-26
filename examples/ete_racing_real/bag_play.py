#!/usr/bin/env python
import rosbag
from cv_bridge import CvBridge
import cv2
import numpy as np

bag_path = "/home/suncc/Downloads/cc_demo1_624.bag"
# depth_topic = "/visfly/depth"
depth_topic = "/camera/depth/image_raw"
output_video = "demo1_onboard.avi"

bridge = CvBridge()
bag = rosbag.Bag(bag_path)
video_writer = None

for topic, msg, t in bag.read_messages(topics=[depth_topic]):
    try:
        depth_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        
        valid_mask = (depth_img != 0).astype(np.uint8)
        
        # 计算有效深度范围
        if np.any(valid_mask):
            min_val = np.min(depth_img[depth_img != 0])
            max_val = np.max(depth_img[depth_img != 0])
        else:
            min_val, max_val = 0, 1  # 防止除以零错误
            
        depth_normalized = np.zeros_like(depth_img, dtype=np.uint8)
        if max_val > min_val:  # 确保有有效的深度范围
            # 仅对有效区域进行归一化
            valid_pixels = depth_img[valid_mask == 1]
            normalized_pixels = 255 * (valid_pixels - min_val) / (max_val - min_val)
            depth_normalized[valid_mask == 1] = normalized_pixels.astype(np.uint8)
        
        depth_bgr = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
        
        if video_writer is None:
            h, w = depth_bgr.shape[:2]
            video_writer = cv2.VideoWriter(
                output_video, 
                cv2.VideoWriter_fourcc(*'FFV1'), 
                30,  # 帧率
                (w, h)
            )
        
        video_writer.write(depth_bgr)
        
        cv2.imshow("Depth Visualization", depth_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()

if video_writer:
    video_writer.release()
bag.close()
cv2.destroyAllWindows()
print("Video saved to", output_video)