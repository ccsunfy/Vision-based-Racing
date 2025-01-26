#!/usr/bin/env python3

import sys
import os
import numpy as np
import torch
import time
import habitat_sim

sys.path.append(os.getcwd())
import torch as th
from envs.CircleEnv_yolo import CircleEnv
from utils.launcher import rl_parser, training_params
from utils.type import Uniform
from PIL import Image
from utils.maths import pixel_to_body,quat_to_matrix

import cv2
from ultralytics import YOLO

save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/rgb_image/"

args = rl_parser().parse_args()
training_params["num_env"] = 1
training_params["learning_step"] = 1e7
training_params["comment"] = args.comment
training_params["max_episode_steps"] = 256
training_params["n_steps"] = training_params["max_episode_steps"]
training_params["batch_size"] = training_params["num_env"] * training_params["n_steps"]
training_params["learning_rate"] = 1e-3

# scene_path = "datasets/spy_datasets/configs/garage_simple_l_medium"
scene_path = "datasets/spy_datasets/configs/cross_circle"
random_kwargs = {
    "state_generator_kwargs": [{
        "position": Uniform(mean=th.tensor([1.0, 0.0, 0.1]), half=th.tensor([0.0, 0.1, 0.1]))
    }]
}
# random_kwargs = {
#     "state_generator_kwargs": [{
#         "position": [0.,0.,0.]
#     }]
# }
latent_dim = 25
latent_dim = None

origin_position = np.array(random_kwargs.get("state_generator_kwargs")[0].get("position").mean)
# mean_coords = random_position["mean"]
# half_coords = random_position["half"]
# origin_position = mean_coords+half_coords
print(f"The origin position is:{origin_position}")

model = YOLO("best.pt")  # 预训练模型路径

env = CircleEnv(num_agent_per_scene=training_params["num_env"],
                        random_kwargs=random_kwargs,
                        visual=True,
                        max_episode_steps=training_params["max_episode_steps"],
                        scene_kwargs={
                             "path": scene_path,
                         },
                        # dynamics_kwargs={
                        #     "dt": 0.02,
                        #     "ctrl_dt": 0.02,
                        #     # "action_type":"velocity",
                        # },
                        # requires_grad=True,
                        latent_dim=latent_dim
                        )

def main():
    # scenc_max_depth = 20.0 #定义场景最大深度
    obs = env.get_observation()
    # obs1 = env.get_observation()
    rgb_image = obs["color"]
    depth_image = obs["depth"]
    print(depth_image)
    
    
    rgb_image = np.squeeze(rgb_image).astype(np.uint8)
    rgb_image = np.transpose(rgb_image, (1, 2, 0))
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
    
    
    depth_show = (depth_image / np.max(depth_image) ).astype(np.float16) #用场景最大深度20m进行归一化
    
    depth_image = np.squeeze(depth_image)
    depth_show = (depth_image / np.max(depth_image) * 255).astype(np.uint8)
    # depth_image = np.transpose(depth_image, (1, 2, 0))
    # rgb_image_pil = Image.fromarray(rgb_image)
    
    results = model(rgb_image)
    
    #计算帧率
    # end_time = time.time()
    # processing_time = end_time - start_time
    # fps = 1 / processing_time
    # print(f"FPS: {fps:.2f}")
    
    # 在图像上绘制检测结果
    for result in results:
        for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = box.cls[0]
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(rgb_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                #计算并打印中心坐标
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = (x2 - x1) // 2 - 5
                print(f"Center of {label}: ({center_x}, {center_y}),Radius :{radius}")

                # 获取圆环上的点的深度
                depths = []
                num_points = 20  # 取圆环上的36个点
                for i in range(num_points):
                    angle = 2 * np.pi * i / num_points
                    point_x = int(center_x + radius * np.cos(angle))
                    point_y = int(center_y + radius * np.sin(angle))
                    if 0 <= point_x < depth_image.shape[1] and 0 <= point_y < depth_image.shape[0]:
                        depth = depth_image[point_y, point_x]
                        depths.append(depth)
                        # 在深度图像上绘制圆环上的点
                        cv2.circle(depth_show, (point_x, point_y), 2, (255, 0, 0), -1)
                if depths:
                    print(depths)
                    average_depth_mm = np.mean(depths)
                    average_depth_m = average_depth_mm
                    print(f"Average depth on the ring: {average_depth_m}")
                
    pixel_coords = (center_x, center_y)  # 图像中心点

    depth = average_depth_m + 0.15 #深度微调0.12m的误差，不知道是不是habitat内部传感器设置的问题

    # 相机内参
    camera_intrinsics = np.array([
            [64., 0., 64.],
            [0., 64., 64.],
            [0., 0., 1.]
    ])

    # print(camera_intrinsics)
    
    # 相机外参
    # camera_extrinsics = np.eye(4)
    # quat = habitat_sim.utils.quat_from_angle_axis(orientation[0], np.array([1.0, 0.0, 0.0])) * \
    #     habitat_sim.utils.quat_from_angle_axis(orientation[1], np.array([0.0, 1.0, 0.0])) * \
    #     habitat_sim.utils.quat_from_angle_axis(orientation[2], np.array([0.0, 0.0, 1.0]))
    # camera_extrinsics[:3, :3] = quat_to_matrix(quat)
    # camera_extrinsics[:3, 3] = position
    camera_extrinsics = np.array([
        [1., 0., 0., 0],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])
    # print(camera_extrinsics)
    
    #像素系到世界系=像素系到机体系/相机系+机体系到世界系（初始position）
    world_coords = pixel_to_body(pixel_coords, depth, camera_intrinsics, camera_extrinsics) + origin_position 
    print(f"World Coordinates: {world_coords}")
    
    cv2.imshow("YOLO Realtime Detection", rgb_image)
    cv2.imshow("Depth Image", depth_show)
    # # # # # env.reset()
    # # # #     # 按下'q'键退出循环
    # # # #     # if cv2.waitKey(1) & 0xFF == ord('q'):
    # # # #     #     break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
      

if __name__ == "__main__":
    main()