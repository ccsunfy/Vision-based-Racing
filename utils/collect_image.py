#!/usr/bin/env python3

import sys
import os
import numpy as np

sys.path.append(os.getcwd())
from envs.demo1_straight  import RacingEnv2
from utils.launcher import rl_parser, training_params
from PIL import Image

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
scene_path = "datasets/spy_datasets/configs/demo1_straight_ob1"
# random_kwargs = {
#     "state_generator_kwargs": [{
#         "position": Uniform(mean=th.tensor([1., 0., 1.5]), half=th.tensor([0.0, 2., 1.]))
#     }]
# }
latent_dim = 256
latent_dim = None

env = RacingEnv2(num_agent_per_scene=training_params["num_env"],
                        # random_kwargs=random_kwargs,
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

# torch.autograd.detect_anomaly()
# random_kwargs = {}



def main():
    
    for episode in range(2):
        env.reset()
        obs = env.get_observation()
        rgb_images = obs["color"]
        depth_images = obs["depth"]
        semantic_images = obs["semantic"]
        
        # rgb
        for i, rgb_image in enumerate(rgb_images):
            # 去掉多余的维度并确保数据类型是uint8
            rgb_image = np.squeeze(rgb_image).astype(np.uint8)
            
            #调整图像维度
            rgb_image = np.transpose(rgb_image, (1, 2, 0))
            
            # 检查图像形状是否正确
            if rgb_image.ndim == 3 and rgb_image.shape[2] == 3:
                
                # 创建PIL图像对象
                rgb_image_pil = Image.fromarray(rgb_image)
                
                # 保存图像为JPEG文件
                rgb_image_path = os.path.join(save_folder, f"rgb_image_{episode}.jpg")
                rgb_image_pil.save(rgb_image_path)
                
                env.reset()
                
                print(f"Saved RGB image to {rgb_image_path}")
            else:
                env.reset()
                print(f"Unexpected image shape: {rgb_image.shape}")
                
        # depth
        for i, depth_image in enumerate(depth_images):
            # 去掉多余的维度并确保数据类型是uint8
            depth_image = (depth_image / np.max(depth_image) * 255).astype(np.uint8)
            depth_image = np.squeeze(depth_image)
            
            # 检查图像形状是否正确
            if depth_image.ndim == 3:
                depth_image = depth_image[0]
                
            # 创建PIL图像对象
            depth_image_pil = Image.fromarray(depth_image)
        
            depth_image_path = os.path.join(save_folder, f"depth_image_{episode}.jpg")
            depth_image_pil.save(depth_image_path)
            env.reset()
            print(f"Saved depth image to {depth_image_path}")
            
        # semantic
        for i, semantic_image in enumerate(semantic_images):
            # 去掉多余的维度并确保数据类型是uint8
            semantic_image = (semantic_image / np.max(semantic_image) * 255).astype(np.uint8)
            semantic_image = np.squeeze(semantic_image)
            if semantic_image.ndim == 3:
                semantic_image = semantic_image[0]
                
            # 创建PIL图像对象
            semantic_image_pil = Image.fromarray(semantic_image)
        
            semantic_image_path = os.path.join(save_folder, f"semantic_image_{episode}.jpg")
            semantic_image_pil.save(semantic_image_path)
            env.reset()
            print(f"Saved semantic image to {semantic_image_path}")


if __name__ == "__main__":
    main()
