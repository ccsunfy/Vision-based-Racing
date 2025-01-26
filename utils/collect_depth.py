#!/usr/bin/env python3

import sys
import os
import numpy as np
import torch
import time

sys.path.append(os.getcwd())
from utils.policies import extractors
from utils.algorithms.ppo import ppo
from utils import savers
import torch as th
# from envs.multicircle917 import CircleEnv
from envs.state_racing_ellipse_1126ok import RacingEnv2
from utils.launcher import rl_parser, training_params
from utils.type import Uniform
from PIL import Image

save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/depth_image/"

args = rl_parser().parse_args()
training_params["num_env"] = 100
training_params["learning_step"] = 1e7
training_params["comment"] = args.comment
training_params["max_episode_steps"] = 256
training_params["n_steps"] = training_params["max_episode_steps"]
training_params["batch_size"] = training_params["num_env"] * training_params["n_steps"]
training_params["learning_rate"] = 1e-3

# scene_path = "datasets/spy_datasets/configs/garage_simple_l_medium"
# scene_path = "datasets/spy_datasets/configs/cross_circle"
scene_path = "datasets/spy_datasets/configs/racing"
scene_path = "datasets/spy_datasets/configs/racing_straight"

# random_kwargs = {
#     "state_generator_kwargs": [{
#         "position": Uniform(mean=th.tensor([1., 0., 1.5]), half=th.tensor([0.0, 2., 1.]))
#     }]
# }
random_kwargs = {}
latent_dim = 256
# latent_dim = None

env = RacingEnv2(num_agent_per_scene=training_params["num_env"],
                        random_kwargs=random_kwargs,
                        visual=True,
                        max_episode_steps=training_params["max_episode_steps"],
                        scene_kwargs={
                             "path": scene_path,
                         },
                        dynamics_kwargs={
                            "dt": 0.02,
                            "ctrl_dt": 0.02,
                            # "action_type":"velocity",
                        },
                        # requires_grad=True,
                        latent_dim=latent_dim
                        )

# torch.autograd.detect_anomaly()



def main():
    env.reset()
    for episode in range(1000):
        obs = env.get_observation()
        depth_image = obs["depth"]
        # depth_image = obs["depth"].cpu().numpy().squeeze()
        
        depth_image_8bit = (depth_image / np.max(depth_image) * 255).astype(np.uint8)
        
        depth_image_8bit = np.squeeze(depth_image_8bit)
    
        if depth_image_8bit.ndim == 3:
            depth_image_8bit = depth_image_8bit[0]
        
        depth_image_pil = Image.fromarray(depth_image_8bit)
        
        depth_image_path = os.path.join(save_folder, f"depth_image_{episode}.jpg")
        depth_image_pil.save(depth_image_path)
        
        env.reset()
        print(f"Saved depth image to {depth_image_path}")
        # print(f"The shape of the depth image is {depth_image_8bit.shape}")

if __name__ == "__main__":
    main()
