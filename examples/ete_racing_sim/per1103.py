#!/usr/bin/env python3

import sys
import os
import numpy as np
import torch
import time
import logging

sys.path.append(os.getcwd())
from utils.policies import extractors
from utils.algorithms.ppo import ppo
from utils import savers
import torch as th
# from envs.projection_fill_one930 import CircleEnv
from envs.multicircle_speed_up1113 import CircleEnv
from utils.launcher import rl_parser, training_params
from utils.type import Uniform
from gymnasium import spaces       

args = rl_parser().parse_args()

""" SAVED HYPERPARAMETERS """
training_params["num_env"] = 96
training_params["learning_step"] = 1e7
training_params["comment"] = args.comment
training_params["max_episode_steps"] = 200
training_params["n_steps"] = training_params["max_episode_steps"]
training_params["batch_size"] = training_params["num_env"] * training_params["n_steps"]
training_params["learning_rate"] = 3e-4
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"

scene_path = "datasets/spy_datasets/configs/cross_circle"
# scene_path = "datasets/spy_datasets/configs/cross_circle"
# random_kwargs = {
#     "state_generator_kwargs": [{
#         "position": Uniform(mean=th.tensor([1., 0., 1.5]), half=th.tensor([0.0, 2., 1.]))
#     }]
# }
# random_kwargs = {
#     "state_generator_kwargs": [{
#         # "position": Uniform(mean=th.tensor([1.0, 0., 0.]), half=th.tensor([0.1, 0.1, 0.1]))
#         # "position": Uniform(mean=th.tensor([1.5, 0., 0.1]), half=th.tensor([1.0, 1.0, 1.0])),
#         "position": Uniform(mean=th.tensor([1.0, 0.0, 0.5]), half=th.tensor([0.5, 0.5, 0.5])),
#         # "orientation": Uniform(mean=th.tensor([0.0, 0.0, 0.0]), half=th.tensor([0.1, 0.1, 0.1])),
#         # "velocity": Uniform(mean=th.tensor([0.0, 0.0, 0.0]), half=th.tensor([0.3, 0.3, 0.0])),
#         # "angular_velocity": Uniform(mean=th.tensor([0.0, 0.0, 0.0]), half=th.tensor([0.0, 0.0, 0.0]))
#     }]
# }
random_kwargs = {
            "state_generator":
                {
                    "class": "Union",

                    "kwargs": [
                        {"randomizers_kwargs":
                            [
                                {
                                    "class": "Uniform",
                                    "kwargs":
                                        {"position": {"mean": [4., 0., 1], "half": [.2, .2, 0.2]}},

                                },
                                {
                                    "class": "Uniform",
                                    "kwargs":
                                        {"position": {"mean": [6., 0., 1.], "half": [.2, .2, 0.2]}},

                                },
                                {
                                    "class": "Uniform",
                                    "kwargs":
                                        {"position": {"mean": [8., -1., 1.], "half": [.2, .2, 0.2]}},

                                },
                                {
                                    "class": "Uniform",
                                    "kwargs":
                                        {"position": {"mean": [10., -1., 1], "half": [.2, .2, 0.2]}},

                                },
                            ]
                        }
                    ]

                }
        }
# 可能一开始飞得太高看不到圈导致出现了问题
# random_kwargs = {
#     "state_generator_kwargs": [{
#         "position": Uniform(mean=th.tensor([1., 0., 1.5]), half=th.tensor([0.0, 2., 1.]))
#     }]
# }

# 只在调试的时候打开即可
# torch.autograd.detect_anomaly()

latent_dim = 256
# latent_dim = None

def main():
    # if train mode, train the model
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    if args.train:
        env = CircleEnv(num_agent_per_scene=training_params["num_env"],
                        # num_agent_per_scene=training_params["num_env"]/2,
                        # 如果需要开启多个环境，需要设置num_scene
                        # num_scene=2,
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
        
        if args.weight is not None:
            model = ppo.load(save_folder + args.weight, env=env)
        else:
            model = ppo(
                policy="CustomMultiInputPolicy",
                policy_kwargs=dict(
                    # features_extractor_class = {},
                    # features_extractor_kwargs = {},
                    pi_features_extractor_class=extractors.ActionImageMaskNoiseIndexExtractor,
                    pi_features_extractor_kwargs={
                        "net_arch": {
                            "depth": {
                                "mlp_layer": [256],
                            },
                            "pastAction":{
                                "mlp_layer": [128, 128],
                            },
                            # "state": {
                            #     "mlp_layer": [128, 128],
                            # },
                            # "target": {
                            #     "mlp_layer": [128, 128],
                            # },
                            "noise_target":{ # 新加入噪声目标
                                "mlp_layer": [128, 128],
                            },
                            "index":{ # gate索引
                                "mlp_layer": [128, 128],
                            },
                            "mask":{
                                "mlp_layer": [256],
                            },
                            "recurrent":{
                                "class": "GRU",
                                "kwargs":{
                                    "hidden_size": latent_dim,
                                }
                            }
                        }
                    },
                    vf_features_extractor_class=extractors.ActionStateTargetImageMaskIndexExtractor,
                    vf_features_extractor_kwargs={
                        "net_arch": {
                            "depth": {
                                "mlp_layer": [256],
                            },
                            "state": {
                                "mlp_layer": [128, 128],
                            },
                            "target": {
                                "mlp_layer": [128, 128],
                            },
                            "pastAction":{
                                "mlp_layer": [128, 128],
                            },
                            "index":{ # 新加入噪声目标
                                "mlp_layer": [128, 128],
                            },
                            "mask":{
                                ""
                                "mlp_layer": [256],
                            },
                            "recurrent":{
                                "class": "GRU",
                                "kwargs":{
                                    "hidden_size": latent_dim,
                                }
                            }
                        }
                    },
                    # net_arch=dict(
                    #     pi=[360, 360],
                    #     vf=[360, 360]),
                    net_arch=dict(
                        pi=[128, 128],
                        vf=[128, 128]),
                    activation_fn=torch.nn.LeakyReLU,
                    optimizer_kwargs=dict(weight_decay=1e-5),
                ),
                env=env,
                verbose=training_params["verbose"],
                tensorboard_log=save_folder,
                gamma=training_params["gamma"],  # lower 0.9 ~ 0.99
                n_steps=training_params["n_steps"],
                ent_coef=training_params["ent_coef"],
                learning_rate=training_params["learning_rate"],
                vf_coef=training_params["vf_coef"],
                max_grad_norm=training_params["max_grad_norm"],
                batch_size=training_params["batch_size"],
                gae_lambda=training_params["gae_lambda"],
                n_epochs=training_params["n_epochs"],
                clip_range=training_params["clip_range"],
                device="cuda",
                seed=training_params["seed"],
                comment=args.comment,
            )

        start_time = time.time()
        
        logging.info('Starting training...')
        model.learn(training_params["learning_step"])
        logging.info('Training completed')
        model.save()
        logging.info('Model saved')
        training_params["time"] = time.time() - start_time

        savers.save_as_csv(save_folder + "training_params.csv", training_params)

    # Testing mode with a trained weight
    else:
        test_model_path = save_folder + args.weight
        print("Loading environment...")
        from test import Test
        env = CircleEnv(num_agent_per_scene=1, visual=True,
                            random_kwargs=random_kwargs,
                            scene_kwargs={
                                 "path": scene_path,
                                 "render_settings": {
                                     "mode": "fix",
                                     "view": "custom",
                                     "resolution": [1080, 1920],
                                    #  "position": th.tensor([[12., 6.8, 5.5], [10,4.8,4.5]]),
                                     "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
                                    # "position": th.tensor([[6.,-1.5,1.], [6.,-1.5,1.]]),

                                     # "point": th.tensor([[9., 0, 1], [1, 0, 1]]),
                                     "trajectory": True,
                                 }
                             },
                            latent_dim=latent_dim)
        print("Environment loaded.")
        model = ppo.load(test_model_path, env=env)
        print("Model loaded.")
        test_handle = Test(
                           model=model,
                           save_path=os.path.dirname(os.path.realpath(__file__)) + "/saved/test",
                           name=args.weight)
        print("Starting test...")
        test_handle.test(is_fig=True, is_fig_save=True, is_render=True, is_video=True, is_video_save=True,
        render_kwargs ={
            "points": th.tensor([[4., 0, 1.],[1, 0, 1]])
        })
        print("Test completed.")


if __name__ == "__main__":
    main()
