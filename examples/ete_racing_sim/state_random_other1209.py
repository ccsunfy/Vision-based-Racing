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
import utils.algorithms.lr_scheduler as lr_scheduler
# from envs.projection_fill_one930 import CircleEnv
# from envs.multi_per_state1112 import RacingEnv2
# from envs.state_racing_ellipse_1126ok import RacingEnv2
from envs.demo1_straight import RacingEnv2
# from envs.state_racing_straight1213 import RacingEnv2
# from envs.state_racing_8_1126 import RacingEnv2
from utils.launcher import rl_parser, training_params
from utils.type import Uniform
from gymnasium import spaces       

args = rl_parser().parse_args()

""" SAVED HYPERPARAMETERS """
training_params["num_env"] = 100
training_params["learning_step"] = 5e8
training_params["comment"] = args.comment
training_params["max_episode_steps"] = 256
training_params["n_steps"] = training_params["max_episode_steps"]
training_params["batch_size"] = training_params["num_env"] * training_params["n_steps"]
training_params["learning_rate"] = 1e-3
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"

# scene_path = "datasets/spy_datasets/configs/garage_empty"
# scene_path = "datasets/spy_datasets/configs/racing8_ob"
# scene_path = "datasets/spy_datasets/configs/racing8_random_ob"
# scene_path = "datasets/spy_datasets/configs/racing_test"
# scene_path = "datasets/spy_datasets/configs/racing8"
# scene_path = "datasets/spy_datasets/configs/racing_straight"
scene_path = "datasets/spy_datasets/configs/racing_straight_random"
# scene_path = "datasets/spy_datasets/configs/racing_no_ob"
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
                                {"position": {"mean": [2., 2., 1], "half": [.2, .2, 0.2]}},
                        },
                        {
                            "class": "Uniform",
                            "kwargs":
                                {"position": {"mean": [6., 2., 1.5], "half": [.2, .2, 0.2]}},
                        },
                        {
                            "class": "Uniform",
                            "kwargs":
                                {"position": {"mean": [6., -2., 1.5], "half": [.2, .2, 0.2]}},
                        },
                        {
                            "class": "Uniform",
                            "kwargs":
                                {"position": {"mean": [2., 0., 1], "half": [.2, .2, 0.2]}},
                        },
                    ]
                }
            ]
        }
}

random_kwargs = {}

# 只在调试的时候打开即可
# torch.autograd.detect_anomaly()

latent_dim = 256
# latent_dim = None

def main():
    # if train mode, train the model
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    if args.train:
        env = RacingEnv2(num_agent_per_scene=training_params["num_env"],
                        # num_agent_per_scene=training_params["num_env"]/2,
                        # 如果需要开启多个环境，需要设置num_scene
                            num_scene=3,
                            visual=True, # 不用视觉要改成False
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
                            latent_dim=latent_dim,
                            random_kwargs=random_kwargs
                            )
        
        if args.weight is not None:
            model = ppo.load(save_folder + args.weight, env=env)
        else:
            model = ppo(
                # lr_schedule = lr_scheduler.linear_schedule(1e-3, 1e-4),
                policy="CustomMultiInputPolicy",
                policy_kwargs=dict(
                    # features_extractor_class = {},
                    # features_extractor_kwargs = {},
                    pi_features_extractor_class=extractors.StateIndexImageExtractor,
                    pi_features_extractor_kwargs={
                        "net_arch": {
                            # "pastAction":{
                            #     "mlp_layer": [128, 128],d
                            # },
                            "depth": {
                                "mlp_layer": [256],
                            },
                            "state":{ 
                                "mlp_layer": [128, 128],
                            },
                            "vd": {
                                "mlp_layer": [128, 128],
                            },
                            "index":{ 
                                "mlp_layer": [128, 128],
                            },
                            "recurrent":{
                                "class": "GRU",
                                "kwargs":{
                                    "hidden_size": latent_dim,
                                }
                            }
                        }
                    },
                    vf_features_extractor_class=extractors.StateIndexVdImageExtractor,
                    vf_features_extractor_kwargs={
                        "net_arch": {
                            "state": {
                                "mlp_layer": [128, 128],
                            },
                            "vd": {
                                "mlp_layer": [128, 128],
                            },
                            # "pastAction":{
                            #     "mlp_layer": [128, 128],
                            # },
                            "depth": {
                                "mlp_layer": [256],
                            },
                            "index":{ 
                                "mlp_layer": [128, 128],
                            },
                            "recurrent":{
                                "class": "GRU",
                                "kwargs":{
                                    "hidden_size": latent_dim,
                                }
                            }
                        }
                    },
                    net_arch=dict(
                        pi=[192, 96],
                        vf=[192, 96]),
                    activation_fn=torch.nn.LeakyReLU,
                    optimizer_kwargs=dict(weight_decay=1e-5),
                ),
                env=env,
                verbose=training_params["verbose"],
                tensorboard_log=save_folder,
                gamma=training_params["gamma"],  # lower 0.9 ~ 0.99
                n_steps=training_params["n_steps"],
                ent_coef=training_params["ent_coef"],
                learning_rate=lr_scheduler.linear_schedule(1e-3, 1e-4),
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
        env = RacingEnv2(num_agent_per_scene=1, visual=True,
                            # random_kwargs=random_kwargs,
                            scene_kwargs={
                                 "path": scene_path,
                                 "render_settings": {
                                     "mode": "fix",
                                     "view": "custom",
                                     "resolution": [1080, 1920],
                                    #  "position": th.tensor([[12., 6.8, 5.5], [10,4.8,4.5]]),
                                     "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
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
        test_handle.test(is_fig=True, is_fig_save=True, is_render=True, is_video=True, is_sub_video=True, is_video_save=True,
        render_kwargs ={
            "points": th.tensor([[4., 0, 1.],[1, 0, 1]])
        })
        print("Test completed.")


if __name__ == "__main__":
    main()
