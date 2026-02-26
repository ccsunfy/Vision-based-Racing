import os
import sys
import numpy as np
import logging
sys.path.append(os.getcwd())

from student_net import StudentPolicy
from utils.algorithms.ppo import ppo
from utils.launcher import rl_parser, training_params
# from envs.tech1_hover import HoverEnv2
from envs.tech2_racing import RacingEnv2
# from envs.tech3_tracking import TrackEnv2
# from envs.tech4_landing import LandingEnv2
import utils.algorithms.lr_scheduler as lr_scheduler
from dagger import DAgger

args = rl_parser().parse_args()

""" SAVED HYPERPARAMETERS """
training_params["num_env"] = 90
training_params["learning_step"] = 5e7
training_params["comment"] = args.comment
training_params["max_episode_steps"] = 512
training_params["n_steps"] = training_params["max_episode_steps"]
training_params["batch_size"] = training_params["num_env"] * training_params["n_steps"]
training_params["learning_rate"] = 1e-3
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"

scene_path = "datasets/spy_datasets/configs/garage_empty"
# scene_path = "datasets/spy_datasets/configs/racing8"

latent_dim = 256
latent_dim = None

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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

env = RacingEnv2(num_agent_per_scene=training_params["num_env"],
                        # num_agent_per_scene=training_params["num_env"]/2,
                        # 如果需要开启多个环境，需要设置num_scene
                        # num_scene=2,
                            visual=True, # 不用视觉要改成False
                            max_episode_steps=training_params["max_episode_steps"],
                            scene_kwargs={
                                "path": scene_path,
                            },
                            dynamics_kwargs={
                                "dt": 0.01,
                                "ctrl_dt": 0.03,
                                "action_type":"bodyrate",
                            },
                            # requires_grad=True,
                            latent_dim=latent_dim,
                            random_kwargs=random_kwargs
                            )

if __name__ == "__main__":
    # model_paths = [
    #     "examples/multi_task_IL/racing.zip",
    #     "examples/multi_task_IL/tracking.zip",
    #     "examples/multi_task_IL/stabling.zip",
    #     "examples/multi_task_IL/landing.zip"
    # ]
    model_paths = ["examples/multi_task_IL/racing_state.zip"]
    
    models = [ppo.load(path, device="cuda") for path in model_paths]
    teachers = [model.policy for model in models]
    
    # teacher_racing, teacher_tracking, teacher_stabling, teacher_landing = teachers
    teacger_racing = teachers
    student = StudentPolicy(backbone_name="resnet18",
                            hidden_dim=256)
    
    DAgger(
        env=env, 
        learning_rate=lr_scheduler.linear_schedule(1e-4, 1e-5),
        seed=42,
        student=student,
        device="cuda",
        save_folder=save_folder,
        teachers=teachers,
        learning_steps=5e4,
        num_episodes_per_iter=5
        )