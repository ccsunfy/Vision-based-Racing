import os, sys
sys.path.append(os.getcwd())
# from examples.stable_flight.droneStableEnvs import droneStableEnvs
from envs.random_circle_1023 import CircleEnv
from utils.launcher import dl_parser as parser
from utils.algorithms.dl_algorithm import ApgBase
from test import Test
from utils.policies import extractors
from utils.launcher import rl_parser, training_params
import torch as th
from utils.type import Uniform

args = parser().parse_args()
training_params["horizon"] = 96
training_params["max_episode_steps"] = 256
training_params["num_env"] = 96

save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/"
# path = "datasets/spy_datasets/configs/garage_simple"
scene_path = "datasets/spy_datasets/configs/special_circle"

# torch.autograd.set_detect_anomaly(True)

random_kwargs = {
    "state_generator_kwargs": [{
        "position": Uniform(mean=th.tensor([1.0, 0.0, 0.1]), half=th.tensor([0.0, 0.1, 0.1]))
    }]
}

latent_dim = 256

def main():

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
            model = ApgBase.load(env, save_folder + args.weight)
        else:
            model = ApgBase(
                env=env,
                policy="CustomMultiInputPolicy",
                policy_kwargs=dict(
                    # features_extractor_class = {},
                    # features_extractor_kwargs = {},
                    pi_features_extractor_class=extractors.ActionImageMaskNoiseExtractor,
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
                    vf_features_extractor_class=extractors.ActionStateTargetImageMaskNoiseExtractor,
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
                    activation_fn=th.nn.LeakyReLU,
                    optimizer_kwargs=dict(weight_decay=1e-5),
                ),
                learning_rate=1e-3,
                device="cuda",
                commit=args.comment,
            )
        model.learn(total_timesteps=1e7,
                  horizon=training_params["horizon"])
        model.save()
    else:
        env = CircleEnv(num_agent_per_scene=1, visual=True,
                            scene_kwargs={
                                 "path": scene_path,
                                 "render_settings": {
                                     "mode": "fix",
                                     "view": "near",
                                     "resolution": [1080, 1080],
                                     "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]])
                                     # "position": torch.tensor([[8., 7., 3.]]),
                                 }
                             })
        model = ApgBase.load(env, save_folder + args.weight)
        test_handle = Test(
            env=env,
            policy=model,
            name="apg",
        )
        test_handle.test(is_fig=True, is_fig_save=True, is_render=True, is_video=True, is_video_save=True)

if __name__ == "__main__":
    main()

