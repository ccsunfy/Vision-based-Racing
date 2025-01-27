import sys
import os
import torch as th
import numpy as np
import logging

sys.path.append(os.getcwd())
from utils.policies import extractors 
from utils.algorithms.ppo import ppo
from utils import savers
import utils.algorithms.lr_scheduler as lr_scheduler
from utils.launcher import training_params

training_params["num_env"] = 100
training_params["learning_step"] = 4e7
# training_params["comment"] = args.comment
training_params["max_episode_steps"] = 256
training_params["n_steps"] = training_params["max_episode_steps"]
training_params["batch_size"] = training_params["num_env"] * training_params["n_steps"]
training_params["learning_rate"] = 1e-3

# student ppo
student_policy_kwargs = dict(
    pi_features_extractor_class=extractors.StateIndexVdImageExtractor,
    pi_features_extractor_kwargs={
        "net_arch": {
            "depth": {
                "mlp_layer": [128],  # 减小网络规模
            },
            "state":{ 
                "mlp_layer": [64, 64],
            },
            "vd": {
                "mlp_layer": [64, 64],
            },
            "index":{ 
                "mlp_layer": [64, 64],
            },
            "recurrent":{
                "class": "GRU",
                "kwargs":{
                    "hidden_size": 128,  # 减小隐藏层维度
                }
            }
        }
    },
    vf_features_extractor_class=extractors.StateIndexVdImageExtractor,
    vf_features_extractor_kwargs={
        "net_arch": {
            "depth": {
                "mlp_layer": [128],
            },
            "state": {
                "mlp_layer": [64, 64],
            },
            "vd": {
                "mlp_layer": [64, 64],
            },
            "index":{ 
                "mlp_layer": [64, 64],
            },
            "recurrent":{
                "class": "GRU",
                "kwargs":{
                    "hidden_size": 128,
                }
            }
        }
    },
    net_arch=dict(
        pi=[96, 48],  # 减小策略网络
        vf=[96, 48]), # 减小价值网络
    activation_fn=th.nn.LeakyReLU,
)

def train_student_ppo(env, teacher_model, student_policy_kwargs):
    student_model = ppo(
        policy="CustomMultiInputPolicy",
        policy_kwargs=student_policy_kwargs,
        env=env,
        verbose=1,
        tensorboard_log="./student_ppo_logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device="cuda"
    )

    # 添加蒸馏损失
    def distill_callback(locals, globals):
        # 获取当前batch的观测
        observations = locals["batch"].observations
        
        # 教师模型预测
        with th.no_grad():
            teacher_actions, _ = teacher_model.policy.predict(observations)
        
        # 学生模型预测
        student_actions, _ = student_model.predict(observations)
        
        # 计算蒸馏损失
        distill_loss = th.nn.MSELoss()(
            th.FloatTensor(student_actions), 
            th.FloatTensor(teacher_actions)
        )
        
        # 添加到总损失中
        locals["loss"] = locals["loss"] + 0.1 * distill_loss
        
        return True

    # 训练学生模型
    student_model.learn(
        total_timesteps=10000000,
        callback=distill_callback
    )
    
    return student_model

if __name__ == "__main__":
    # 加载教师模型
    teacher_model = ppo.load("examples/nature_cross/ppo_436.zip")
    
    # 创建环境
    from envs.waypoint_test import RacingEnv2
    env = RacingEnv2(
        num_agent_per_scene=training_params["num_env"],
        visual=True,
        max_episode_steps=training_params["max_episode_steps"],
        latent_dim=128  # 使用更小的潜在维度
    )
    
    # 训练学生PPO
    student_model = train_student_ppo(env, teacher_model, student_policy_kwargs)
    
    # 保存模型
    student_model.save("student_ppo_model")