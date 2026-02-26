#!/usr/bin/env python3
import sys
import os
import numpy as np
import torch
import time
import logging
from transitions import Machine
import cv2

sys.path.append(os.getcwd())
from utils.policies import extractors
from utils.algorithms.ppo import ppo
from utils import savers
import torch as th
from envs.OK1029_random import CircleEnv
from utils.launcher import rl_parser, training_params
from utils.type import Uniform
from gymnasium import spaces     

scene_path = "datasets/spy_datasets/configs/cross_circle"

random_kwargs = {
    "state_generator_kwargs": [{
        "position": Uniform(mean=th.tensor([1.0, 0.0, 0.5]), half=th.tensor([1.0, 1.0, 0.5]))
        # "velocity": Uniform(mean=th.tensor([0.0, 0.0, 0.0]), half=th.tensor([0.3, 0.3, 0.3]))
    }]
}

latent_dim = 256

env  = CircleEnv(num_agent_per_scene=1,
                        # num_agent_per_scene=training_params["num_env"]/2,
                        # 如果需要开启多个环境，需要设置num_scene
                        # num_scene=2,
                            random_kwargs=random_kwargs,
                            visual=True,
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
                            # dynamics_kwargs={
                            #     "dt": 0.02,
                            #     "ctrl_dt": 0.02,
                            #     # "action_type":"velocity",
                            # },
                            # requires_grad=True,
                            latent_dim=latent_dim
                            )

model_path = 'examples/nature_cross/saved/ppo_174.zip'
model = ppo.load(model_path, env=env)

class SimToSimStateMachine:
    states = ['initializing', 'running', 'paused', 'stopped']

    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.machine = Machine(model=self, states=SimToSimStateMachine.states, initial='initializing')

        self.machine.add_transition(trigger='start', source='initializing', dest='running', after='on_start')
        self.machine.add_transition(trigger='pause', source='running', dest='paused', after='on_pause')
        self.machine.add_transition(trigger='resume', source='paused', dest='running', after='on_resume')
        self.machine.add_transition(trigger='stop', source=['running', 'paused'], dest='stopped', after='on_stop')

    def on_start(self):
        print("Simulation started.")
        self.env.reset()

    def on_pause(self):
        print("Simulation paused.")

    def on_resume(self):
        print("Simulation resumed.")

    def on_stop(self):
        print("Simulation stopped.")

    def run_simulation(self, num_steps):
        for step in range(num_steps):
            if self.state == 'running':
                obs = self.env.get_observation()
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                print(f"Step: {step}, Reward: {reward}")
                # 可视化仿真过程
                self.visualize(step)
                
                if done:
                    self.env.reset()
                    
    def visualize(self, step):
        # 获取当前环境的图像
        image = cv2.cvtColor(self.model.env.render(render_kwargs)[0], cv2.COLOR_RGBA2RGB)
        # 显示图像
        cv2.imshow('Simulation', image)
        # 保存图像
        cv2.imwrite(f'simulation_step_{step}.png', image)
        # 等待一段时间以便显示图像
        cv2.waitKey(1)

if __name__ == "__main__":
    state_machine = SimToSimStateMachine(env, model)

    state_machine.start()
    state_machine.run_simulation(100)
    state_machine.pause()
    state_machine.resume()
    state_machine.run_simulation(100)
    state_machine.stop()