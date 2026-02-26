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
from envs.gate_avoid1104 import CircleEnv
from utils.launcher import rl_parser, training_params
from utils.type import Uniform
from gymnasium import spaces
from utils.evaluate import TestBase
import copy
from typing import Optional
from matplotlib import pyplot as plt
from utils.FigFashion.FigFashion import FigFon

scene_path = "datasets/spy_datasets/configs/cross_circle"

random_kwargs = {
    "state_generator_kwargs": [{
        "position": Uniform(mean=th.tensor([1.0, 0.0, 0.5]), half=th.tensor([0.5, 0.5, 0.5]))
    }]
}

latent_dim = 256

env = CircleEnv(
    num_agent_per_scene=1,
    random_kwargs=random_kwargs,
    visual=True,
    scene_kwargs={
        "path": scene_path,
        "render_settings": {
            "mode": "fix",
            "view": "custom",
            "resolution": [1080, 1920],
            "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
            "trajectory": True,
        }
    },
    latent_dim=latent_dim
)

model_path = 'examples/nature_cross/saved/ppo_200.zip'
model = ppo.load(model_path, env=env)

class SimToSimStateMachine(TestBase):
    states = ['initializing', 'running', 'paused', 'stopped']

    def __init__(self, env, model, name, save_path: Optional[str] = None):
        super(SimToSimStateMachine, self).__init__(model, name, save_path)
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

    def run_simulation(self, num_steps, is_fig=False, is_video=False, is_obs_video=False, is_fig_save=False, is_video_save=False, is_render=False, render_kwargs={}):
        done_all = th.full((self.model.env.num_envs,), False)
        obs = self.model.env.reset()
        self._img_names = [name for name in obs.keys() if (("color" in name) or ("depth" in name) or ("semantic" in name))]
        self.obs_all.append(obs)
        self.info_all.append([{} for _ in range(self.model.env.num_envs)])
        self.t.append(self.model.env.t.clone())
        self.collision_all.append({"col_dis": self.model.env.collision_dis,
                                   "is_col": self.model.env.is_collision,
                                   "col_pt": self.model.env.collision_point})
        while True:
            with th.no_grad():
                action = self.model.policy.predict(obs)
                if isinstance(action, tuple):
                    action = action[0]
                self.model.env.step(action, is_test=True)
                obs, reward, done, info = self.model.env.get_observation(), self.model.env.reward, self.model.env.done, self.model.env.info
                col_dis, is_col, col_pt = self.model.env.collision_dis, self.model.env.is_collision, self.model.env.collision_point
                self.collision_all.append({"col_dis": col_dis, "is_col": is_col, "col_pt": col_pt})

            self.reward_all.append(reward)
            self.action_all.append(action)
            self.obs_all.append(obs)
            self.info_all.append(copy.deepcopy(info))
            self.t.append(self.model.env.t.clone())
            if is_render:
                render_image = cv2.cvtColor(self.model.env.render(render_kwargs)[0], cv2.COLOR_RGBA2RGB)
                self.render_image_all.append(render_image)
            done_all[done] = True
            
            # if (obs.position - self.model.env.target()) < 0.1:
            # 检查是否离下一个 gate 只有 0.1 米
            next_target_position = self.model.env.targets[self.model.env._next_target_i.clamp_max(len(self.model.env.targets) - 1)]
            distance_to_next_target = th.norm(obs['position'] - next_target_position, dim=1)
            if th.any(distance_to_next_target < 0.1):
                self.trigger('pause')
                self.trigger('resume')
                
            if done_all.all():# 录制结束逻辑
                break
            # if self.success_final():
            #     break
            
        print(f"Average Rewards:{np.array([info['episode']['r'] for info in self.info_all[-1]]).mean()}")

        if is_fig:
            figs = self.draw()
            if is_fig_save:
                for i, fig in enumerate(figs):
                    self.save_fig(fig, c=i)
        if is_video:
            self.play(is_obs_video=is_obs_video)
            if is_video_save:
                self.save_video()

    def draw(self, names=None):
        state_data = [obs["state"] for obs in self.obs_all]
        state_data = np.array(state_data)
        t = np.stack(self.t)[:, 0]
        for i in range(self.model.env.num_envs):
            fig = plt.figure(figsize=(5, 4))
            plt.subplot(2, 2, 1)
            plt.plot(t, state_data[:, i, 0:3], label=["x", "y", "z"])
            plt.legend()
            plt.subplot(2, 2, 2)
            plt.plot(t, state_data[:, i, 3:7], label=["w", "x", "y", "z"])
            plt.legend()
            plt.subplot(2, 2, 3)
            plt.plot(t, state_data[:, i, 7:10], label=["vx", "vy", "vz"])
            plt.legend()
            plt.subplot(2, 2, 4)
            plt.plot(t, state_data[:, i, 10:13], label=["wx", "wy", "wz"])
            plt.legend()
            plt.tight_layout()
            plt.show()
        col_dis = np.array([collision["col_dis"] for collision in self.collision_all])
        fig2, axes = FigFon.get_figure_axes(SubFigSize=(1, 1))
        axes.plot(t, col_dis)
        axes.set_xlabel("t/s")
        axes.set_ylabel("closest distance/m")
        plt.show()
        return [fig, fig2]

    # def play(self, is_obs_video):
    #     if len(self.render_image_all) == 0:
    #         print("No render images to play.")
    #         return

    #     # 获取每个 agent 的视频帧
    #     num_agents = self.model.env.num_envs
    #     agent_videos = [[] for _ in range(num_agents)]
    #     for i, frame in enumerate(self.render_image_all):
    #         agent_index = i % num_agents
    #         agent_videos[agent_index].append(frame)

    #     # 显示每个 agent 的视频
    #     for agent_index, video_frames in enumerate(agent_videos):
    #         for frame in video_frames:
    #             cv2.imshow(f'Agent {agent_index}', frame)
    #             if cv2.waitKey(30) & 0xFF == ord('q'):
    #                 break
    #         cv2.destroyAllWindows()

    # def save_video(self):
    #     if len(self.render_image_all) == 0:
    #         print("No render images to save.")
    #         return

    #     # 获取每个 agent 的视频帧
    #     num_agents = self.model.env.num_envs
    #     agent_videos = [[] for _ in range(num_agents)]
    #     for i, frame in enumerate(self.render_image_all):
    #         agent_index = i % num_agents
    #         agent_videos[agent_index].append(frame)

    #     # 保存每个 agent 的视频
    #     for agent_index, video_frames in enumerate(agent_videos):
    #         height, width, _ = video_frames[0].shape
    #         video_path = f'agent_{agent_index}_video.avi'
    #         out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    #         for frame in video_frames:
    #             out.write(frame)
    #         out.release()
    #         print(f'Saved video for Agent {agent_index} to {video_path}')
            
if __name__ == "__main__":
    state_machine = SimToSimStateMachine(env, model, name="SimToSimTest")

    state_machine.start()
    state_machine.run_simulation(100, is_fig=True, is_video=True, is_obs_video=True, is_fig_save=True, is_video_save=True, is_render=True)
    state_machine.pause()
    state_machine.resume()
    # state_machine.run_simulation(100, is_fig=True, is_video=True, is_obs_video=True, is_fig_save=True, is_video_save=True, is_render=True)
    state_machine.stop()
    cv2.destroyAllWindows()