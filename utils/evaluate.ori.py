import os.path
from abc import ABC, abstractmethod
import torch as th
from matplotlib import pyplot as plt
from typing import List, Union
import cv2
import sys
import numpy as np
import copy
from utils.FigFashion.FigFashion import FigFon

FigFon.set_fashion("IEEE")


class  TestBase:
    def __init__(
            self,
            model=None,
            name: Union[List[str], None] = None,
            save_path: str = None,

    ):
        self.save_path = os.path.join(save_path, name) if save_path is not None else os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/test/" + name
        self.model = model
        self.name = name

        self.obs_all = []
        self.info_all = []
        self.action_all = []
        self.collision_all = []
        self.render_image_all = []
        self.reward_all = []
        self.t = []

    def test(
            self,
            is_fig: bool = False,
            is_video: bool = False,
            is_obs_video: bool = False,
            is_fig_save: bool = False,
            is_video_save: bool = False,
            is_render: bool = False,
            render_kwargs={},
    ):
        if is_fig_save:
            if not is_fig:
                raise ValueError("is_fig_save must be True if is_fig is True")
        if is_video_save:
            if not is_video:
                raise ValueError("is_video_save must be True if is_video is True")

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
                # obs, reward, done, info = self.model.env.step(action, is_test=True)
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
            if done_all.all():
                break

        test = 1
        if is_fig:
            figs = self.draw()
            if is_fig_save:
                for i, fig in enumerate(figs):
                    self.save_fig(fig, c=i)
        if is_video:
            self.play(is_obs_video=is_obs_video)
            if is_video_save:
                self.save_video()

    @abstractmethod
    def draw(self, names: Union[List[str], None] = "video") -> plt.Figure:
        raise NotImplementedError

    # @abstractmethod
    def play(self, render_name: Union[List[str], None] = "video",is_obs_video=False):
        """
        how to play the video
        """
        """
        how to draw the figures
        """
        for image, t, obs in zip(self.render_image_all, self.t, self.obs_all):
            cv2.imshow(winname=render_name, mat=image)
            if is_obs_video:
                for name in self._img_names:
                    cv2.imshow(winname=name,
                               mat=np.hstack(np.transpose(obs[name], (0,2,3,1) ))
                               )
            cv2.waitKey(int(self.model.env.env.dynamics.ctrl_dt * 1000))

    def save_fig(self, fig, path=None, c=""):
        path = path if path is not None else self.save_path
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        fig.savefig(f"{path}_{c}.png")
        print(f"fig saved in {path}_{c}.png")

    def save_video(self, is_sub_video=False):
        height, width, layers = self.render_image_all[0].shape
        names = self.name if self.name is not None else "video"

        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))

        # render video
        path = f"{self.save_path}.avi"
        video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
        # obs video
        path_obs = []
        video_obs = []
        if is_sub_video:
            for name in self._img_names:
                path_obs.append(f"{self.save_path}_{name}.avi")
                width, height = self.obs_all[0][name].shape[3]*self.obs_all[0][name].shape[0], self.obs_all[0][name].shape[2]
                video_obs.append(cv2.VideoWriter(path_obs[-1], cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height)))

        # 将图片写入视频
        for image, t, obs in zip(self.render_image_all, self.t, self.obs_all):
            video.write(image)
            if is_sub_video:
                for i, name in enumerate(self._img_names):
                    if "depth" in name:
                        max_depth = 10
                        img = np.clip(np.hstack(np.transpose(obs[name], (0, 2, 3, 1))),None, max_depth)
                        img = (cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)*255/max_depth).astype(np.uint8)
                        video_obs[i].write(img)
                    elif "color" in name:
                        img = np.hstack(np.transpose(obs[name], (0, 2, 3, 1)))
                        img = img.astype(np.uint8)
                        video_obs[i].write(img)
                        # img = (cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).astype(np.uint8)
                        video_obs[i].write(img)

        cv2.imwrite(f"{self.save_path}_render.jpg", image)

        video.release()
        if is_sub_video:
            for i in range(len(video_obs)):
                video_obs[i].release()

        print(f"video saved in {path}")
        # raise NotImplementedError
