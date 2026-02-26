import rosbag
import rospy
import numpy as np
import os
import sys
import torch as th
import matplotlib.pyplot as plot
import json
import time
sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as R_scipy
from std_msgs.msg import Float32MultiArray  # 假设动作数据是 Float32MultiArray 类型
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from sensor_msgs.msg import Imu
from torch.nn import functional as F
from envs.waypoint_test import RacingEnv2
from scipy.spatial.transform import Rotation as R_scipy
from scipy import interpolate
from utils.type import bound
from utils.FigFashion.color import colorsets


MODE_CHANNEL = 6 
HOVER_ACC = 9.81
MODE_SHIFT_VALUE = 0.25
colors = colorsets["Modern Scientific"]

class bag_plot:
    def __init__(self):
        
        # self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.scene_path = "datasets/spy_datasets/configs/garage_empty"
        self.m = 0.68
        self.bag_file = '/home/suncc/suncc/My-research/Vision-based-Racing/pid_test_all.bag'
        self.bag = rosbag.Bag(self.bag_file)
        self.actions = []
        self.actions_real = []
        self.action_drone = []
        self.anglevel = []
        self.thrust = []
        self.reward_all = []
        self.action_all = []
        self.state_all = []
        self.obs_all = []
        self.position_all = []
        self.actions_time = []  
        self.ctrli = []
        self.action_drone_time = []  
        self.state_bf = []
        self.state_bf_time = []
        
        self.env= RacingEnv2(num_agent_per_scene=1,
                            # num_agent_per_scene=training_params["num_env"]/2,
                            # 如果需要开启多个环境，需要设置num_scene
                                # num_scene=1,
                                visual=False, # 不用视觉要改成False
                                max_episode_steps=512,
                                scene_kwargs={
                                    "path": self.scene_path,
                                },
                                latent_dim=256
                                )
        
        self.bag_parser()
        # self.align_data()
        self.load("configs/example.json")
        self.print_actions()
        self.simulator_actions()
        self.plot()
        
    # def bag_parser(self):
    #     # self.actions = np.zeros((num_points, 4), dtype=np.float32)  # 示例
    #     # 生成正弦信号参数
    #     freq = 30.0  # 正弦波频率 (Hz)
    #     amplitude = 0.01  # 角速度幅度 (rad/s)
    #     duration = 10.0  # 信号持续时间 (秒)
        
    #     # 生成时间序列 (使用实际bag的时间戳或新建时间轴)
    #     start_time = self.bag.get_start_time()
    #     end_time = self.bag.get_end_time()
    #     t = np.linspace(start_time, end_time, int((end_time - start_time)*30))  # 30Hz采样
        
    #     # 生成纯x轴正弦角速度信号
    #     ctbr_y= amplitude * np.sin(2 * np.pi * freq * (t - start_time))
    #     actions = np.zeros((len(t), 4),dtype=np.float32)  # [thrust, ctbr_x, ctbr_y, ctbr_z]
    #     actions[:, 1] = ctbr_y
    #     actions[:, 0] = self.m * HOVER_ACC  # 0.5 * m * g
    #     # 赋值到类变量
    #     self.actions = actions
    #     self.actions_time = t
        
    #     # 其他保持原逻辑
    #     self.actions_real = th.as_tensor(self.actions.copy())
    #     # self.actions_role = self.actions[558:800]
    
    def bag_parser(self):
        
        for topic, msg, t in self.bag.read_messages(self.topics):  
            # if topic == self.topic_dict['trigger'] and msg.data and self.trigger_t < 0:
            #     self.trigger_t = t.to_sec()
                
            # if topic == self.topic_dict['trigger'] and not msg.data and self.trigger_t > 0:
            #     self.trigger_t = -1.0
            
            # if self.trigger_t < 0:
                # continue
            if topic == '/bfctrl/cmd':
                ctbr_x , ctbr_y, ctbr_z, thrust = msg.angularVel.x, msg.angularVel.y, msg.angularVel.z, msg.thrust
                self.actions.append(np.array([thrust, ctbr_x, ctbr_y, ctbr_z], dtype=np.float32))  # 假设动作数据存储在 msg.data 中
                self.actions_time.append(t.to_sec())  
                
            if topic == '/mavros/setpoint_raw/attitude':
                thrust, anglevel_x, anglevel_y, anglevel_z = msg.thrust, msg.body_rate.x, msg.body_rate.y, msg.body_rate.z
                self.action_drone.append(np.array([thrust, anglevel_x, anglevel_y, anglevel_z], dtype=np.float32))
                self.action_drone_time.append(t.to_sec())
                
            if topic =='/bfctrl/local_odom':
                pos_x, pos_y, pos_z, ori_w, ori_x, ori_y, ori_z = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z
                self.state_bf.append(np.array([pos_x, pos_y, pos_z, ori_w, ori_x, ori_y, ori_z], dtype=np.float32))
                self.state_bf_time.append(t.to_sec()) 
                
        amplitude = 0.5  # 阶跃幅度 (rad/s)
        total_duration = 0.2 + 0.3 + 0.1 + 0.1  # 总持续时间7秒
        
        # 生成时间序列 (从0开始)
        t = np.linspace(0, total_duration, int(total_duration * 200))  # 30Hz采样
        
        # 初始化角速度信号
        ctbr_y = np.zeros_like(t)
        
        # 按时间段设置信号
        time_segments = [
            (0, 0.2, amplitude),    # 0-2秒 正阶跃
            (0.2, 0.5, -amplitude),   # 2-5秒 负阶跃 
            (0.5, 0.6, amplitude),    # 5-6秒 正阶跃
            (0.6, 0.7, -amplitude)    # 6-7秒 负阶跃
        ]
        
        # 应用阶跃信号
        for start, end, value in time_segments:
            mask = (t >= start) & (t < end)
            ctbr_y[mask] = value
        
        # 构造动作数组
        actions = np.zeros((len(t), 4), dtype=np.float32)  # [thrust, ctbr_x, ctbr_y, ctbr_z]
        actions[:, 1] = ctbr_y
        actions[:, 0] = self.m * HOVER_ACC  # 保持悬停推力
        
        # 赋值到类变量
        self.actions = actions
        self.actions_time = t
        
        # 其他保持原逻辑
        self.actions_real = th.as_tensor(self.actions.copy())
                    
    def load(self, path=""):
        with open(path, "r") as f:
            data = json.load(f)
        self._bd_rate = bound(
            max=th.tensor(data["max_rate"]), min=th.tensor(-data["max_rate"])
        )
        
    def normalize(self, command, normal_range: [float, float] = (-1, 1)):
        # thrust
        thrust_scale = 1 * HOVER_ACC
        normalized_thrust = (command[:, :1] / self.m - 1 * HOVER_ACC) / thrust_scale

        # anglevel
        bodyrate_scale = (self._bd_rate.max - self._bd_rate.min) / (normal_range[1] - normal_range[0])
        bodyrate_bias = self._bd_rate.max - bodyrate_scale * normal_range[1]
        normalized_bodyrate = (command[:, 1:] - bodyrate_bias) / bodyrate_scale

        normalized_command = th.hstack([normalized_thrust, normalized_bodyrate])
        return normalized_command

    def print_actions(self):
        for action in self.actions:
            print(action)
            
    #     self.action_drone_aligned = self.action_drone[drone_mask]
    
        
    def simulator_actions(self):
        obs = self.env.reset()
        self.actions = self.normalize(th.as_tensor(self.actions))
        # self.actions_role = self.normalize(th.as_tensor(self.actions_role))
        # self.actions_role = th.tensor(self.actions_role)
        for action in self.actions:
            action = action.unsqueeze(0)
            obs, reward, done, info = self.env.step(action) 
            self.obs_all.append(obs)
            self.reward_all.append(reward)
            self.action_all.append(action)
            self.state_all.append(self.env.state)
            self.anglevel.append(self.env.angular_velocity)
            self.ctrli.append(self.env.envs.dynamics._ctrl_i)
            # self.thrust.append(self.env.thrust)
            
            # print(f"Observation: {obs}, Reward: {reward}, Done: {done}")

            if done:
                obs = self.env.reset()  # 如果 episode 结束，重置环境
                
        self.anglevel = th.cat(self.anglevel, dim=0)
        # self.thrust = th.cat(self.thrust, dim=0)
        # self.reward_all = th.tensor(self.reward_all)
        self.action_all = th.cat(self.action_all, dim=0)
        self.state_all = th.cat(self.state_all, dim=0)
        self.ctrli = th.cat(self.ctrli, dim=1)
        # self.obs_all = th.cat(self.obs_all, dim=0)
        # self.position_all = th.cat(self.position_all, dim=0)
    
    def plot(self):
        # 执行时间戳对齐
        # self.align_data()
        
        # fig_ctbr, ax = plot.subplots(2, 2, figsize=(25, 15))
        fig_ctbr, ax = plot.subplots(figsize=(35, 10))
        ax.plot(self.actions_time, self.actions_real[:, 1].numpy(), label="anglez_real", color=colors[0])
        # ax.plot(self.actions_time, self.action_drone_aligned[:, 1], label="anglex_drone", color=colors[2])
        ax.plot(self.actions_time, self.anglevel[:, 0].numpy(), label="anglez_sim", color=colors[1])
        ax.set_title("anglevel_x")
        ax.legend()

        fig_ctbr1, ax1 = plot.subplots(2,1, figsize=(35, 10))
        ax1[0].plot(self.actions_time, self.state_all[:,:3].numpy(), label=["x", "y", "z"])
        # ax[1].plot(self.actions_time, self.anglevel[:, 0].numpy(), label="anglex_sim", color=colors[1])
        ax1[0].set_title("pos")
        ax1[0].legend()
        
        ax1[1].plot(self.actions_time, self.state_all[:,3:7].numpy(), label=["qx", "qy", "qz", "qw"])
        # ax[1].plot(self.actions_time, self.anglevel[:, 0].numpy(), label="anglex_sim", color=colors[1])
        ax1[1].set_title("ori")
        ax1[1].legend()
        
        plot.show()
        fig_ctbr.savefig("without_d.png", dpi=1600)            
        
bag_data = bag_plot()