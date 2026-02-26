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
from scipy.spatial.transform import Slerp
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
        self.m = 0.58
        # self.bag_file = '/home/suncc/suncc/My-research/Vision-based-Racing/waypoint1.bag'
        self.bag_file = '/home/suncc/suncc/My-research/Vision-based-Racing/pid_test_all.bag'
        self.bag = rosbag.Bag(self.bag_file)
        self.topics = ['/bfctrl/cmd', '/mavros/setpoint_raw/attitude', '/bfctrl/local_odom', '/mavros/imu/data']
        self.actions = []
        self.actions_real = []
        self.action_drone = []
        self.anglevel = []
        self.thrust = []
        self.reward_all = []
        self.action_all = []
        self.state_all = []
        self.state_bf = []
        self.state_bf_time = []
        self.obs_all = []
        self.position_all = []
        self.actions_time = []  
        self.ctrli = []
        self.action_drone_time = []  
        self.imu_angle_v = []
        self.imu_time = []
        
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
        self.align_data()
        self.load("configs/example.json")
        # self.print_actions()
        self.simulator_actions()
        self.plot()
        
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
            
            if topic == '/mavros/imu/data':
                angle_x, angle_y, angle_z = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
                self.imu_angle_v.append(np.array([angle_x, angle_y, angle_z], dtype=np.float32))
                self.imu_time.append(t.to_sec()) 
                
                 
                
        self.actions_real = th.as_tensor(self.actions.copy())
        self.state_bf = th.as_tensor(self.state_bf)
        self.action_drone = th.as_tensor(self.action_drone)
        self.actions_time = np.array(self.actions_time)
        self.action_drone_time = np.array(self.action_drone_time)
        # self.imu_angle_v = th.as_tensor(self.imu_angle_v)
        
        # Clip actions from 550 to 1250
        # mask = (self.actions_time >= 4.4 * 1.74161342e9) & (self.actions_time <= 6.0 * 1.74161342e9)
        # self.actions = np.array(self.actions)[mask]
        # self.actions_time = self.actions_time[mask]
        self.actions_role = self.actions
        # self.actions_role = np.zeros_like(self.actions_role)
                    
    def load(self, path=""):
        with open(path, "r") as f:
            data = json.load(f)
        self._bd_rate = bound(
            max=th.tensor(data["max_rate"]), min=th.tensor(-data["max_rate"])
        )
        
    def normalize(self, command, normal_range: [float, float] = (-1, 1)):
        # thrust
        thrust_scale = 1 * HOVER_ACC
        normalized_thrust = (command[:, :1] - 1 * HOVER_ACC) / thrust_scale

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
    
    def align_data(self):
        # """将action_drone的数据插值对齐到actions_time的时间戳"""
        # 将action_drone数据转换为numpy数组
        action_drone_np = self.action_drone.numpy()
        
        # 创建插值函数（对每个维度单独插值）
        self.action_drone_aligned = np.zeros_like(self.actions_real.numpy())
        for i in range(4):  # 遍历thrust, anglex, angley, anglez四个维度
            interp_func = interpolate.interp1d(
                self.action_drone_time,  
                action_drone_np[:, i],    
                kind='linear',            
                fill_value="extrapolate"  # 超出范围时外推
            )
            self.action_drone_aligned[:, i] = interp_func(self.actions_time)
            
        # Align position
        state_bf_np = self.state_bf.numpy()
        state_bf_time = np.array(self.state_bf_time)
        
        self.state_bf_aligned = np.zeros((len(self.actions_time), 3))
        for i in range(3):
            interp_func = interpolate.interp1d(
                state_bf_time, 
                state_bf_np[:, i],
                kind='linear', 
                fill_value="extrapolate"
            )
            self.state_bf_aligned[:, i] = interp_func(self.actions_time)

        # Align orientation
        state_bf_np = self.state_bf.numpy()
        state_bf_time = np.array(self.state_bf_time)
        quaternions = state_bf_np[:, 3:7]  # [qw, qx, qy, qz]

        # 转换为 SciPy 的 Rotation 对象（注意顺序为 [x, y, z, w]）
        rotations = R_scipy.from_quat(quaternions[:, [1, 2, 3, 0]])  # 转换为 [x, y, z, w]
        times = state_bf_time
        key_rots = R_scipy.concatenate([rotations])
        slerp = Slerp(times, key_rots)  # 创建插值路径

        # 对齐到 actions_time 的时间戳
        self.state_bf_ori_aligned = []
        for t in self.actions_time:
            if t < times[0]:
                q = rotations[0].as_quat()  # 早于第一个时间点，取第一个姿态
            elif t > times[-1]:
                q = rotations[-1].as_quat()  # 晚于最后一个时间点，取最后一个姿态
            else:
                q = slerp(t).as_quat()  # 插值获取四元数
            # 转换回 [qw, qx, qy, qz] 格式
            self.state_bf_ori_aligned.append([q[3], q[0], q[1], q[2]])
        
        self.state_bf_ori_aligned = np.array(self.state_bf_ori_aligned)
        
        # Align imu data
        imu_angle_v_np = np.array(self.imu_angle_v)
        imu_time_np = np.array(self.imu_time)
        
        # Initialize aligned IMU data container
        self.imu_angle_v_aligned = np.zeros((len(self.actions_time), 3))
        
        # Interpolate each angular velocity component
        for i in range(3):
            # Create interpolation function
            interp_func = interpolate.interp1d(
                imu_time_np,
                imu_angle_v_np[:, i],
                kind='linear',
                fill_value="extrapolate"
            )
            # Interpolate to action timestamps
            self.imu_angle_v_aligned[:, i] = interp_func(self.actions_time)
            
        
    def simulator_actions(self):
        obs = self.env.reset()
        # self.actions = self.normalize(th.as_tensor(self.actions))
        self.actions_role = self.normalize(th.as_tensor(self.actions_role))
        # self.actions_role = th.tensor(self.actions_role)
        for action in self.actions_role:
            action[0] = 0
            action = action.unsqueeze(0)
            print(f"Action: {action}")
            obs, reward, done, info = self.env.step(action) 
            self.obs_all.append(obs)
            self.reward_all.append(reward)
            self.action_all.append(action)
            self.state_all.append(self.env.state)
            self.anglevel.append(self.env.angular_velocity)
            self.ctrli.append(self.env.envs.dynamics._ctrl_i)
            # self.thrust.append(self.env.thrust)

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
        self.imu_angle_v = np.stack(self.imu_angle_v)
        # fig_ctbr, ax = plot.subplots(2, 2, figsize=(25, 15))
        fig_ctbr1, ax = plot.subplots(figsize=(35, 10))
        ax.plot(self.actions_time[0:170], self.actions_real[0:170, 1].numpy(), label="anglex_cmd_real", color=colors[0])
        ax.plot(self.actions_time[0:170], self.anglevel[0:170, 0].numpy(), label="anglex_sim", color=colors[1])
        ax.plot(self.actions_time[0:170], self.imu_angle_v_aligned[337:507,0], label="anglex_imu_real", color=colors[2])
        ax.set_title("anglevel_x")
        ax.legend()
        
        # fig2,ax2 = plot.subplots(figsize=(35, 10))
        # ax2.plot(self.imu_angle_v[1400:1600,0], label="anglex_imu_real", color=colors[2])
        # ax2.set_title("anglevel_x")
        # ax2.legend()
        
        # fig_ctbr2, ay = plot.subplots(figsize=(35, 10))
        # ay.plot(self.actions_time[0:190], self.actions_real[0:190, 2].numpy(), label="angley_real", color=colors[0])
        # # ax.plot(self.actions_time, self.action_drone_aligned[:, 3], label="anglex_drone", color=colors[2])
        # ay.plot(self.actions_time[0:190], self.anglevel[0:190, 1].numpy(), label="angley_sim", color=colors[1])
        # ay.set_title("anglevel_y")
        # ay.legend()
        
        # fig_ctbr3, az = plot.subplots(figsize=(35, 10))
        # az.plot(self.actions_time[0:190], self.actions_real[0:190, 3].numpy(), label="anglez_real", color=colors[0])
        # # ax.plot(self.actions_time, self.action_drone_aligned[:, 3], label="anglex_drone", color=colors[2])
        # az.plot(self.actions_time[0:190], self.anglevel[0:190, 2].numpy(), label="anglez_sim", color=colors[1])
        # az.set_title("anglevel_z")
        # az.legend()
        
        # fig_1_1, posiiton_x = plot.subplots(figsize=(35, 10))
        # posiiton_x.plot(self.actions_time, self.state_all[:,0].numpy(), label=["x_sim"])
        # posiiton_x.plot(self.actions_time, self.state_bf_aligned[:,0], label=["x_real"])
        # posiiton_x.set_title("pos_x")
        # posiiton_x.legend()
        
        # fig_1_2, posiiton_y = plot.subplots(figsize=(35, 10))
        # posiiton_y.plot(self.actions_time, self.state_all[:,1].numpy(), label=["y_sim"])
        # posiiton_y.plot(self.actions_time, self.state_bf_aligned[:,1], label=["y_real"])
        # posiiton_y.set_title("pos_y")
        # posiiton_y.legend()
        
        # fig_1_3, posiiton_z = plot.subplots(figsize=(35, 10))
        # posiiton_z.plot(self.actions_time, self.state_all[:,2].numpy(), label=["z_sim"])
        # posiiton_z.plot(self.actions_time, self.state_bf_aligned[:,2], label=["z_real"])
        # posiiton_z.set_title("pos_z")
        # posiiton_z.legend()
        
        # fig_2, ori_w = plot.subplots(figsize=(35, 10))
        # ori_w.plot(self.state_all[0:200,3].numpy(), label=["qw_sim"])
        # ori_w.plot(self.state_bf_ori_aligned[0:200,0], label=["qw_real"])
        # ori_w.set_title("orientation")
        # ori_w.legend()
        
        # fig_3, ori_x = plot.subplots(figsize=(35, 10))
        # ori_x.plot(self.state_all[0:200,4].numpy(), label=["qx_sim"])
        # ori_x.plot(self.state_bf_ori_aligned[0:200,1], label=["qx_real"])
        # ori_x.set_title("orientation")
        # ori_x.legend()
        
        # fig_4, ori_y = plot.subplots(figsize=(35, 10))
        # ori_y.plot(self.state_all[0:200,5].numpy(), label=["qy_sim"])
        # ori_y.plot(self.state_bf_ori_aligned[0:200,2], label=["qy_real"])
        # ori_y.set_title("orientation")
        # ori_y.legend()
        
        # fig_5, ori_z = plot.subplots(figsize=(35, 10))
        # ori_z.plot(self.state_all[0:200,6].numpy(), label=["qz_sim"])
        # ori_z.plot(self.state_bf_ori_aligned[0:200,3], label=["qz_real"])
        # ori_z.set_title("orientation")
        # ori_z.legend()
        
        plot.show()
        # fig_ctbr.savefig("ctbr.png", dpi=1600)
            
        
bag_data = bag_plot()