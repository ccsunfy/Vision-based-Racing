import rosbag
import numpy as np
import os
import sys
import torch as th
import matplotlib.pyplot as plt
import json
sys.path.append(os.getcwd())
from utils.FigFashion.FigFashion import FigFon
from envs.demo2_3Dcircle_onboard import RacingEnv2
from scipy import interpolate
from utils.type import bound
from utils.FigFashion.color import colorsets
from scipy.spatial.transform import Rotation as R

# 设置与TensorBoard代码相同的字体和格式
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Calibri'],  # 使用Calibri字体
    'font.size': 24,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'text.usetex': False
})

MODE_CHANNEL = 6 
HOVER_ACC = 9.81 
MODE_SHIFT_VALUE = 0.25
colors = colorsets["Modern Scientific"]

class SimplifiedBagPlot:
    def __init__(self, path, step_type=None):
        self.scene_path = "datasets/spy_datasets/configs/garage_empty"
        self.m = 0.57
        self.bag_file = path
        self.step_type = step_type  # 'x', 'y', 'z', or 'thrust'
        self.bag = rosbag.Bag(self.bag_file)
        self.topics = ['/bfctrl/cmd', '/bfctrl/local_odom', '/mavros/imu/data', '/bfctrl/traj_start_trigger']
        
        self.actions_cmd = []
        self.actions_time = []
        self.state_bf = []
        self.state_bf_time = []
        self.imu_angle_v = []
        self.imu_time = []
        self.trigger = []
        self.trigger_time = []

        self.anglevel = []
        self.accz = []
        self.action_all = []
        self.state_all = []
        self.sim_t = []
        
        self.env = RacingEnv2(
            num_agent_per_scene=1,
            visual=False, 
            max_episode_steps=512,
            scene_kwargs={"path": self.scene_path},
            latent_dim=256
        )
        
        self.bag_parser()
        self.align_data()
        self.load("configs/example.json")
        self.simulator_actions()
        
    def bag_parser(self):
        for topic, msg, t in self.bag.read_messages(self.topics): 
            if topic == '/bfctrl/cmd':
                ctbr_x, ctbr_y, ctbr_z, thrust = msg.angularVel.x, msg.angularVel.y, msg.angularVel.z, msg.thrust
                self.actions_cmd.append(np.array([thrust, ctbr_x, ctbr_y, ctbr_z], dtype=np.float32))
                self.actions_time.append(t.to_sec())
                
            if topic == '/bfctrl/traj_start_trigger':
                trigger = msg.data
                self.trigger.append(np.array(trigger, dtype=np.float32))
                self.trigger_time.append(t.to_sec())
                
            if topic == '/bfctrl/local_odom':
                pos_x, pos_y, pos_z = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z
                ori_w, ori_x, ori_y, ori_z = msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z
                vel_x, vel_y, vel_z = msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z
                ang_x, ang_y, ang_z = msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z
                
                self.state_bf.append(np.array([
                    pos_x, pos_y, pos_z, ori_w, ori_x, ori_y, ori_z, 
                    vel_x, vel_y, vel_z, ang_x, ang_y, ang_z
                ], dtype=np.float32))
                self.state_bf_time.append(t.to_sec())
            
            if topic == '/mavros/imu/data':
                angle_x, angle_y, angle_z, acc_z = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z, msg.linear_acceleration.z
                self.imu_angle_v.append(np.array([angle_x, angle_y, angle_z, acc_z], dtype=np.float32))
                self.imu_time.append(t.to_sec())
                
        self.actions_real = th.as_tensor(self.actions_cmd.copy())
        self.state_bf = th.as_tensor(self.state_bf)
        self.state_bf_time = np.array(self.state_bf_time)
        self.actions_time = np.array(self.actions_time)
        self.trigger_time = np.array(self.trigger_time)
        
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

    def align_data(self):
        self.start_time, self.end_time = self.trigger_time[0], self.trigger_time[-1]
        
        def get_cut_data(ori_data, ori_time, start_time, end_time):
            mask = (ori_time >= start_time) & (ori_time <= end_time)
            return ori_data[mask], np.array(ori_time)[mask]
            
        self.ori_action, self.ori_action_time = get_cut_data(
            self.actions_real, self.actions_time, self.start_time, self.end_time)
        
        self.imu_angle_v, self.imu_time = get_cut_data(
            np.array(self.imu_angle_v), self.imu_time, self.start_time, self.end_time)
        
        self.state_bf, self.state_bf_time = get_cut_data(
            self.state_bf, self.state_bf_time, self.start_time, self.end_time)
        
        self.ori_action_time -= self.start_time
        self.imu_time -= self.start_time
        self.state_bf_time -= self.start_time
        
        act_f = [interpolate.interp1d(
            self.ori_action_time, self.ori_action[:,i],
            kind='nearest',
            fill_value=(self.ori_action[0,i], self.ori_action[-1,i]),
            bounds_error=False
        ) for i in range(self.ori_action.shape[1])]
        
        self.act_f = lambda t: np.array([f(t) for f in act_f]).T
        
    def simulator_actions(self):
        self.env.reset()
        
        while self.env.envs.dynamics.t < self.end_time - self.start_time:
            action = self.normalize(th.as_tensor(self.act_f(self.env.envs.dynamics.t)))
            action = th.as_tensor(action, dtype=th.float32)
            
            self.sim_t.append(self.env.envs.dynamics.t.clone().detach())
            self.env.envs.dynamics.step(action)
            
            self.action_all.append(action)
            self.state_all.append(self.env.state)
            self.anglevel.append(self.env.angular_velocity)
            self.accz.append(self.env.acceleration[:,2])
                
        self.anglevel = th.cat(self.anglevel, dim=0)
        self.accz = th.as_tensor(self.accz, dtype=th.float32)
        self.action_all = th.cat(self.action_all, dim=0)
        self.state_all = th.cat(self.state_all, dim=0)
        self.sim_t = th.cat(self.sim_t)
        
    def plot_step_response(self, axs):
        self.imu_angle_v = np.stack(self.imu_angle_v)

        # 只保留前三秒的数据
        time_limit = 3.0
        
        if self.step_type == 'x':
            # 过滤时间数据
            time_mask = self.ori_action_time <= time_limit
            filtered_cmd_time = self.ori_action_time[time_mask]
            filtered_cmd_data = self.ori_action[time_mask, 1]
            
            time_mask_imu = self.imu_time <= time_limit
            filtered_imu_time = self.imu_time[time_mask_imu]
            filtered_imu_data = self.imu_angle_v[time_mask_imu, 0]
            
            time_mask_sim = self.sim_t <= time_limit
            filtered_sim_time = self.sim_t[time_mask_sim]
            filtered_sim_data = self.anglevel.numpy()[time_mask_sim, 0]
            
            axs[0].plot(filtered_cmd_time, filtered_cmd_data, linestyle='--', linewidth=6, label="cmd", color='gray')
            axs[0].plot(filtered_imu_time, filtered_imu_data, linewidth=6, label="real", color=colors[1])
            axs[0].plot(filtered_sim_time, filtered_sim_data, linewidth=6, label="sim", color=colors[2])
            axs[0].set_title('X-axis Angular Velocity', fontsize=28)
            axs[0].set_xlabel('Time (s)', fontsize=28)
            axs[0].set_ylabel('rad/s', fontsize=28)
            axs[0].grid(True, linewidth=2, alpha=0.6, zorder=0)
            axs[0].set_xlim(0, time_limit)
            
        elif self.step_type == 'y':
            # 过滤时间数据
            time_mask = self.ori_action_time <= time_limit
            filtered_cmd_time = self.ori_action_time[time_mask]
            filtered_cmd_data = self.ori_action[time_mask, 2]
            
            time_mask_imu = self.imu_time <= time_limit
            filtered_imu_time = self.imu_time[time_mask_imu]
            filtered_imu_data = self.imu_angle_v[time_mask_imu, 1]
            
            time_mask_sim = self.sim_t <= time_limit
            filtered_sim_time = self.sim_t[time_mask_sim]
            filtered_sim_data = self.anglevel.numpy()[time_mask_sim, 1]
            
            axs[1].plot(filtered_cmd_time, filtered_cmd_data, linestyle='--', linewidth=6, label="cmd", color='gray')
            axs[1].plot(filtered_imu_time, filtered_imu_data, linewidth=6, label="real", color=colors[1])
            axs[1].plot(filtered_sim_time, filtered_sim_data, linewidth=6, label="sim", color=colors[2])
            axs[1].set_title('Y-axis Angular Velocity', fontsize=28)
            axs[1].set_xlabel('Time (s)', fontsize=28)
            axs[1].set_ylabel('rad/s', fontsize=28)
            axs[1].grid(True, linewidth=2, alpha=0.6, zorder=0)
            axs[1].set_xlim(0, time_limit)
            
        elif self.step_type == 'z':
            # 过滤时间数据
            time_mask = self.ori_action_time <= time_limit
            filtered_cmd_time = self.ori_action_time[time_mask]
            filtered_cmd_data = self.ori_action[time_mask, 3]
            
            time_mask_imu = self.imu_time <= time_limit
            filtered_imu_time = self.imu_time[time_mask_imu]
            filtered_imu_data = self.imu_angle_v[time_mask_imu, 2]
            
            time_mask_sim = self.sim_t <= time_limit
            filtered_sim_time = self.sim_t[time_mask_sim]
            filtered_sim_data = self.anglevel.numpy()[time_mask_sim, 2]
            
            axs[2].plot(filtered_cmd_time, filtered_cmd_data, linestyle='--', linewidth=6, label="cmd", color='gray')
            axs[2].plot(filtered_imu_time, filtered_imu_data, linewidth=6, label="real", color=colors[1])
            axs[2].plot(filtered_sim_time, filtered_sim_data, linewidth=6, label="sim", color=colors[2])
            axs[2].set_title('Z-axis Angular Velocity', fontsize=28)
            axs[2].set_xlabel('Time (s)', fontsize=28)
            axs[2].set_ylabel('rad/s', fontsize=28)
            axs[2].grid(True, linewidth=2, alpha=0.6, zorder=0)
            axs[2].set_xlim(0, time_limit)

        elif self.step_type == 'thrust':
            # 过滤时间数据
            time_mask = self.ori_action_time <= time_limit
            filtered_cmd_time = self.ori_action_time[time_mask]
            filtered_cmd_data = self.ori_action[time_mask, 0]
            
            time_mask_imu = self.imu_time <= time_limit
            filtered_imu_time = self.imu_time[time_mask_imu]
            filtered_imu_data = self.imu_angle_v[time_mask_imu, 3]
            
            time_mask_sim = self.sim_t <= time_limit
            filtered_sim_time = self.sim_t[time_mask_sim]
            filtered_sim_data = self.accz.numpy()[time_mask_sim] + 9.8
            
            # 过滤掉小于-5和大于26的数据
            mask_real = (filtered_imu_data >= -5) & (filtered_imu_data <= 26)
            mask_sim = (filtered_sim_data >= -5) & (filtered_sim_data <= 26)
            mask_cmd = (filtered_cmd_data >= -5) & (filtered_cmd_data <= 26)
            
            # 使用掩码过滤数据
            filtered_real = np.where(mask_real, filtered_imu_data, np.nan)
            filtered_sim = np.where(mask_sim, filtered_sim_data, np.nan)
            filtered_cmd = np.where(mask_cmd, filtered_cmd_data, np.nan)
            
            axs[3].plot(filtered_cmd_time, filtered_cmd, linestyle='--', linewidth=6, label="cmd", color='gray')
            axs[3].plot(filtered_imu_time, filtered_real, linewidth=6, label="real", color=colors[1])
            axs[3].plot(filtered_sim_time, filtered_sim, linewidth=6, label="sim", color=colors[2])
            axs[3].set_title('Z-axis Thrust', fontsize=28)
            axs[3].set_xlabel('Time (s)', fontsize=28)
            axs[3].set_ylabel('N', fontsize=28)
            axs[3].grid(True, linewidth=2, alpha=0.6, zorder=0)
            axs[3].set_xlim(0, time_limit)

    def plot_policy_response(self, axs):
        """绘制策略响应曲线，按照1×4布局"""
        self.imu_angle_v = np.stack(self.imu_angle_v)
        
        # ========================== Angular Velocity X ==========================
        axs[0].plot(self.ori_action_time, self.ori_action[:,1], linestyle='--', linewidth=6, label="cmd", color='gray')
        axs[0].plot(self.imu_time, self.imu_angle_v[:,0], linewidth=6, label="real", color=colors[1])
        axs[0].plot(self.sim_t, self.anglevel.numpy()[:,0], linewidth=6, label="sim", color=colors[2])
        axs[0].set_title('X-axis Angular Velocity', fontsize=28)
        axs[0].set_xlabel('Time (s)', fontsize=28)
        axs[0].set_ylabel('rad/s', fontsize=28)
        axs[0].grid(True, linewidth=2, alpha=0.6, zorder=0)

        # ========================== Angular Velocity Y ==========================
        axs[1].plot(self.ori_action_time, self.ori_action[:,2], linestyle='--', linewidth=6, label="cmd", color='gray')
        axs[1].plot(self.imu_time, self.imu_angle_v[:,1], linewidth=6, label="real", color=colors[1])
        axs[1].plot(self.sim_t, self.anglevel.numpy()[:,1], linewidth=6, label="sim", color=colors[2])
        axs[1].set_title('Y-axis Angular Velocity', fontsize=28)
        axs[1].set_xlabel('Time (s)', fontsize=28)
        axs[1].set_ylabel('rad/s', fontsize=28)
        axs[1].grid(True, linewidth=2, alpha=0.6, zorder=0)

        # ========================== Angular Velocity Z ==========================
        axs[2].plot(self.ori_action_time, self.ori_action[:,3], linestyle='--', linewidth=6, label="cmd", color='gray')
        axs[2].plot(self.imu_time, self.imu_angle_v[:,2], linewidth=6, label="real", color=colors[1])
        axs[2].plot(self.sim_t, self.anglevel.numpy()[:,2], linewidth=6, label="sim", color=colors[2])
        axs[2].set_title('Z-axis Angular Velocity', fontsize=28)
        axs[2].set_xlabel('Time (s)', fontsize=28)
        axs[2].set_ylabel('rad/s', fontsize=28)
        axs[2].grid(True, linewidth=2, alpha=0.6, zorder=0)

        # ========================== Thrust Z ==========================
        # 过滤掉小于-5和大于26的数据
        real_data = self.imu_angle_v[:,3]
        sim_data = self.accz.numpy()[:] + 9.8
        cmd_data = self.ori_action[:,0]                                                   
        
        # 创建掩码，只保留在[-5, 26]范围内的数据
        mask_real = (real_data >= -5) & (real_data <= 26)
        mask_sim = (sim_data >= -5) & (sim_data <= 26)
        mask_cmd = (cmd_data >= -5) & (cmd_data <= 26)
        
        # 使用掩码过滤数据
        filtered_real = np.where(mask_real, real_data, np.nan)
        filtered_sim = np.where(mask_sim, sim_data, np.nan)
        filtered_cmd = np.where(mask_cmd, cmd_data, np.nan)
        
        axs[3].plot(self.ori_action_time, filtered_cmd, linestyle='--', linewidth=6, label="cmd", color='gray')
        axs[3].plot(self.imu_time, filtered_real, linewidth=6, label="real", color=colors[1])
        axs[3].plot(self.sim_t, filtered_sim, linewidth=6, label="sim", color=colors[2])
        axs[3].set_title('Z-axis Thrust', fontsize=28)
        axs[3].set_xlabel('Time (s)', fontsize=28)
        axs[3].set_ylabel('N', fontsize=28)
        axs[3].grid(True, linewidth=2, alpha=0.6, zorder=0)

# 统一图形尺寸和格式
fig_size = (32, 6)  # 与TensorBoard代码相同的尺寸
title_fontsize = 24
subtitle_fontsize = 24
label_fontsize = 20
legend_fontsize = 24

# step curves
fig1, axs1 = plt.subplots(1, 4, figsize=fig_size)
# plt.subplots_adjust(wspace=0.3)
# fig1.suptitle('Step Response: Real vs Simulation', fontsize=title_fontsize, y=1.05)
for ax in axs1:
    for spine in ax.spines.values():
        spine.set_linewidth(3)

x_step = SimplifiedBagPlot(path='/home/suncc/suncc/My-research/Vision-based-Racing/x_713_3.bag', step_type='x')
x_step.plot_step_response(axs1)

y_step = SimplifiedBagPlot(path='/home/suncc/suncc/My-research/Vision-based-Racing/y_713_1.bag', step_type='y')
y_step.plot_step_response(axs1)

z_step = SimplifiedBagPlot(path='/home/suncc/suncc/My-research/Vision-based-Racing/z_713_1.bag', step_type='z')
z_step.plot_step_response(axs1)

thrust_step = SimplifiedBagPlot(path='/home/suncc/suncc/My-research/Vision-based-Racing/accz_2g_85.bag', step_type='thrust')
thrust_step.plot_step_response(axs1)

# 添加图例（与TensorBoard代码相同的方式）
handles, labels = axs1[0].get_legend_handles_labels()
fig1.legend(handles, labels, loc='lower center', ncol=3, frameon=False, prop={'size': legend_fontsize})

plt.tight_layout()
plt.subplots_adjust(bottom=0.24)  # 为底部图例留出空间

# policy curves
fig2, axs2 = plt.subplots(1, 4, figsize=fig_size)

for ax in axs2:
    for spine in ax.spines.values():
        spine.set_linewidth(3)
# plt.subplots_adjust(wspace=0.3)
# fig2.suptitle('Policy Response: Real vs Simulation', fontsize=title_fontsize, y=1.05)

# policy_response = SimplifiedBagPlot(path='/home/suncc/suncc/My-research/Vision-based-Racing/demo3_yanzheng_depth_2.bag')
policy_response = SimplifiedBagPlot(path='diff-land-cylinder-115.bag')
policy_response.plot_policy_response(axs2)

# 添加图例（与TensorBoard代码相同的方式）
handles, labels = axs2[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc='lower center', ncol=3, frameon=False, prop={'size': legend_fontsize})

plt.tight_layout()
plt.subplots_adjust(bottom=0.24)  # 为底部图例留出空间

# 保存图像
fig1.savefig('step_response.pdf', format='pdf', bbox_inches='tight', dpi=400)
fig2.savefig('policy_response.pdf', format='pdf', bbox_inches='tight', dpi=400)

plt.show()