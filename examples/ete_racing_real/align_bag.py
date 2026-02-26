import rosbag
import numpy as np
import os
import sys
import torch as th
import matplotlib.pyplot as plt
import json
sys.path.append(os.getcwd())
# from envs.demo3_ellipse_onboard import RacingEnv2
from utils.FigFashion.FigFashion import FigFon
from envs.demo2_3Dcircle_onboard import RacingEnv2
from scipy import interpolate
from utils.type import bound
from utils.FigFashion.color import colorsets
from scipy.spatial.transform import Rotation as R

MODE_CHANNEL = 6 
HOVER_ACC = 9.81 
MODE_SHIFT_VALUE = 0.25
colors = colorsets["Modern Scientific"]

class bag_plot:
    def __init__(self,path):
        
        # self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.scene_path = "datasets/spy_datasets/configs/garage_empty"
        self.m = 0.57
        self.bag_file = path
        # self.bag_file = '/home/suncc/suncc/My-research/Vision-based-Racing/pid_onboard3.bag'
        # self.bag_file = '/home/suncc/suncc/My-research/Vision-based-Racing/cc_demo1_624.bag'
        # self.bag_file = '/home/suncc/suncc/My-research/Vision-based-Racing/cc_demo3_final_624.bag'
        # self.bag_file = '/home/suncc/suncc/My-research/Vision-based-Racing/accz_onboard_1.bag'
        # self.bag_file = '/home/suncc/suncc/My-research/Vision-based-Racing/accz_2g.bag'
        # self.bag_file = '/home/suncc/suncc/My-research/Vision-based-Racing/x_713_3.bag'
        # self.bag_file = '/home/suncc/suncc/My-research/Vision-based-Racing/y_713_1.bag'
        # self.bag_file = '/home/suncc/suncc/My-research/Vision-based-Racing/z_713_1.bag'
        # self.bag_file = '/home/suncc/suncc/My-research/Vision-based-Racing/demo2_76_1.bag'
        # self.bag_file = '/home/suncc/suncc/My-research/Vision-based-Racing/demo3_yanzheng_depth_3.bag'
        # self.bag_file = '/home/suncc/suncc/My-research/Vision-based-Racing/demo2_724_2131.bag'
        self.bag = rosbag.Bag(self.bag_file)
        self.topics = ['/bfctrl/cmd', '/mavros/setpoint_raw/attitude', '/bfctrl/local_odom', '/mavros/imu/data','/bfctrl/traj_start_trigger']
        self.actions_cmd = []
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
        self.trigger = []
        self.trigger_time = []
        self.accz = []
        self.env= RacingEnv2(num_agent_per_scene=1,
                                # num_scene=1,
                                visual=False, 
                                max_episode_steps=512,
                                scene_kwargs={
                                    "path": self.scene_path,
                                },
                                latent_dim=256
                                )
        
        self.bag_parser()
        self.align_data()
        self.load("configs/example.json")
        self.simulator_actions()
        # self.plot()
        
    def bag_parser(self):
        for topic, msg, t in self.bag.read_messages(self.topics): 

            if topic == '/bfctrl/cmd':
                ctbr_x , ctbr_y, ctbr_z, thrust = msg.angularVel.x, msg.angularVel.y, msg.angularVel.z, msg.thrust
                self.actions_cmd.append(np.array([thrust, ctbr_x, ctbr_y, ctbr_z], dtype=np.float32))  # 假设动作数据存储在 msg.data 中
                self.actions_time.append(t.to_sec())  
                
            if topic == '/bfctrl/traj_start_trigger':
                trigger = msg.data
                self.trigger.append(np.array(trigger, dtype=np.float32))  # 假设动作数据存储在 msg.data 中
                self.trigger_time.append(t.to_sec())  
                
            if topic == '/mavros/setpoint_raw/attitude':
                thrust, anglevel_x, anglevel_y, anglevel_z = msg.thrust, msg.body_rate.x, msg.body_rate.y, msg.body_rate.z
                self.action_drone.append(np.array([thrust, anglevel_x, anglevel_y, anglevel_z], dtype=np.float32))
                self.action_drone_time.append(t.to_sec())
                
            if topic =='/bfctrl/local_odom':
                pos_x, pos_y, pos_z, ori_w, ori_x, ori_y, ori_z, vel_x, vel_y, vel_z, ang_x, ang_y, ang_z = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z, msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z
                self.state_bf.append(np.array([pos_x, pos_y, pos_z, ori_w, ori_x, ori_y, ori_z, vel_x, vel_y, vel_z, ang_x, ang_y, ang_z], dtype=np.float32))
                self.state_bf_time.append(t.to_sec()) 
            
            if topic == '/mavros/imu/data':
                angle_x, angle_y, angle_z, acc_z = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z, msg.linear_acceleration.z
                self.imu_angle_v.append(np.array([angle_x, angle_y, angle_z, acc_z], dtype=np.float32))
                self.imu_time.append(t.to_sec())
                
                 
                
        self.actions_real = th.as_tensor(self.actions_cmd.copy())
        self.state_bf = th.as_tensor(self.state_bf)
        self.action_drone = th.as_tensor(self.action_drone)
        self.state_bf_time = np.array(self.state_bf_time)
        self.actions_time = np.array(self.actions_time)
        self.trigger_time = np.array(self.trigger_time)
        self.action_drone_time = np.array(self.action_drone_time)
        

        self.actions_role = self.actions_cmd
                    
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

    
    def align_data(self):
        self.start_time, self.end_time = self.trigger_time[0], self.trigger_time[-1]
        # self.start_time, self.end_time = self.actions_time[0], self.actions_time[-1]
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
        
        self.act_f = lambda t:  np.array([f(t) for f in act_f]).T
        
    def simulator_actions(self):
        self.env.reset()
        self.action_role = self.normalize(th.as_tensor(self.act_f(self.env.envs.dynamics.t)))
        self.sim_t = []
        while self.env.envs.dynamics.t < self.end_time-self.start_time:
            action= self.normalize(th.as_tensor(self.act_f(self.env.envs.dynamics.t)))
            # action[:,0] = 0
            # action = action.unsqueeze(0)
            # print(f"Action: {action}")
            self.sim_t.append(self.env.envs.dynamics.t.clone().detach())
            action = th.as_tensor(action, dtype=th.float32)
            self.env.envs.dynamics.step(action) 
            # self.obs_all.append(obs)
            self.action_all.append(action)
            self.state_all.append(self.env.state)
            # self.state_all.append(self.env.position)
            self.anglevel.append(self.env.angular_velocity)
            self.accz.append(self.env.acceleration[:,2])
                
        self.anglevel = th.cat(self.anglevel, dim=0)
        self.accz = th.as_tensor(self.accz, dtype=th.float32)
        self.action_all = th.cat(self.action_all, dim=0)
        # self.obs_all = th.cat(self.obs_all, dim=0)
        self.state_all = th.cat(self.state_all, dim=0)
        self.sim_t = th.cat(self.sim_t)
        
    def plot_full(self, axs2):
        env= RacingEnv2(num_agent_per_scene=1,
                                    # num_scene=1,
                                    visual=False, 
                                    max_episode_steps=512,
                                    scene_kwargs={
                                        "path": self.scene_path,
                                    },
                                    latent_dim=256
                                    )
        def quaternion_to_euler(quaternions):
            """
            将四元数 (w, x, y, z) 转换为欧拉角 (roll, pitch, yaw) 弧度
            参数:
                quaternions: (N, 4) 数组，顺序为 [w, x, y, z]
            返回:
                euler_angles: (N, 3) 数组，顺序为 [roll, pitch, yaw]
            """
            # 调整顺序为 scipy 需要的 (x, y, z, w)
            adjusted_quats = np.zeros((quaternions.shape[0], 4))
            adjusted_quats[:, 0] = quaternions[:, 1]  # x
            adjusted_quats[:, 1] = quaternions[:, 2]  # y
            adjusted_quats[:, 2] = quaternions[:, 3]  # z
            adjusted_quats[:, 3] = quaternions[:, 0]  # w
            
            # 创建 Rotation 对象并转换为欧拉角
            rot = R.from_quat(adjusted_quats)
            euler_angles = rot.as_euler('xyz', degrees=False)  # 返回弧度
            return euler_angles
        
        self.imu_angle_v = np.stack(self.imu_angle_v)
        self.state_bf = np.stack(self.state_bf)
        real_euler = quaternion_to_euler(self.state_bf[:, 3:7])  # [w, x, y, z]
        sim_euler = quaternion_to_euler(-self.state_all.numpy()[:, 3:7])  # 注意：这里取了负
        
        # ========================== Angular Velocity ==========================
        axs2[0,0].plot(self.ori_action_time, self.ori_action[:,1], linestyle='--', linewidth=1.8, label="cmd", color='gray')
        axs2[0,0].plot(self.imu_time, self.imu_angle_v[:,0], linewidth=2.5, label="anglevelx_real", color=colors[1])
        axs2[0,0].plot(self.sim_t, self.anglevel.numpy()[:,0], linewidth=2.5, label="anglevelx_sim", color=colors[2])
        axs2[0,0].set_title('Angular Velocity (X-axis)', fontsize=28, fontweight='bold', pad=10)
        axs2[0,0].set_xlabel('Time (s)', fontsize=24)
        axs2[0,0].set_ylabel('rad/s', fontsize=24)
        axs2[0,0].legend(fontsize=20, prop={'weight':'bold'})
        axs2[0,0].tick_params(axis='both', labelsize=18)

        axs2[0,1].plot(self.ori_action_time, self.ori_action[:,2], linestyle='--', linewidth=1.8, label="cmd", color='gray')
        axs2[0,1].plot(self.imu_time, self.imu_angle_v[:,1], linewidth=2.5, label="anglevely_real", color=colors[1])
        axs2[0,1].plot(self.sim_t, self.anglevel.numpy()[:,1], linewidth=2.5, label="anglevely_sim", color=colors[2])
        axs2[0,1].set_title('Angular Velocity (Y-axis)', fontsize=28, fontweight='bold', pad=10)
        axs2[0,1].set_xlabel('Time (s)', fontsize=24)
        axs2[0,1].set_ylabel('rad/s', fontsize=24)
        axs2[0,1].legend(fontsize=20, prop={'weight':'bold'})
        axs2[0,1].tick_params(axis='both', labelsize=18)

        axs2[0,2].plot(self.ori_action_time, self.ori_action[:,3], linestyle='--', linewidth=1.8, label="cmd", color='gray')
        axs2[0,2].plot(self.imu_time, self.imu_angle_v[:,2], linewidth=2.5, label="anglevelz_real", color=colors[1])
        axs2[0,2].plot(self.sim_t, self.anglevel.numpy()[:,2], linewidth=2.5, label="anglevelz_sim", color=colors[2])
        axs2[0,2].set_title('Angular Velocity (Z-axis)', fontsize=28, fontweight='bold', pad=10)
        axs2[0,2].set_xlabel('Time (s)', fontsize=24)
        axs2[0,2].set_ylabel('rad/s', fontsize=24)
        axs2[0,2].legend(fontsize=20, prop={'weight':'bold'})
        axs2[0,2].tick_params(axis='both', labelsize=18)

        # ========================== Orientation (Quaternions) ==========================
        # 横滚角 (Roll)
        axs2[1,0].plot(self.state_bf_time, real_euler[:, 0], linewidth=2.5, label="roll_real", color=colors[1])
        axs2[1,0].plot(self.sim_t, sim_euler[:, 0], linewidth=2.5, label="roll_sim", color=colors[2])
        axs2[1,0].set_title('Roll Angle', fontsize=28, fontweight='bold', pad=10)
        axs2[1,0].set_xlabel('Time (s)', fontsize=24)
        axs2[1,0].set_ylabel('rad', fontsize=24)
        axs2[1,0].legend(fontsize=20, prop={'weight':'bold'})
        axs2[1,0].tick_params(axis='both', labelsize=18)
        axs2[1,0].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        # 俯仰角 (Pitch)
        axs2[1,1].plot(self.state_bf_time, real_euler[:, 1], linewidth=2.5, label="pitch_real", color=colors[1])
        axs2[1,1].plot(self.sim_t, sim_euler[:, 1], linewidth=2.5, label="pitch_sim", color=colors[2])
        axs2[1,1].set_title('Pitch Angle', fontsize=28, fontweight='bold', pad=10)
        axs2[1,1].set_xlabel('Time (s)', fontsize=24)
        axs2[1,1].set_ylabel('rad', fontsize=24)
        axs2[1,1].legend(fontsize=20, prop={'weight':'bold'})
        axs2[1,1].tick_params(axis='both', labelsize=18)
        axs2[1,1].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        # 偏航角 (Yaw)
        axs2[1,2].plot(self.state_bf_time, real_euler[:, 2], linewidth=2.5, label="yaw_real", color=colors[1])
        axs2[1,2].plot(self.sim_t, sim_euler[:, 2], linewidth=2.5, label="yaw_sim", color=colors[2])
        axs2[1,2].set_title('Yaw Angle', fontsize=28, fontweight='bold', pad=10)
        axs2[1,2].set_xlabel('Time (s)', fontsize=24)
        axs2[1,2].set_ylabel('rad', fontsize=24)
        axs2[1,2].legend(fontsize=20, prop={'weight':'bold'})
        axs2[1,2].tick_params(axis='both', labelsize=18)
        axs2[1,2].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # ========================== Linear Velocity ==========================
        axs2[2,0].plot(self.state_bf_time, self.state_bf[:,7], linewidth=2.5, label="velx_real", color=colors[1])
        axs2[2,0].plot(self.sim_t, self.state_all.numpy()[:,7], linewidth=2.5, label="velx_sim", color=colors[2])
        axs2[2,0].set_title('Linear Velocity (X)', fontsize=28, fontweight='bold', pad=10)
        axs2[2,0].set_xlabel('Time (s)', fontsize=24)
        axs2[2,0].set_ylabel('m/s', fontsize=24)
        axs2[2,0].legend(fontsize=20, prop={'weight':'bold'})
        axs2[2,0].tick_params(axis='both', labelsize=18)
        axs2[2,0].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        axs2[2,1].plot(self.state_bf_time, self.state_bf[:,8], linewidth=2.5, label="vely_real", color=colors[1])
        axs2[2,1].plot(self.sim_t, self.state_all.numpy()[:,8], linewidth=2.5, label="vely_sim", color=colors[2])
        axs2[2,1].set_title('Linear Velocity (Y)', fontsize=28, fontweight='bold', pad=10)
        axs2[2,1].set_xlabel('Time (s)', fontsize=24)
        axs2[2,1].set_ylabel('m/s', fontsize=24)
        axs2[2,1].legend(fontsize=20, prop={'weight':'bold'})
        axs2[2,1].tick_params(axis='both', labelsize=18)
        axs2[2,1].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        axs2[2,2].plot(self.state_bf_time, self.state_bf[:,9], linewidth=2.5, label="velz_real", color=colors[1])
        axs2[2,2].plot(self.sim_t, self.state_all.numpy()[:,9], linewidth=2.5, label="velz_sim", color=colors[2])
        axs2[2,2].set_title('Linear Velocity (Z)', fontsize=28, fontweight='bold', pad=10)
        axs2[2,2].set_xlabel('Time (s)', fontsize=24)
        axs2[2,2].set_ylabel('m/s', fontsize=24)
        axs2[2,2].legend(fontsize=20, prop={'weight':'bold'})
        axs2[2,2].tick_params(axis='both', labelsize=18)
        axs2[2,2].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # ========================== Position ==========================
        axs2[3,0].plot(self.state_bf_time, self.state_bf[:,0], linewidth=2.5, label="posx_real", color=colors[1])
        axs2[3,0].plot(self.sim_t, self.state_all.numpy()[:,0], linewidth=2.5, label="posx_sim", color=colors[2])
        axs2[3,0].set_title('Position (X)', fontsize=28, fontweight='bold', pad=10)
        axs2[3,0].set_xlabel('Time (s)', fontsize=24)
        axs2[3,0].set_ylabel('m', fontsize=24)
        axs2[3,0].legend(fontsize=20, prop={'weight':'bold'})
        axs2[3,0].tick_params(axis='both', labelsize=18)
        axs2[3,0].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        axs2[3,1].plot(self.state_bf_time, self.state_bf[:,1], linewidth=2.5, label="posy_real", color=colors[1])
        axs2[3,1].plot(self.sim_t, self.state_all.numpy()[:,1], linewidth=2.5, label="posy_sim", color=colors[2])
        axs2[3,1].set_title('Position (Y)', fontsize=28, fontweight='bold', pad=10)
        axs2[3,1].set_xlabel('Time (s)', fontsize=24)
        axs2[3,1].set_ylabel('m', fontsize=24)
        axs2[3,1].legend(fontsize=20, prop={'weight':'bold'})
        axs2[3,1].tick_params(axis='both', labelsize=18)
        axs2[3,1].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        axs2[3,2].plot(self.state_bf_time, self.state_bf[:,2], linewidth=2.5, label="posz_real", color=colors[1])
        axs2[3,2].plot(self.sim_t, self.state_all.numpy()[:,2], linewidth=2.5, label="posz_sim", color=colors[2])
        axs2[3,2].set_title('Position (Z)', fontsize=28, fontweight='bold', pad=10)
        axs2[3,2].set_xlabel('Time (s)', fontsize=24)
        axs2[3,2].set_ylabel('m', fontsize=24)
        axs2[3,2].legend(fontsize=20, prop={'weight':'bold'})
        axs2[3,2].tick_params(axis='both', labelsize=18)
        axs2[3,2].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
            
        # =========================== Velocity in Body Frame ==========================
        v_world_real = self.state_bf[:, 7:10]  # [vx, vy, vz] in world frame
        v_body_real = np.zeros_like(v_world_real)

        for i in range(len(self.state_bf)):
            quat = self.state_bf[i, 3:7]
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # 注意顺序：scipy使用(x,y,z,w)
            v_body_real[i] = r.apply(v_world_real[i], inverse=True)

        v_world_sim = self.state_all.numpy()[:, 7:10]  # [vx, vy, vz] in world frame
        v_body_sim = np.zeros_like(v_world_sim)

        for i in range(len(self.state_all)):
            quat = -self.state_all.numpy()[i, 3:7]  # 注意负号，与之前处理一致
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # 注意顺序：scipy使用(x,y,z,w)
            v_body_sim[i] = r.apply(v_world_sim[i], inverse=True)
            
        axs2[4,0].plot(self.state_bf_time, v_body_real[:, 0], linewidth=2.5, label="velx_bf_real", color=colors[1])
        axs2[4,0].plot(self.sim_t, v_body_sim[:, 0], linewidth=2.5, label="velx_bf_sim", color=colors[2])
        axs2[4,0].set_title('Velocity in Body Frame (X)', fontsize=28, fontweight='bold', pad=10)
        axs2[4,0].set_xlabel('Time (s)', fontsize=24)
        axs2[4,0].set_ylabel('m/s', fontsize=24)
        axs2[4,0].legend(fontsize=20, prop={'weight':'bold'})
        axs2[4,0].tick_params(axis='both', labelsize=18)
        axs2[4,0].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        axs2[4,1].plot(self.state_bf_time, v_body_real[:, 1], linewidth=2.5, label="vely_bf_real", color=colors[1])
        axs2[4,1].plot(self.sim_t, v_body_sim[:, 1], linewidth=2.5, label="vely_bf_sim", color=colors[2])
        axs2[4,1].set_title('Velocity in Body Frame (Y)', fontsize=28, fontweight='bold', pad=10)
        axs2[4,1].set_xlabel('Time (s)', fontsize=24)
        axs2[4,1].set_ylabel('m/s', fontsize=24)
        axs2[4,1].legend(fontsize=20, prop={'weight':'bold'})
        axs2[4,1].tick_params(axis='both', labelsize=18)
        axs2[4,1].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        axs2[4,2].plot(self.state_bf_time, v_body_real[:, 2], linewidth=2.5, label="velz_bf_real", color=colors[1])
        axs2[4,2].plot(self.sim_t, v_body_sim[:, 2], linewidth=2.5, label="velz_bf_sim", color=colors[2])
        axs2[4,2].set_title('Velocity in Body Frame (Z)', fontsize=28, fontweight='bold', pad=10)
        axs2[4,2].set_xlabel('Time (s)', fontsize=24)
        axs2[4,2].set_ylabel('m/s', fontsize=24)
        axs2[4,2].legend(fontsize=20, prop={'weight':'bold'})
        axs2[4,2].tick_params(axis='both', labelsize=18)
        axs2[4,2].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # ========================== Acceleration ==========================
        axs2[5,0].plot(self.ori_action_time, self.ori_action[:,0], linestyle='--', linewidth=1.8, label="cmd", color='gray')
        axs2[5,0].plot(self.imu_time, self.imu_angle_v[:,3], linewidth=2.5, label="accz_real", color=colors[1])
        axs2[5,0].plot(self.sim_t, (self.accz.numpy()[:]+9.8), linewidth=2.5, label="accz_sim", color=colors[2])
        axs2[5,0].set_title('Acceleration (Z)', fontsize=28, fontweight='bold', pad=10)
        axs2[5,0].set_xlabel('Time (s)', fontsize=24)
        axs2[5,0].set_ylabel('m/s²', fontsize=24)
        axs2[5,0].legend(fontsize=20, prop={'weight':'bold'})
        axs2[5,0].tick_params(axis='both', labelsize=18)
        axs2[5,0].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        
    def plot(self, axs_row):
        self.imu_angle_v = np.stack(self.imu_angle_v)
        self.state_bf = np.stack(self.state_bf)
        row = axs_row
        # ========================== Angular Velocity ==========================
        axs[row,0].plot(self.ori_action_time, self.ori_action[:,1], linestyle='--', linewidth=1.8, label="cmd", color='gray')
        axs[row,0].plot(self.imu_time, self.imu_angle_v[:,0], linewidth=2.5, label="anglevelx_real", color=colors[1])
        axs[row,0].plot(self.sim_t, self.anglevel.numpy()[:,0], linewidth=2.5, label="anglevelx_sim", color=colors[2])
        axs[row,0].set_title('Angular Velocity (X-axis)', fontsize=28, fontweight='bold', pad=10)
        axs[row,0].set_xlabel('Time (s)', fontsize=24)
        axs[row,0].set_ylabel('rad/s', fontsize=24)
        axs[row,0].legend(fontsize=20, prop={'weight':'bold'})
        axs[row,0].tick_params(axis='both', labelsize=18)

        axs[row,1].plot(self.ori_action_time, self.ori_action[:,2], linestyle='--', linewidth=1.8, label="cmd", color='gray')
        axs[row,1].plot(self.imu_time, self.imu_angle_v[:,1], linewidth=2.5, label="anglevely_real", color=colors[1])
        axs[row,1].plot(self.sim_t, self.anglevel.numpy()[:,1], linewidth=2.5, label="anglevely_sim", color=colors[2])
        axs[row,1].set_title('Angular Velocity (Y-axis)', fontsize=28, fontweight='bold', pad=10)
        axs[row,1].set_xlabel('Time (s)', fontsize=24)
        axs[row,1].set_ylabel('rad/s', fontsize=24)
        axs[row,1].legend(fontsize=20, prop={'weight':'bold'})
        axs[row,1].tick_params(axis='both', labelsize=18)

        axs[row,2].plot(self.ori_action_time, self.ori_action[:,3], linestyle='--', linewidth=1.8, label="cmd", color='gray')
        axs[row,2].plot(self.imu_time, self.imu_angle_v[:,2], linewidth=2.5, label="anglevelz_real", color=colors[1])
        axs[row,2].plot(self.sim_t, self.anglevel.numpy()[:,2], linewidth=2.5, label="anglevelz_sim", color=colors[2])
        axs[row,2].set_title('Angular Velocity (Z-axis)', fontsize=28, fontweight='bold', pad=10)
        axs[row,2].set_xlabel('Time (s)', fontsize=24)
        axs[row,2].set_ylabel('rad/s', fontsize=24)
        axs[row,2].legend(fontsize=20, prop={'weight':'bold'})
        axs[row,2].tick_params(axis='both', labelsize=18)

        # ========================== Orientation (Quaternions) ==========================
        # axs[0,3].plot(self.state_bf_time, self.state_bf[:,3], linewidth=2.5, label="oriw_real", color=colors[1])
        # axs[0,3].plot(self.sim_t, -self.state_all.numpy()[:,3], linewidth=2.5, label="oriw_sim", color=colors[2])
        # axs[0,3].set_title('Orientation (W)', fontsize=28, fontweight='bold', pad=10)
        # axs[0,3].set_xlabel('Time (s)', fontsize=24)
        # axs[0,3].set_ylabel('Quaternion', fontsize=24)
        # axs[0,3].legend(fontsize=20, prop={'weight':'bold'})
        # axs[0,3].tick_params(axis='both', labelsize=18)
        # axs[0,3].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # axs[1,0].plot(self.state_bf_time, self.state_bf[:,4], linewidth=2.5, label="orix_real", color=colors[1])
        # axs[1,0].plot(self.sim_t, -self.state_all.numpy()[:,4], linewidth=2.5, label="orix_sim", color=colors[2])
        # axs[1,0].set_title('Orientation (X)', fontsize=28, fontweight='bold', pad=10)
        # axs[1,0].set_xlabel('Time (s)', fontsize=24)
        # axs[1,0].set_ylabel('Quaternion', fontsize=24)
        # axs[1,0].legend(fontsize=20, prop={'weight':'bold'})
        # axs[1,0].tick_params(axis='both', labelsize=18)
        # axs[1,0].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # axs[1,1].plot(self.state_bf_time, self.state_bf[:,5], linewidth=2.5, label="oriy_real", color=colors[1])
        # axs[1,1].plot(self.sim_t, -self.state_all.numpy()[:,5], linewidth=2.5, label="oriy_sim", color=colors[2])
        # axs[1,1].set_title('Orientation (Y)', fontsize=28, fontweight='bold', pad=10)
        # axs[1,1].set_xlabel('Time (s)', fontsize=24)
        # axs[1,1].set_ylabel('Quaternion', fontsize=24)
        # axs[1,1].legend(fontsize=20, prop={'weight':'bold'})
        # axs[1,1].tick_params(axis='both', labelsize=18)
        # axs[1,1].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # axs[1,2].plot(self.state_bf_time, self.state_bf[:,6], linewidth=2.5, label="oriz_real", color=colors[1])
        # axs[1,2].plot(self.sim_t, -self.state_all.numpy()[:,6], linewidth=2.5, label="oriz_sim", color=colors[2])
        # axs[1,2].set_title('Orientation (Z)', fontsize=28, fontweight='bold', pad=10)
        # axs[1,2].set_xlabel('Time (s)', fontsize=24)
        # axs[1,2].set_ylabel('Quaternion', fontsize=24)
        # axs[1,2].legend(fontsize=20, prop={'weight':'bold'})
        # axs[1,2].tick_params(axis='both', labelsize=18)
        # axs[1,2].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # ========================== Linear Velocity ==========================
        # axs[1,3].plot(self.state_bf_time, self.state_bf[:,7], linewidth=2.5, label="velx_real", color=colors[1])
        # axs[1,3].plot(self.sim_t, self.state_all.numpy()[:,7], linewidth=2.5, label="velx_sim", color=colors[2])
        # axs[1,3].set_title('Linear Velocity (X)', fontsize=28, fontweight='bold', pad=10)
        # axs[1,3].set_xlabel('Time (s)', fontsize=24)
        # axs[1,3].set_ylabel('m/s', fontsize=24)
        # axs[1,3].legend(fontsize=20, prop={'weight':'bold'})
        # axs[1,3].tick_params(axis='both', labelsize=18)
        # axs[1,3].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # axs[2,0].plot(self.state_bf_time, self.state_bf[:,8], linewidth=2.5, label="vely_real", color=colors[1])
        # axs[2,0].plot(self.sim_t, self.state_all.numpy()[:,8], linewidth=2.5, label="vely_sim", color=colors[2])
        # axs[2,0].set_title('Linear Velocity (Y)', fontsize=28, fontweight='bold', pad=10)
        # axs[2,0].set_xlabel('Time (s)', fontsize=24)
        # axs[2,0].set_ylabel('m/s', fontsize=24)
        # axs[2,0].legend(fontsize=20, prop={'weight':'bold'})
        # axs[2,0].tick_params(axis='both', labelsize=18)
        # axs[2,0].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # axs[2,1].plot(self.state_bf_time, self.state_bf[:,9], linewidth=2.5, label="velz_real", color=colors[1])
        # axs[2,1].plot(self.sim_t, self.state_all.numpy()[:,9], linewidth=2.5, label="velz_sim", color=colors[2])
        # axs[2,1].set_title('Linear Velocity (Z)', fontsize=28, fontweight='bold', pad=10)
        # axs[2,1].set_xlabel('Time (s)', fontsize=24)
        # axs[2,1].set_ylabel('m/s', fontsize=24)
        # axs[2,1].legend(fontsize=20, prop={'weight':'bold'})
        # axs[2,1].tick_params(axis='both', labelsize=18)
        # axs[2,1].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # ========================== Position ==========================
        # axs[2,2].plot(self.state_bf_time, self.state_bf[:,0], linewidth=2.5, label="posx_real", color=colors[1])
        # axs[2,2].plot(self.sim_t, self.state_all.numpy()[:,0], linewidth=2.5, label="posx_sim", color=colors[2])
        # axs[2,2].set_title('Position (X)', fontsize=28, fontweight='bold', pad=10)
        # axs[2,2].set_xlabel('Time (s)', fontsize=24)
        # axs[2,2].set_ylabel('m', fontsize=24)
        # axs[2,2].legend(fontsize=20, prop={'weight':'bold'})
        # axs[2,2].tick_params(axis='both', labelsize=18)
        # axs[2,2].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # axs[2,3].plot(self.state_bf_time, self.state_bf[:,1], linewidth=2.5, label="posy_real", color=colors[1])
        # axs[2,3].plot(self.sim_t, self.state_all.numpy()[:,1], linewidth=2.5, label="posy_sim", color=colors[2])
        # axs[2,3].set_title('Position (Y)', fontsize=28, fontweight='bold', pad=10)
        # axs[2,3].set_xlabel('Time (s)', fontsize=24)
        # axs[2,3].set_ylabel('m', fontsize=24)
        # axs[2,3].legend(fontsize=20, prop={'weight':'bold'})
        # axs[2,3].tick_params(axis='both', labelsize=18)
        # axs[2,3].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # axs[3,0].plot(self.state_bf_time, self.state_bf[:,2], linewidth=2.5, label="posz_real", color=colors[1])
        # axs[3,0].plot(self.sim_t, self.state_all.numpy()[:,2], linewidth=2.5, label="posz_sim", color=colors[2])
        # axs[3,0].set_title('Position (Z)', fontsize=28, fontweight='bold', pad=10)
        # axs[3,0].set_xlabel('Time (s)', fontsize=24)
        # axs[3,0].set_ylabel('m', fontsize=24)
        # axs[3,0].legend(fontsize=20, prop={'weight':'bold'})
        # axs[3,0].tick_params(axis='both', labelsize=18)
        # axs[3,0].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # ========================== Acceleration ==========================
        axs[row,3].plot(self.ori_action_time, self.ori_action[:,0], linestyle='--', linewidth=1.8, label="cmd", color='gray')
        axs[row,3].plot(self.imu_time, self.imu_angle_v[:,3], linewidth=2.5, label="accz_real", color=colors[1])
        axs[row,3].plot(self.sim_t, (self.accz.numpy()[:]+9.8), linewidth=2.5, label="accz_sim", color=colors[2])
        axs[row,3].set_title('Acceleration (Z)', fontsize=28, fontweight='bold', pad=10)
        axs[row,3].set_xlabel('Time (s)', fontsize=24)
        axs[row,3].set_ylabel('m/s²', fontsize=24)
        axs[row,3].legend(fontsize=20, prop={'weight':'bold'})
        axs[row,3].tick_params(axis='both', labelsize=18)
        axs[row,3].grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
            
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(60, 60))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
fig.suptitle('System State Comparison: Real vs Simulation', fontsize=24, y=0.98)

x_step = bag_plot(path='/home/suncc/suncc/My-research/Vision-based-Racing/x_713_3.bag')
x_step.plot(axs_row=0)

y_step = bag_plot(path='/home/suncc/suncc/My-research/Vision-based-Racing/y_713_1.bag')
y_step.plot(axs_row=1)

z_step = bag_plot(path='/home/suncc/suncc/My-research/Vision-based-Racing/z_713_1.bag')
z_step.plot(axs_row=2)

thrust_step = bag_plot(path='/home/suncc/suncc/My-research/Vision-based-Racing/accz_2g_85.bag')
thrust_step.plot(axs_row=3)
for i in range(4):
    axs[2, i].set_xlabel('Time (s)', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.96])  

# ================= 单独绘制 free_step 的所有状态量 =================
# free_step1 = bag_plot(path='/home/suncc/suncc/My-research/Vision-based-Racing/demo2_724_2131.bag')
free_step1 = bag_plot(path='/home/suncc/suncc/My-research/Vision-based-Racing/diff-land-cylinder-116.bag')
# free_step2 = bag_plot(path='/home/suncc/suncc/My-research/Vision-based-Racing/demo3_yanzheng_depth_3.bag')   
# free_step3 = bag_plot(path='/home/suncc/suncc/My-research/Vision-based-Racing/demo2_fast_6_4_2.bag')
fig2, axs2 = plt.subplots(nrows=6, ncols=3, figsize=(35, 70))
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig2.suptitle('Free Flight: Real vs Simulation', fontsize=36, y=0.99)
# fig3, axs3 = plt.subplots(nrows=6, ncols=3, figsize=(35, 70))
# plt.subplots_adjust(hspace=0.4, wspace=0.3)
# fig3.suptitle('Free Flight: Real vs Simulation', fontsize=36, y=0.99)
# fig4, axs4 = plt.subplots(nrows=6, ncols=3, figsize=(35, 70))
# plt.subplots_adjust(hspace=0.4, wspace=0.3)
# fig3.suptitle('Free Flight: Real vs Simulation', fontsize=36, y=0.99)

free_step1.plot_full(axs2)
# free_step2.plot_full(axs3)
# free_step3.plot_full(axs4)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()