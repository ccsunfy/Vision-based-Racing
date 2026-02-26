
import sys
sys.path.append('/home/henryhuyu/SciRobotic/drone_rl_opti/')
# from src.functions import state_transition_pytorch, update_rotation_pytorch
from rotation import matrix_to_euler_angles

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from sensor_msgs.msg import Imu

import rosbag
import rospy
import torch
from torch.nn import functional as F

import math
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy import interpolate
import matplotlib.pyplot as plt
import tqdm

# bag_path = "/home/henryhuyu/SciRobotic/rebuttal/vins_data/bags_4_clipped/2023-07-31-13-50-12_clipped.bag"
# bag_path = "/home/henryhuyu/SciRobotic/rebuttal/vins_data/bags_7_clipped/2023-07-31-14-52-08_clipped.bag"
# bag_path = "/home/henryhuyu/SciRobotic/rebuttal/vins_data/bags_13_clipped/2023-07-31-15-07-53_clipped.bag"
# for i in /home/henryhuyu/SciRobotic/rebuttal/vins_data/bags_*_clipped/*.bag; do python /home/henryhuyu/SciRobotic/drone_rl_opti/tools/fit_dynamic.py $i; done


def quaternion_to_matrix(q):
    return np.array([[1-2*(q[2]**2+q[3]**2), 2*(q[1]*q[2]-q[3]*q[0]), 2*(q[1]*q[3]+q[2]*q[0])],
              [2*(q[1]*q[2]+q[3]*q[0]), 1-2*(q[1]**2+q[3]**2), 2*(q[2]*q[3]-q[1]*q[0])],
              [2*(q[1]*q[3]-q[2]*q[0]), 2*(q[2]*q[3]+q[1]*q[0]), 1-2*(q[1]**2+q[2]**2)]], dtype=np.float32)

def get_local_coord_R(R):
    """Prepare local coordinate (R with 0 pitch and roll)."""
    fwd = R[:, :, 0].clone()
    up = torch.zeros_like(fwd)
    fwd[:, 2] = 0
    up[:, 2] = 1
    fwd = F.normalize(fwd, 2., -1)
    R = torch.stack([fwd, torch.cross(up, fwd), up], -1)
    return R

class BagParser():
    def __init__(self, bag_path) -> None:
        self.device = 'cpu'
        self.batch_size = 1
        self.grad_decay = 0.4
        self.bag_path = bag_path
        self.bag = rosbag.Bag(self.bag_path)
        
        self.topic_dict = {
            'imu': '/mavros/imu/data',
            'imu_odom': '/imu_odom',
            'action': '/planner/a_set',
            'rpy_cmd': '/planner/rpy',
            'mocap_fusion': '/mocap_fusion/odom/local',
            'trigger': '/traj_start_trigger',
            'v_pred': '/planner/v_pred'
        }   
        self.topic_list = list(self.topic_dict.items())
        self.topics = ['/mavros/imu/data', '/imu_odom', '/planner/a_set', '/mocap_fusion/odom/local', '/planner/rpy', '/traj_start_trigger']
        
        
        # output area, TODO might need to use imu_odom ? 
        self.trigger_t = -1.0
        self.first_action_t = -1.0
        self.imu_t = []
        self.imu_linear_acc_b = []
        self.imu_linear_acc_w = []
        self.imu_quat = []
        
        
        self.mocap_fusion_t = []
        self.mocap_fusion_p = []
        self.mocap_fusion_v = []
        self.mocap_fusion_quat = []
        self.mocap_fusion_R = []
        self.mocap_fusion_eular = []
        
        self.net_t = []
        self.actions = []
        self.v_pred = []
        self.rpy_cmd = []
        
        self.imu_odom_t = []
        self.imu_odom_quat = []
        self.imu_odom_R = []
        self.imu_odom_eular = []
        self.imu_odom_p = []
        self.imu_odom_v = []
        self.imu_odom_a = []
        
        self.init_R, self.init_p, self.init_v, self.init_a = None, None, None, None
        
        self.bag_parser()
        
             
        
    def bag_parser(self):
        
        for topic, msg, t in self.bag.read_messages(self.topics):
            if topic == self.topic_dict['trigger'] and msg.data and self.trigger_t < 0:
                self.trigger_t = t.to_sec()
                
            if topic == self.topic_dict['trigger'] and not msg.data and self.trigger_t > 0:
                self.trigger_t = -1.0
            
            if self.trigger_t < 0:
                continue
                
            if topic == self.topic_dict['imu']:
                msg:Imu
                linear_acc_body = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z], dtype=np.float32)
                imu_q = [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
                self.imu_t.append(t.to_sec() - self.trigger_t)
                self.imu_linear_acc_b.append(linear_acc_body)
                self.imu_linear_acc_w.append(self.imu_odom_R[-1] @ linear_acc_body)
                self.imu_quat.append(imu_q)
                
            if topic == self.topic_dict['imu_odom']:
                msg:Odometry
                self.imu_odom_t.append(t.to_sec() - self.trigger_t)
                imu_odom_q = np.array([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z], dtype=np.float32)
                imu_odom_R_ = quaternion_to_matrix(imu_odom_q)
                imu_odom_p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.float32)
                imu_odom_v = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z], dtype=np.float32)
                self.imu_odom_quat.append(imu_odom_q)
                self.imu_odom_R.append(imu_odom_R_)
                self.imu_odom_p.append(np.array(imu_odom_p))
                self.imu_odom_v.append(np.array(imu_odom_v))
                yaw, pitch, roll = R_scipy.from_matrix(imu_odom_R_).as_euler('ZYX', degrees=False)
                self.imu_odom_eular.append([yaw, pitch, roll])
                
            if topic == self.topic_dict['rpy_cmd']:
                pitch_cmd, roll_cmd, yaw_cmd = msg.pitch, msg.roll, msg.yaw
                self.rpy_cmd.append(np.array([pitch_cmd, roll_cmd, yaw_cmd], dtype=np.float32))
                
                
            if topic == self.topic_dict['action']:
                msg:Point
                if self.first_action_t < 0:
                    self.first_action_t = t.to_sec() - self.trigger_t
                    self.init_R = self.imu_odom_R[-1]
                    self.init_p = self.imu_odom_p[-1]
                    self.init_v = self.imu_odom_v[-1]
                    print('reading init a body as ', self.imu_linear_acc_b[-1])
                    print('reading init imu odom R as ', self.init_R)
                    self.init_a = np.array((self.imu_odom_R[-1] @ self.imu_linear_acc_b[-1]), dtype=np.float32)
                    self.init_a = self.init_a - np.array([0.0, 0.0, 9.80665], dtype=np.float32)
                    print('init a as ', self.init_a)
                    
                    
                self.net_t.append(t.to_sec() - self.trigger_t)
                self.actions.append(np.array([msg.x, msg.y, msg.z], dtype=np.float32))
                
            if topic == self.topic_dict['v_pred']:
                msg:Point
                self.v_pred.append(np.array([msg.x, msg.y, msg.z], dtype=np.float32))
                
            if topic == self.topic_dict['mocap_fusion']:
                msg:Odometry
                self.mocap_fusion_t.append(t.to_sec() - self.trigger_t)
                
                p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.float32)
                v = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z], dtype=np.float32)
                q = np.array([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z], dtype=np.float32)
                
                self.mocap_fusion_p.append(p)
                self.mocap_fusion_v.append(v)
                self.mocap_fusion_quat.append(q)
                self.mocap_fusion_R.append(quaternion_to_matrix(q))
        
        # all data in list to float32
        
        

 
class DynamicFitter():
    def __init__(self) -> None:
        self.device = 'cpu'
        self.batch_size = 1
        self.grad_decay = 0.4
        
        # fixed
        self.v_wind = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat((self.batch_size, 1))
        print('v_wind shape', self.v_wind.shape)
        self.dg = torch.zeros((self.batch_size, 3), device=self.device)
           
        self.simulated_p, self.simulated_v, self.simulated_a, self.simulated_R = [], [], [], []
        self.simulated_eular = []
        self.simulated_t = []
        
        # 可训练参数定义 (带batch维度)
        self.pitch_ctl_delay = torch.nn.Parameter(torch.ones(self.batch_size, 1) * 18.0)  # [B,1]
        self.roll_ctl_delay = torch.nn.Parameter(torch.ones(self.batch_size, 1) * 18.0)  # [B,1]
        self.yaw_ctl_delay = torch.nn.Parameter(torch.ones(self.batch_size, 1) * 6.0)  # [B,1]
        self.drag = torch.nn.Parameter(torch.ones(self.batch_size, 2) * torch.tensor([0.012, 0.2])) # [B, 2]

        self.z_drag_coef = torch.nn.Parameter(torch.ones(self.batch_size, 1))         
        
        self.reset() 
        
    def reset(self):
        device = self.device
        B = self.batch_size

        self.target_dir = torch.tensor([1.0, 0.0, 0.0], device=device).repeat((B, 1))
        
        self.p = torch.tensor([0.0, 0.0, 1.0], device=device).repeat((B, 1))
        self.v = torch.zeros((B, 3), device=device)
        
        self.act = torch.zeros_like(self.v)
        self.a = self.act
        

        R = torch.zeros((B, 3, 3), device=device)
        self.R, self.roll, self.pitch, self.yaw = self.update_rotation_pytorch(R, self.act, torch.randn((B, 3), device=device) * 0.2 + self.target_dir, torch.zeros_like(self.yaw_ctl_delay), 5)
        
        
    def state_transition_pytorch(self, R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, last_act, p, v, v_wind, a, grad_decay, ctl_dt):

        alpha = torch.exp(-pitch_ctl_delay * ctl_dt)
        act_next = act_pred * (1 - alpha) + last_act * alpha

        v_fwd_s, v_left_s, v_up_s = (v.add(-v_wind)[:, None] @ R).unbind(-1)
        
        # quadratic drag 
        drag = drag_2[:, :1] * (v_fwd_s.abs() * v_fwd_s * R[..., 0] + v_left_s.abs() * v_left_s * R[..., 1] + v_up_s.abs() * v_up_s * R[..., 2] * z_drag_coef)
        # linear drag 
        drag += drag_2[:, 1:] * (v_fwd_s * R[..., 0] + v_left_s * R[..., 1] + v_up_s * R[..., 2] * z_drag_coef)

        # Integrate
        a_next = act_next + dg - drag # - 0.5 * torch.tensor([0.0, 0.0, 1])
        p_next = p + v * ctl_dt + 0.5 * a * ctl_dt**2
        v_next = v + (a + a_next) / 2 * ctl_dt

        return act_next, p_next, v_next, a_next   
    
    @torch.no_grad()
    def update_rotation_pytorch(self, R, a_thr, v_pred, alpha, yaw_inertia=5):
        self_forward_vec = R[..., 0]
        g_std = torch.tensor([0, 0, -9.80665], device=R.device)
        a_thr = a_thr - g_std
        thrust = torch.norm(a_thr, 2, -1, True)
        self_up_vec = a_thr / thrust
        forward_vec = self_forward_vec * yaw_inertia + v_pred
        forward_vec = self_forward_vec * alpha + F.normalize(forward_vec, 2, -1) * (1 - alpha)

        forward_vec[:, 2] = (forward_vec[:, 0] * self_up_vec[:, 0] + forward_vec[:, 1] * self_up_vec[:, 1]) / -self_up_vec[:, 2]
        self_forward_vec = F.normalize(forward_vec, 2, -1)
        self_left_vec = torch.cross(self_up_vec, self_forward_vec)

        roll = torch.arctan2(self_left_vec[:, 2], self_up_vec[:, 2])
        pitch = torch.arcsin(-self_forward_vec[:, 2])
        yaw = torch.arctan2(self_forward_vec[:, 1], self_forward_vec[:, 0])

        return torch.stack([
            self_forward_vec,
            self_left_vec,
            self_up_vec,
        ], -1), roll, pitch, yaw
        
    def set_state(self, R, p, v, a=None):
        self.R = R
        self.p = p
        self.v = v
        if a is not None:
            self.a = a
            self.act = a
        

        self.yaw, self.pitch, self.roll = matrix_to_euler_angles(self.R, 'ZYX').unbind(-1)
        

        
       
        
        

    def run(self, act_pred, ctl_dt=1/15, v_pred=None):
        self.act, self.p, self.v, self.a = self.state_transition_pytorch(
            self.R, self.dg, self.z_drag_coef, self.drag, self.pitch_ctl_delay,
            act_pred, self.act, self.p, self.v, self.v_wind, self.a,
            self.grad_decay, ctl_dt)
        
        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        
        self.R, self.roll, self.pitch, self.yaw = self.update_rotation_pytorch(self.R, self.act, v_pred, alpha, 5)
    
    # Most basic simulation
    def base_sim(self, bag_data:BagParser, lag=0, plot=False):
        init_R, init_p, init_v, init_a = bag_data.init_R, bag_data.init_p, bag_data.init_v, bag_data.init_a
        init_R, init_p, init_v, init_a = map(lambda x: torch.tensor(x, device=self.device).unsqueeze(0), [init_R, init_p, init_v, init_a])
        self.set_state(init_R, init_p, init_v, init_a)
        
        # run simulation 
        act_buf = [self.act] * (1 + lag)
        last_act_t = bag_data.first_action_t - 1/15
        
        for i in range(len(bag_data.actions)):
            act_t = bag_data.net_t[i]
            act = torch.tensor(bag_data.actions[i], device=self.device).unsqueeze(0)
            
            self.run(torch.tensor(act_buf[i], device=self.device), act_t - last_act_t, torch.tensor([1.0, 0.0, 0.0]))
            if i == 0:
                print('act t0', act_t - last_act_t)
            
            self.simulated_p.append(self.p[0])
            self.simulated_v.append(self.v[0])
            self.simulated_a.append(self.a[0])
            self.simulated_R.append(self.R[0])
            self.simulated_eular.append([self.yaw[0], self.pitch[0], self.roll[0]])
            self.simulated_t.append(act_t)
            
            last_act_t = act_t
            act_buf.append(act)

        pitch_delay = self.sim_gt_lag_fitter(bag_data)
        lag_cmd = self.cmd_lag_fitter(bag_data)
        
        return pitch_delay, lag_cmd
            
        
       
        
        
    
    def param_fitter(self, bag_data:BagParser, num_epochs=2000, lr=0.01):
        # use torch and autograd to fit the parameters

        gt_p = bag_data.mocap_fusion_p
        gt_v = bag_data.mocap_fusion_v
        
        gt_px_interp = self.interp_data_to_t([item[0] for item in gt_p], bag_data.mocap_fusion_t, self.simulated_t)
        gt_py_interp = self.interp_data_to_t([item[1] for item in gt_p], bag_data.mocap_fusion_t, self.simulated_t)
        gt_pz_interp = self.interp_data_to_t([item[2] for item in gt_p], bag_data.mocap_fusion_t, self.simulated_t)

        gt_vx_interp = self.interp_data_to_t([item[0] for item in gt_v], bag_data.mocap_fusion_t, self.simulated_t)
        gt_vy_interp = self.interp_data_to_t([item[1] for item in gt_v], bag_data.mocap_fusion_t, self.simulated_t)
        gt_vz_interp = self.interp_data_to_t([item[2] for item in gt_v], bag_data.mocap_fusion_t, self.simulated_t)
        
        gt_p_interp = torch.tensor(np.array([gt_px_interp, gt_py_interp, gt_pz_interp]).T, device=self.device)
        gt_v_interp = torch.tensor(np.array([gt_vx_interp, gt_vy_interp, gt_vz_interp]).T, device=self.device)
        
        print('gt_p_interp shape is ', gt_p_interp.shape)
        optimizer = torch.optim.Adam([
            {'params': [self.pitch_ctl_delay, self.roll_ctl_delay, self.yaw_ctl_delay, self.drag, self.z_drag_coef], 
             'lr': lr}
        ])
        for epoch in tqdm.tqdm(range(num_epochs)):
            optimizer.zero_grad()
            self.reset()
            init_R, init_p, init_v, init_a = bag_data.init_R, bag_data.init_p, bag_data.init_v, bag_data.init_a
            init_R, init_p, init_v, init_a = map(lambda x: torch.tensor(x, device=self.device).unsqueeze(0), [init_R, init_p, init_v, init_a])
            self.set_state(init_R, init_p, init_v, init_a)
            
            sim_p, sim_v, sim_t = [], [], []
            last_act_t = bag_data.first_action_t
        
            # 添加初始状态
            sim_p.append(self.p.clone())
            sim_v.append(self.v.clone())
            sim_t.append(last_act_t)
            
            for i in range(len(bag_data.actions)):
                act_t = bag_data.net_t[i]
                act = torch.tensor(bag_data.actions[i], device=self.device).unsqueeze(0)
                
                if i > 0:
                    ctl_dt = act_t - last_act_t
                    self.run(act, ctl_dt, torch.tensor([1.0, 0.0, 0.0]))  # 使用当前速度作为预测
                    
                    sim_p.append(self.p.clone())
                    sim_v.append(self.v.clone())
                    sim_t.append(act_t)
                
                last_act_t = act_t
                
            # 转换模拟数据为张量 [T_sim, 3]
            sim_p_tensor = torch.cat(sim_p, dim=0)  # shape: [T_sim, 3]
            sim_v_tensor = torch.cat(sim_v, dim=0)
            sim_t_tensor = torch.tensor(sim_t, device=self.device)  # shape: [T_sim]
            
          
            # 计算损失函数
            position_loss = F.mse_loss(sim_p_tensor, gt_p_interp)
            velocity_loss = F.mse_loss(sim_v_tensor, gt_v_interp)
            total_loss = position_loss + velocity_loss
            
            total_loss.backward()
            optimizer.step()
            
            # 打印训练信息
            if epoch % 10 == 0:
                print(f'Epoch {epoch:03d} | Loss: {total_loss.item():.4f} '
                    f'(Pos: {position_loss.item():.4f}, Vel: {velocity_loss.item():.4f})')
                print(f'Current params - pitch_delay: {self.pitch_ctl_delay.item():.2f} '
                    f'roll_delay: {self.roll_ctl_delay.item():.2f} '
                    f'drag: {self.drag.detach().cpu().numpy()} '
                    f'z_drag: {self.z_drag_coef.item():.3f}')
                
        # plot 
        fig_p, ax = plt.subplots(1, 3)
        ax[0].plot(sim_t_tensor.detach().cpu().numpy(), sim_p_tensor[:, 0].detach().cpu().numpy(), label='sim_x')
        ax[0].plot(sim_t_tensor.detach().cpu().numpy(), gt_p_interp[:, 0].detach().cpu().numpy(), label='gt_x')
        ax[0].legend()
        
        ax[1].plot(sim_t_tensor.detach().cpu().numpy(), sim_p_tensor[:, 1].detach().cpu().numpy(), label='sim_y')
        ax[1].plot(sim_t_tensor.detach().cpu().numpy(), gt_p_interp[:, 1].detach().cpu().numpy(), label='gt_y')
        ax[1].legend()
        
        ax[2].plot(sim_t_tensor.detach().cpu().numpy(), sim_p_tensor[:, 2].detach().cpu().numpy(), label='sim_z')
        ax[2].plot(sim_t_tensor.detach().cpu().numpy(), gt_p_interp[:, 2].detach().cpu().numpy(), label='gt_z')
        ax[2].legend()
        
        fig_v, ax_v = plt.subplots(1, 3)
        ax_v[0].plot(sim_t_tensor.detach().cpu().numpy(), sim_v_tensor[:, 0].detach().cpu().numpy(), label='sim_vx')
        ax_v[0].plot(sim_t_tensor.detach().cpu().numpy(), gt_v_interp[:, 0].detach().cpu().numpy(), label='gt_vx')
        ax_v[0].legend()
        
        ax_v[1].plot(sim_t_tensor.detach().cpu().numpy(), sim_v_tensor[:, 1].detach().cpu().numpy(), label='sim_vy')
        ax_v[1].plot(sim_t_tensor.detach().cpu().numpy(), gt_v_interp[:, 1].detach().cpu().numpy(), label='gt_vy')
        ax_v[1].legend()
        
        ax_v[2].plot(sim_t_tensor.detach().cpu().numpy(), sim_v_tensor[:, 2].detach().cpu().numpy(), label='sim_vz')
        ax_v[2].plot(sim_t_tensor.detach().cpu().numpy(), gt_v_interp[:, 2].detach().cpu().numpy(), label='gt_vz')
        ax_v[2].legend()
        
        plt.show()
        
            
                
            
        
     
        
        
        
    def sim_gt_lag_fitter(self, bag_data:BagParser, plot=False):
        # using pitch and roll to fit the delay
        gt_pitchs = np.array([item[1] for item in bag_data.imu_odom_eular])
        gt_rolls = np.array([item[2] for item in bag_data.imu_odom_eular])
        
        sim_pitchs = np.array([item[1] for item in self.simulated_eular])
        sim_rolls = np.array([item[2] for item in self.simulated_eular])
        
        pitch_delay = self.lag_fitter(gt_pitchs, bag_data.imu_odom_t, sim_pitchs, self.simulated_t, plot=plot)
        roll_delay = self.lag_fitter(gt_rolls, bag_data.imu_odom_t, sim_rolls, self.simulated_t, plot=plot)
        print('pitch_delay is ', pitch_delay)
        print('roll_delay is ', roll_delay)
        return pitch_delay
        
        
    def cmd_lag_fitter(self, bag_data:BagParser, plot=False):
        gt_pitchs = np.array([item[1] for item in bag_data.imu_odom_eular])
        cmd_pitchs = np.array([item[0] for item in bag_data.rpy_cmd])
        lag_cmd = self.lag_fitter(gt_pitchs, bag_data.imu_odom_t, cmd_pitchs, bag_data.net_t, plot=plot)
        print(f'lag_cmd is {lag_cmd}, time is {lag_cmd * 1/30}')
        return lag_cmd
        
    def lag_fitter(self, src1, src1_t, src2, src2_t, plot=False):
        sub_freq = 30
        src1 = np.array(src1)
        src2 = np.array(src2)
        src1_t = np.array(src1_t)
        src2_t = np.array(src2_t)
        
        t_min = max(src1_t.min(), src2_t.min())
        t_max = min(src1_t.max(), src2_t.max())
        
        t_common = np.linspace(t_min, t_max, int((t_max - t_min) * sub_freq) + 1)
        print('lag fitter, interpolated t_common shape is ', t_common.shape)
        src1_interp = np.interp(t_common, src1_t, src1)
        src2_interp = np.interp(t_common, src2_t, src2)
        
        scr1_unbiased = src1_interp - src1_interp.mean()
        src2_unbiased = src2_interp - src2_interp.mean()
        
        corr = np.correlate(scr1_unbiased, src2_unbiased, mode='full')
        max_corr_index = np.argmax(corr)
        delay = max_corr_index - (len(src1_interp) - 1)
        
        
        if plot:
            fig = plt.figure()
            plt.plot(t_common, src1_interp, label='src1_interp')
            plt.plot(t_common, src2_interp, label='src2_interp')
            plt.plot(t_common + delay * 1/sub_freq, src2_interp, label='src2_interp_delayed')
            # plt.plot(corr, label='corr')
            plt.legend()
            plt.show()

        return delay
        
    def interp_data_to_t(self, arr, t_arr, target_t_arr, plot=False):
        arr = np.array(arr, dtype=np.float32)
        t_arr = np.array(t_arr, dtype=np.float32)
        target_t_arr = np.array(target_t_arr, dtype=np.float32)
        
        f_interp = np.interp(target_t_arr, t_arr, arr).astype(np.float32)
        
        if plot:
            fig = plt.figure()
            plt.plot(t_arr, arr, label='original')
            plt.plot(target_t_arr, f_interp, label='interpolated')
            plt.legend()
            plt.show()
            
        return f_interp
        
    def corr_lag(self, x, y):
        x = x - x.mean()
        y = y - y.mean()
        
        corr = np.correlate(x, y, mode='full')
        max_corr_index = np.argmax(corr)
        delay = max_corr_index - (len(x) - 1)
        print('delay is ', delay)
        return delay
        
    def plot(self, bag_data:BagParser, lag=0):
        sim_t = [t + lag * 1/30 for t in self.simulated_t]
        
        # for p
        fig_p, ax = plt.subplots(1, 3)
        ax[0].plot(sim_t, [item.detach()[0] for item in self.simulated_p], label='sim_x')
        ax[0].plot(bag_data.mocap_fusion_t, [item[0] for item in bag_data.mocap_fusion_p], label='gt_x')
        ax[0].legend()
        
        ax[1].plot(sim_t, [item.detach()[1] for item in self.simulated_p], label='sim_y')
        ax[1].plot(bag_data.mocap_fusion_t, [item[1] for item in bag_data.mocap_fusion_p], label='gt_y')
        ax[1].legend()
        
        ax[2].plot(sim_t, [item.detach()[2] for item in self.simulated_p], label='sim_z')
        ax[2].plot(bag_data.mocap_fusion_t, [item[2] for item in bag_data.mocap_fusion_p], label='gt_z')
        ax[2].legend()
        
        # for v
        fig_v, ax_v = plt.subplots(1, 3)
        ax_v[0].plot(sim_t, [item.detach()[0] for item in self.simulated_v], label='sim_x')
        ax_v[0].plot(bag_data.mocap_fusion_t, [item[0] for item in bag_data.mocap_fusion_v], label='gt_x')
        ax_v[0].legend()
        
        ax_v[1].plot(sim_t, [item.detach()[1] for item in self.simulated_v], label='sim_y')
        ax_v[1].plot(bag_data.mocap_fusion_t, [item[1] for item in bag_data.mocap_fusion_v], label='gt_y')
        ax_v[1].legend()
        
        ax_v[2].plot(sim_t, [item.detach()[2] for item in self.simulated_v], label='sim_z')
        ax_v[2].plot(bag_data.mocap_fusion_t, [item[2] for item in bag_data.mocap_fusion_v], label='gt_z')
        ax_v[2].legend()
        
        # for euler
        yaws = np.rad2deg([item[0] for item in self.simulated_eular])
        pitches = np.rad2deg([item[1] for item in self.simulated_eular])
        rolls = np.rad2deg([item[2] for item in self.simulated_eular])
        
        gt_yaws = np.rad2deg([item[0] for item in bag_data.imu_odom_eular])
        gt_pitches = np.rad2deg([item[1] for item in bag_data.imu_odom_eular])
        gt_rolls = np.rad2deg([item[2] for item in bag_data.imu_odom_eular])
        
        
        
        fig_e, ax_e = plt.subplots(1, 3)
        ax_e[0].plot(sim_t, yaws, label='sim_yaw')
        ax_e[0].plot(bag_data.imu_odom_t, gt_yaws, label='gt_yaw')
        ax_e[0].legend()
        
        ax_e[1].plot(sim_t, pitches, label='sim_pitch')
        ax_e[1].plot(bag_data.imu_odom_t, gt_pitches, label='gt_pitch')
        ax_e[1].legend()
        
        ax_e[2].plot(sim_t, rolls, label='sim_roll')
        ax_e[2].plot(bag_data.imu_odom_t, gt_rolls, label='gt_roll')
        ax_e[2].legend()
        

        fig_p.savefig('/home/henryhuyu/SciRobotic/drone_rl_opti/tools/fit_dynamics/dyna_p.png')
        fig_v.savefig('/home/henryhuyu/SciRobotic/drone_rl_opti/tools/fit_dynamics/dyna_v.png')
        fig_e.savefig('/home/henryhuyu/SciRobotic/drone_rl_opti/tools/fit_dynamics/dyna_eular.png')
        
        
        # plt.show()
        
        

bag_path = "/home/henryhuyu/2025-03-09-13-01-50_35dyna.bag"
bag_data = BagParser(bag_path)
fitter = DynamicFitter()

# fit angle lag using base sim
pitchlag, lag_cmd = fitter.base_sim(bag_data, lag=0)
# fitter.plot(bag_data)
fitter.plot(bag_data, pitchlag)

# fitter.param_fitter(bag_data)
# fitter.cmd_lag_fitter(bag_data)
# fitter.plot(bag_data)
# fitter.interpolate_to_t([item[0] for item in bag_data.imu_odom_eular], bag_data.imu_odom_t, bag_data.mocap_fusion_t, plot=True)
# fitter.lag_fitter()