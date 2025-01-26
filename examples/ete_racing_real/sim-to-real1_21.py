#!/usr/bin/env python

import rospy
import time
import torch as th
import numpy as np
import cv2
import time
import math, json
# import onnxruntime as ort

from tqdm import tqdm
from sensor_msgs.msg import Image,Imu
# from std_msgs.msg import Float32 
from std_msgs.msg import Bool, Float32MultiArray
from collections import deque
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion, Twist
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from mavros_msgs.msg import RCIn
from quadrotor_msgs.msg import Command,TRPYCommand
from cv_bridge import CvBridge, CvBridgeError
from utils.algorithms.ppo import ppo
from scipy.spatial.transform import Rotation as R
from utils.type import bound

MODE_CHANNEL = 6 
HOVER_ACC = 9.77
HOVER_THRUST = 0.3
MODE_SHIFT_VALUE = 0.25

th.set_grad_enabled(False)

class RealEnv:
    def __init__(self, log_prefix=""):
        self.log_prefix = log_prefix
        rospy.loginfo(f"{self.log_prefix} preheating...")
        
        start_model = time.time()
        self.model_path = 'src/sim_to_real/ppo_436.zip'

        # self.traced_model = th.jit.trace(self.model_path, self.example_input)
        # print(f"Traced model: {self.traced_model}")

        self.model = ppo.load(self.model_path)
        self.model.policy.eval()
        end_model = time.time()
        print(f"模型加载耗时: {end_model - start_model:.4f} 秒")

        rospy.loginfo(f"{self.log_prefix} Obs: {self.model.policy.observation_space.shape}")
        rospy.loginfo(f"{self.log_prefix} Action: {self.model.policy.action_space.shape}")
        
        self.bridge = CvBridge()
        self.depth_image = None

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        self.position = None
        self.orientation = None
        self.velocity = None
        self.angular_velocity = None
        self.yaw = 0
        self.num_envs = 1
        self.max_sense_radius = 10.0
        self.success_radius = 0.3
        self.number = 0

        self.traj_list = []
        
        self.targets = th.as_tensor([ # waypoint简单飞行测试
            [3, 0, 1],    
            [4, 1, 1],    
            [5, 0, 1],   
            [4, -1, 1],
            [3, 0, 1]   
        ])
        
        self.length_target = len(self.targets)
        self._next_target_num = 2
        self._next_target_i = th.zeros((self.num_envs,), dtype=th.int)
        self._past_targets_num = th.zeros((self.num_envs,), dtype=th.int)
        self._is_pass_next = th.zeros((self.num_envs,), dtype=th.bool)
        
        self.latent = th.zeros(256, dtype=th.float32)

        self.v_d = 2.0 * th.ones((self.num_envs,),dtype=th.float32)
        self.m = 0.75  # mass of the drone
        
        # sub
        start_depth_sub = time.time()
        self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
        end_depth_sub = time.time()
        print(f"深度图像订阅耗时: {end_depth_sub - start_depth_sub:.4f} 秒")

        rospy.Subscriber("/bfctrl/local_odom", Odometry, self.odom_callback)
        rospy.Subscriber('/mavros/rc/in', RCIn, self.rc_callback, queue_size=10)
        rospy.Subscriber('/mavros/imu/data', Imu, self.angular_callback, queue_size=10)
        
        # pub
        self.ctbr_pub = rospy.Publisher('/bfctrl/cmd', Command, queue_size=10, tcp_nodelay=True)
        # self.state_pub = rospy.Publisher('/state', Float32MultiArray, queue_size=10)
        # self.action_pub = rospy.Publisher('/action', Float32MultiArray, queue_size=10)
        # self.targets_pub = rospy.Publisher('/rl/targets', PoseArray, queue_size=10)
        # self.success_poses_pub = rospy.Publisher('/rl/success_poses', PoseArray, queue_size=10)
        # self.traj_pub = rospy.Publisher('/rl/traj', Marker, queue_size=10)
        self.load("src/sim_to_real/example.json")
        
    def depth_callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            self.depth_image = cv2.resize(cv_image, (64, 64))
        except CvBridgeError as e:
            print(e)
    
    def rc_callback(self, data: RCIn):
        if (data.channels[MODE_CHANNEL] - 1000.0) / 1000 < MODE_SHIFT_VALUE:
            if not self.mode_enable:
                rospy.logwarn(f"{self.log_prefix}please enable the mode.")
            self.mode_enable = True
        else:
            self.mode_enable = False

    def angular_callback(self, data):
        angular_velocity = data.angular_velocity
        self.angular_velocity = th.tensor([angular_velocity.x, angular_velocity.y, angular_velocity.z], dtype=th.float32).to(self.device)
    
    def odom_callback(self, data):
        orientation = data.pose.pose.orientation
        position = data.pose.pose.position
        velocity = data.twist.twist.linear
        # angular_velocity = data.twist.twist.angular
        
        # orientation = np.array([orientation.x, orientation.y, orientation.z, orientation.w], dtype=np.float32)
        # print("q: ", q)
        # rpy = R.from_quat(orientation).as_euler("xyz")
        # position = th.as_tensor([position.x, position.y, position.z], dtype=th.float32)
        # self.rpy = np.array(rpy, dtype=np.float32)
        # # rospy.loginfo(f"{self.log_prefix} cur rpy: {self.rpy}")
        # # print(f"{self.log_prefix} cur rpy: {self.rpy}")
        # velocity = np.array([velocity.x, velocity.y, velocity.z], dtype=np.float32)
        # angular_velocity = np.array([angular_velocity.x, angular_velocity.y, angular_velocity.z], dtype=np.float32)

        self.orientation = th.tensor([orientation.x, orientation.y, orientation.z, orientation.w], dtype=th.float32).to(self.device)
        self.position = th.tensor([position.x, position.y, position.z], dtype=th.float32).to(self.device)
        self.velocity = th.tensor([velocity.x, velocity.y, velocity.z], dtype=th.float32).to(self.device)
        # self.angular_velocity = th.tensor([angular_velocity.x, angular_velocity.y, angular_velocity.z], dtype=th.float32).to(self.device)
        
        self.traj_list.append(self.position)
    
    def traj_publish(self):
        line_strip = Marker()
        line_strip.header.frame_id = "world"
        line_strip.header.stamp = rospy.Time.now()
        line_strip.ns = "trajectory"
        line_strip.id = 0
        line_strip.type = Marker.LINE_STRIP
        line_strip.action = Marker.ADD

        line_strip.color.r = 0.0
        line_strip.color.g = 1.0
        line_strip.color.b = 0.0
        line_strip.color.a = 0.5
        line_strip.scale.x = 0.01

        for pos in self.traj_list:
            point = Point()
            point.x = pos[0]
            point.y = pos[1]
            point.z = pos[2]
            line_strip.points.append(point)

        self.traj_pub.publish(line_strip)

    def targets_publish(self):
        pose_array_msg = PoseArray()
        pose_array_msg.header.frame_id = "world"
        pose_array_msg.header.stamp = rospy.Time.now()
        
        for target in self.target_array:
            pose = Pose()
            pose.position.x = target[0]
            pose.position.y = target[1]
            pose.position.z = target[2]

            target_quat = R.from_euler("xyz", [0, 0, target[3]]).as_quat()
            pose.orientation.x = target_quat[0]
            pose.orientation.y = target_quat[1]
            pose.orientation.z = target_quat[2]
            pose.orientation.w = target_quat[3]

            pose_array_msg.poses.append(pose)
        self.targets_pub.publish(pose_array_msg)
        
    def load(self, path=""):
        with open(path, "r") as f:
            data = json.load(f)
        self._bd_rate = bound(
            max=th.tensor(data["max_rate"]), min=th.tensor(-data["max_rate"])
        )
        
    def de_normalize(self, command, normal_range: [float, float] = (-1, 1)):
        thrust_scale = (self.m * HOVER_ACC) * 2 / self.m
        thrust_bias = (self.m * HOVER_ACC) / self.m
        bodyrate_scale = (self._bd_rate.max - self._bd_rate.min) / (
                    normal_range[1] - normal_range[0]
            )
        bodyrate_bias = self._bd_rate.max - bodyrate_scale * normal_range[1]
        command = th.hstack([
                (command[:, :1] * thrust_scale + thrust_bias) * self.m,
                command[:, 1:] * bodyrate_scale+ bodyrate_bias])
        return command.T

    def world_to_body(self, relative_pos_world):
        # 使用四元数将世界坐标系下的相对坐标转换到机体系下
        rotation = R.from_quat(np.array(self.orientation.cpu().tolist()))
        rotation_matrix = th.as_tensor(rotation.as_matrix()).float()  
        relative_pos_world = relative_pos_world.float()  

        # 进行矩阵乘法，将世界坐标系下的相对坐标转换到机体系下
        relative_pos_body = th.einsum('bij,bjk->bik', rotation_matrix.unsqueeze(0).transpose(1, 2), relative_pos_world.unsqueeze(0).transpose(1, 2)).transpose(1, 2)
        return relative_pos_body

    def get_success(self) -> th.Tensor:
        _next_target_i_clamp = self._next_target_i
        # print("pos:",self.position, "  \ntarget:",self.targets, "\nnext_trget_i", _next_target_i_clamp)
        self._is_pass_next = ((self.position - self.targets[_next_target_i_clamp]).norm(dim=1) <= self.success_radius)
        if self._is_pass_next:
            rospy.logerr(f"{self.log_prefix} reach target number {self.number+1} !!!!!")
            self.number = self.number + 1
        self._next_target_i = self._next_target_i + self._is_pass_next
        self._past_targets_num = self._past_targets_num + self._is_pass_next
        return self._next_target_i == len(self.targets)-1

    def process(self):
            if self.position is None:
                rospy.logwarn(f"{self.log_prefix} No odom!")
                return
            if self.depth_image is None:
                rospy.logwarn(f"{self.log_prefix} No depth image!")
                return
            start_time = time.time()  # 记录开始时间
            
            # 处理目标点
            # start_targets = time.time()
            _next_targets_i_clamp = th.stack([self._next_target_i + i for i in range(self._next_target_num)]).T % len(self.targets)
            next_targets = self.targets[_next_targets_i_clamp.squeeze()]
            relative_pos_world = (next_targets - self.position.unsqueeze(0))
            relative_pos_body = self.world_to_body(relative_pos_world)
            relative_pos = relative_pos_body.reshape(self.num_envs, -1)
            # end_targets = time.time()
            # print(f"处理目标点耗时: {end_targets - start_targets:.4f} 秒")
            
            # 构建状态
            # start_state = time.time()
            state = th.hstack([
                relative_pos / self.max_sense_radius,
                self.orientation.unsqueeze(0),
                self.velocity.unsqueeze(0) / 10,
                self.angular_velocity.unsqueeze(0) / 10,
            ]).to(self.device)
            # end_state = time.time()
            # print(f"构建状态耗时: {end_state - start_state:.4f} 秒")
            
            # 处理深度图像
            # start_depth = time.time()
            depth_image = th.as_tensor(self.depth_image.astype(np.float32), dtype=th.float32).to(self.device)
            depth_image = depth_image.unsqueeze(0) / 1000.
            # end_depth = time.time()
            # print(f"处理深度图像耗时: {end_depth - start_depth:.4f} 秒")
            
            # 模型预测
            # start_predict = time.time()
            obs = {
                "depth": depth_image,
                "state": state, 
                "vd": self.v_d,
                "index": self._next_target_i,
                "latent": self.latent
            }
            obs = {k: v.cpu().numpy() for k, v in obs.items()}
            
            action, _ = self.model.predict(obs)
            # print(f"action: {action}")
            # action_msg = Float32MultiArray()
            # action_msg.data = action
            # self.action_pub.publish(action_msg)
            # action = self.traced_model(obs)
            command = self.de_normalize(th.as_tensor(action, device=self.device))
            # end_predict = time.time()
            # print(f"模型预测耗时: {end_predict - start_predict:.4f} 秒")
            
            # 发布控制指令
            # start_publish = time.time()
            ctbr_msg = Command()
            ctbr_msg.thrust = HOVER_ACC + command[0]
            ctbr_msg.angularVel.x, ctbr_msg.angularVel.y, ctbr_msg.angularVel.z = command[1:4]
            ctbr_msg.mode = Command.ANGULAR_MODE
            self.ctbr_pub.publish(ctbr_msg)
            # self.get_success()
            # end_publish = time.time()
            # print(f"发布控制指令耗时: {end_publish - start_publish:.4f} 秒")
            
            # 其他发布操作
            # start_other_publish = time.time()
            # state_msg = Float32Multi0Array()
            # state_msg.data = obs["state"]
            # action_msg = Float32MultiArray()
            # action_msg.data = command.numpy().astype(np.float32)
            # self.state_pub.publish(state_msg)    
            # self.action_pub.publish(action_msg)
            # self.traj_publish()
            # end_other_publish = time.time()
            # print(f"其他发布操作耗时: {end_other_publish - start_other_publish:.4f} 秒")
            
            end_time = time.time()  # 记录结束时间
            print(f"总耗时: {end_time - start_time:.4f} 秒")

if __name__ == "__main__":
    rospy.init_node("real_end2end")
    log_prefix = "[visfly_ctrl]"

    start_init = time.time()
    agent = RealEnv()
    end_init = time.time()
    print(f"初始化耗时: {end_init - start_init:.4f} 秒")

    rospy.loginfo(f"{log_prefix} running {__file__}.")
    rospy.loginfo(f"{log_prefix} Waiting for trigger.")

    rate = rospy.Rate(20)  # 30 Hz
    rospy.sleep(2.0)

    try:
        while not agent.get_success() and not rospy.is_shutdown():
            with th.no_grad():
                agent.process()
            rate.sleep()
    except rospy.ROSInterruptException:
        exit(0)

