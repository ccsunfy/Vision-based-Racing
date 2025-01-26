#!/usr/bin/env python

import rospy
import torch as th
import numpy as np
import cv2
import tf.transformations
import time
import os
import math

from tqdm import tqdm
from sensor_msgs.msg import Image
from std_msgs.msg import Float32 
from collections import deque
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion, Twist
from nav_msgs.msg import Odometry
from mavros_msgs.msg import RCIn
from quadrotor_msgs.msg import Command,TRPYCommand
from cv_bridge import CvBridge, CvBridgeError
from utils.algorithms.ppo import ppo
from scipy.spatial.transform import Rotation as R

MODE_CHANNEL = 6 #改通道
HOVER_ACC = 9.80
HOVER_THRUST = 0.27
MODE_SHIFT_VALUE = 0.25

class RealEnv:
    def __init__(self, log_prefix=""):
        self.log_prefix = log_prefix
        rospy.loginfo(f"{self.log_prefix} preheating...")
        
        self.model_path = 'examples/nature_cross/saved/ppo_174.zip'
        self.model = ppo.load(self.model_path)
        rospy.loginfo(f"{self.log_prefix} Obs: {self.model.policy.observation_space.shape}")
        rospy.loginfo(f"{self.log_prefix} Action: {self.model.policy.action_space.shape}")
        
        self.bridge = CvBridge()
        self.depth_image = None

        self.position = None
        self.orientation = None
        self.velocity = None
        self.angular_velocity = None
        self.yaw = 0
        
        #松江场地最大长度
        self.max_sense_radius = 20.0
        self.success_radius = 0.3
        
        self.length_target = len(self.targets)
        self._next_target_num = 2
        self._next_target_i = np.zeros((self.num_envs,), dtype=np.int)
        self._past_targets_num = np.zeros((self.num_envs,), dtype=np.int)
        self._is_pass_next = np.zeros((self.num_envs,), dtype=np.bool)
        
        self.latent = np.zeros(256, dtype=np.float32)
        
        self.targets = th.as_tensor([
            [3, 0, 1],    # 第一个门
            [6, -1, 1],    # 第二个门
            [9, 1, 1],   # 第三个门
            [12, 0, 1],   # 第四个门
            [14, 0, 1],   # 第五个门
        ])
        
        # sub
        self.depth_image = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
        rospy.Subscriber("~odom", Odometry, self.odom_callback)
        rospy.Subscriber('/mavros/rc/in', RCIn, self.rc_callback, queue_size=10)
        # rospy.Subscriber('/bfctrl/local_odom',Odometry, self.update_odometry, queue_size=10)
        
        # pub
        self.ctbr_pub = rospy.Publisher('/cmd', Command, queue_size=10, tcp_nodelay=True)
        # self.rpy_pub = rospy.Publisher('~rpy', TRPYCommand, queue_size=10, tcp_nodelay=True)
        
    def depth_callback(self):
        try:
            # 将 ROS 图像消息转换为 OpenCV 图像
            cv_image = self.bridge.imgmsg_to_cv2(self.depth_image, desired_encoding="passthrough")
            self.depth_image = cv2.resize(cv_image, (64, 64))
        except CvBridgeError as e:
            print(e)
    
    def rc_callback(self, data: RCIn):
        if (data.channels[MODE_CHANNEL] - 1000.0) / 1000 < MODE_SHIFT_VALUE:
            if not self.mode_enable:
                rospy.logwarn(f"{self.log_prefix} mode enable.")
            self.mode_enable = True
        else:
            self.mode_enable = False
    
    def odom_callback(self, data):
        orientation = data.pose.pose.orientation
        position = data.pose.pose.position
        velocity = data.twist.twist.linear
        angular_velocity = data.twist.twist.angular
        
        orientation = np.array([orientation.x, orientation.y, orientation.z, orientation.w], dtype=np.float32)
        # print("q: ", q)
        rpy = R.from_quat(orientation).as_euler("xyz")
        self.position = np.array([position.x, position.y, position.z], dtype=np.float32)
        self.rpy = np.array(rpy, dtype=np.float32)
        # rospy.loginfo(f"{self.log_prefix} cur rpy: {self.rpy}")
        # print(f"{self.log_prefix} cur rpy: {self.rpy}")
        self.velocity = np.array([velocity.x, velocity.y, velocity.z], dtype=np.float32)
        self.angular_velocity = np.array([angular_velocity.x, angular_velocity.y, angular_velocity.z], dtype=np.float32)

        # self.traj_list.append(self.position)
        

    def world_to_body(self, relative_pos_world):
        # 使用四元数将世界坐标系下的相对坐标转换到机体系下
        rotation = R.from_quat(self.orientation.cpu().numpy())
        rotation_matrix = th.from_numpy(rotation.as_matrix()).to(self.device).float()  # 确保 rotation_matrix 是 float 类型
        relative_pos_world = relative_pos_world.float()  # 确保 relative_pos_world 是 float 类型

        # 进行矩阵乘法，将世界坐标系下的相对坐标转换到机体系下
        relative_pos_body = th.einsum('bij,bjk->bik', rotation_matrix.transpose(1, 2), relative_pos_world.transpose(1, 2)).transpose(1, 2)
        return relative_pos_body

    def get_success(self) -> th.Tensor:
        _next_target_i_clamp = self._next_target_i
        self._is_pass_next = ((self.position - self.targets[_next_target_i_clamp]).norm(dim=1) <= self.success_radius)
        self._next_target_i = self._next_target_i + self._is_pass_next
        # self._next_target_i = self._next_target_i % len(self.targets)
        self._past_targets_num = self._past_targets_num + self._is_pass_next
        return self._next_target_i == len(self.targets)-1

    def get_observation(self):
        
        # image preprocess
        depth_image = self.depth_image / 1000.0  # 假设深度图像以毫米为单位，需要转换为米
        depth_image = th.tensor(depth_image, dtype=th.float32).unsqueeze(0).unsqueeze(0)
        # relative_pos = (next_targets - self.position.unsqueeze(1)).reshape(self.num_envs, -1)

    def process(self, action):
        if self.position is None:
            rospy.logwarn(f"{self.log_prefix} No odom!")
            return
        if self.depth_image is None:
            rospy.logwarn(f"{self.log_prefix} No depth image!")
            return
        
        _next_targets_i_clamp = th.stack([self._next_target_i + i for i in range(self._next_target_num)]).T % len(self.targets)
        next_targets = self.targets[_next_targets_i_clamp]
        
        relative_pos_world = (next_targets - self.position.unsqueeze(1))
        relative_pos_body = self.world_to_body(relative_pos_world)
        relative_pos = relative_pos_body.reshape(self.num_envs, -1)
        
        state = th.hstack([
            relative_pos / self.max_sense_radius,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)
                
        # observation
        obs = {
            "depth": self.depth_image,
            "state": state, 
            "index": self._next_target_i,
            "latent": self.latent
        }

        action, _ = self.model.predict(obs)
        
        ctbr_msg = Command()
        # ctbr_msg.thrust = HOVER_ACC + self.k_acc * action[0]
        ctbr_msg.thrust = action[0]
        ctbr_msg.angularVel.x, \
        ctbr_msg.angularVel.y, \
        ctbr_msg.angularVel.z = action[1:4]
        ctbr_msg.mode = Command.ANGULAR_MODE
        self.ctbr_pub.publish(ctbr_msg)


if __name__ == "__main__":
    rospy.init_node("real_end2end")
    log_prefix = "[visfly_ctrl]"
    agent = RealEnv()
    rospy.loginfo(f"{log_prefix} running {__file__}.")
    rospy.loginfo(f"{log_prefix} Waiting for mode selection and trigger.")

    rate = rospy.Rate(30)  # 30 Hz

    try:
        while not agent.task_success and not rospy.is_shutdown():
            agent.process()
            rate.sleep()
    except rospy.ROSInterruptException:
        exit(0)