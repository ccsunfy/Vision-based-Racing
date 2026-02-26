#!/usr/bin/env python3

import torch as th
import rospy
import numpy as np

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
from envs.waypoint_depth_noise import RacingEnv2
from envs.droneEnv import DroneEnvsBase

class HITL:
    def __init__(self, log_prefix=""):
        self.log_prefix = log_prefix
        self.scene_path = "datasets/spy_datasets/configs/garage_empty"
        self.env = RacingEnv2(num_agent_per_scene=1,
                            visual=True, # 不用视觉要改成False
                            max_episode_steps=512,
                            scene_kwargs={
                                 "path": self.scene_path,
                             },
                            )
        self.env_base = DroneEnvsBase(num_agent_per_scene=1,
                            visual=True, # 不用视觉要改成False
                            max_episode_steps=512,
                            scene_kwargs={
                                 "path": self.scene_path,
                             },
                            )
        rospy.loginfo(f"{self.log_prefix} preheating...")
        self.bridge = CvBridge()
        self.depth_image = None

        self.position = None
        self.orientation = None
        self.velocity = None
        self.angular_velocity = None

        # sub
        rospy.Subscriber("/bfctrl/local_odom", Odometry, self.odom_callback)

        # pub
        self.depth_pub = rospy.Publisher('/visfly/depth', Image, queue_size=10, tcp_nodelay=True)
    
    def odom_callback(self, data):
        orientation = data.pose.pose.orientation
        position = data.pose.pose.position
        velocity = data.twist.twist.linear
        angular_velocity = data.twist.twist.angular
        
        self.orientation = np.array([orientation.x, orientation.y, orientation.z, orientation.w], dtype=np.float32)
        rpy = R.from_quat(orientation).as_euler("xyz")
        self.position = np.array([position.x, position.y, position.z], dtype=np.float32)
        self.rpy = np.array(rpy, dtype=np.float32)
        self.velocity = np.array([velocity.x, velocity.y, velocity.z], dtype=np.float32)
        self.angular_velocity = np.array([angular_velocity.x, angular_velocity.y, angular_velocity.z], dtype=np.float32)
    
    def get_depth(self):
        self.env_base.reset()
        self.env_base.sceneManager.set_pose(self.position, self.orientation)
        self.env_base.update_observation()
        self.depth_image = self.env.get_observation()["depth"]
        
        
    def process(self):
        self.get_depth()
        if self.depth_image is not None:
            try:
                depth_image = self.bridge.imgmsg_to_cv2(self.depth_image, desired_encoding="passthrough")
                depth_image = np.array(depth_image, dtype=np.float32)
                depth_image = np.expand_dims(depth_image, axis=0)
                depth_image = np.transpose(depth_image, (0, 2, 3, 1))
                depth_image = np.clip(depth_image, 0.0, 10.0) / 10.0
                depth_image = th.from_numpy(depth_image).to(th.float32)
                
                # Publish the depth image
                self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depth_image[0].cpu().numpy(), encoding="passthrough"))
            except CvBridgeError as e:
                rospy.logerr(f"Error converting depth image: {e}")
                
if __name__ == "__main__":
    rospy.init_node("depth_HITL")
    log_prefix="depth_HITL"
    hitl = HITL()
    rospy.loginfo(f"{log_prefix} initialized.")
    rate = rospy.Rate(30)  
    while not rospy.is_shutdown():
        hitl.process()
        rate.sleep()