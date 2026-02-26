#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from mavros_msgs.msg import AttitudeTarget
from stable_baselines3 import PPO
from mavros_msgs.msg import RCIn
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Pose, PoseArray, Point
import numpy as np
from scipy.spatial.transform import Rotation
from quadrotor_msgs.msg import Command
from mavros_msgs.msg import AttitudeTarget
from visualization_msgs.msg import Marker
import time

USE_LAST_ACTION = True
HOVER_ACC = 9.80
HOVER_THRUST = 0.30
MODE_CHANNEL = 6
MODE_SHIFT_VALUE = 0.25

class PosYawCtrl:
    def __init__(self, log_prefix = "") -> None:
        self.log_prefix = log_prefix
        rospy.loginfo(f"{self.log_prefix} preheating...")
        self.target_p = np.zeros(3)
        self.target_yaw = 0
        self.target_array = None
        self.target_count = 0
        self.target_dist_threshold = 0.1
        self.target_yaw_threshold = 0.2
        self.weight = None
        self.traj_list = []

        self.p = None
        self.rpy = None
        self.v = None
        self.w = None

        self.task_success = False
        self.hover_enable = False
        self.hover_success_count = 0
        self.mode_enable = False
        self.triggered = False
        self.first_triggered = False
        self.pub_mavros_cmd = False
        self.k_thrust = 0.2
        self.k_acc = 2.0

        self.success_poses_msg = PoseArray()
        self.success_poses_msg.header.frame_id = "world"

        ### sub
        rospy.Subscriber("~odom", Odometry, self.odom_callback)
        rospy.Subscriber('/mavros/rc/in', RCIn, self.rc_callback, queue_size=10)
        rospy.Subscriber('/bfctrl/traj_start_trigger', Bool, self.trigger, queue_size=10)
        
        ### pub
        if self.pub_mavros_cmd:
            self.ctbr_pub = rospy.Publisher('/cmd', AttitudeTarget, queue_size=10, tcp_nodelay=True)
        else:
            self.ctbr_pub = rospy.Publisher('/cmd', Command, queue_size=10, tcp_nodelay=True)
        self.obs_pub = rospy.Publisher('/obs', Float32MultiArray, queue_size=10)
        self.action_pub = rospy.Publisher('/action', Float32MultiArray, queue_size=10)
        self.targets_pub = rospy.Publisher('/rl/targets', PoseArray, queue_size=10)
        self.success_poses_pub = rospy.Publisher('/rl/success_poses', PoseArray, queue_size=10)
        self.traj_pub = rospy.Publisher('/rl/traj', Marker, queue_size=10)

        # read params from launch
        self.read_params()

        self.model = PPO.load(self.weight)
        rospy.loginfo(f"{self.log_prefix} Obs: {self.model.policy.observation_space.shape}")
        rospy.loginfo(f"{self.log_prefix} Action: {self.model.policy.action_space.shape}")
        self.action_size = self.model.policy.action_space.shape[1]
        self.obs_size = self.model.policy.observation_space.shape
        self.last_action = np.zeros(self.action_size)
        self.obs = np.zeros(self.obs_size)

        # set print options for numpy
        np.set_printoptions(precision=2, suppress=True)
    
    def read_params(self) -> None:
        # read target params
        self.target_dist_threshold = rospy.get_param('~target_dist_threshold', 0.1)
        self.target_yaw_threshold = rospy.get_param('~target_yaw_threshold', 0.2)
        self.hover_enable = rospy.get_param('~hover_enable', False)

        # read weight
        self.weight = rospy.get_param('~weight', '../weights/pos_yaw_ctrl/20241119_20_49_obs16act4.zip')
        rospy.loginfo(f"{self.log_prefix} Load weight from: {self.weight}")
    
    def rc_callback(self, data: RCIn):
        if (data.channels[MODE_CHANNEL] - 1000.0) / 1000 < MODE_SHIFT_VALUE:
            if not self.mode_enable:
                rospy.logwarn(f"{self.log_prefix} mode enable.")
            self.mode_enable = True
        else:
            self.mode_enable = False

    def odom_callback(self, data):
        p = data.pose.pose.position
        q = data.pose.pose.orientation
        v = data.twist.twist.linear
        w = data.twist.twist.angular

        q = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
        # print("q: ", q)
        rpy = Rotation.from_quat(q).as_euler("xyz")
        self.p = np.array([p.x, p.y, p.z], dtype=np.float32)
        self.rpy = np.array(rpy, dtype=np.float32)
        # rospy.loginfo(f"{self.log_prefix} cur rpy: {self.rpy}")
        # print(f"{self.log_prefix} cur rpy: {self.rpy}")
        self.v = np.array([v.x, v.y, v.z], dtype=np.float32)
        self.w = np.array([w.x, w.y, w.z], dtype=np.float32)

        self.traj_list.append(self.p)

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

            target_quat = Rotation.from_euler("xyz", [0, 0, target[3]]).as_quat()
            pose.orientation.x = target_quat[0]
            pose.orientation.y = target_quat[1]
            pose.orientation.z = target_quat[2]
            pose.orientation.w = target_quat[3]

            pose_array_msg.poses.append(pose)
        self.targets_pub.publish(pose_array_msg)
    
    def success_pos_publish(self):
        self.success_poses_msg.header.stamp = rospy.Time.now()
        pose = Pose()
        pose.position.x = self.p[0]
        pose.position.y = self.p[1]
        pose.position.z = self.p[2]

        target_quat = Rotation.from_euler("xyz", self.rpy).as_quat()
        pose.orientation.x = target_quat[0]
        pose.orientation.y = target_quat[1]
        pose.orientation.z = target_quat[2]
        pose.orientation.w = target_quat[3]
        self.success_poses_msg.poses.append(pose)

        self.success_poses_pub.publish(self.success_poses_msg)

    def set_target(self):
        points = []
        point_num = rospy.get_param('~point_num', 1)
        if self.hover_enable:
            hover_point = np.hstack((self.p, self.rpy[2]))
            points.append(hover_point)
        elif point_num > 0:
            for i in range(point_num):
                point = [
                    rospy.get_param(f'~point{i}_x', 0.0),
                    rospy.get_param(f'~point{i}_y', 0.0),
                    rospy.get_param(f'~point{i}_z', 0.0),
                    rospy.get_param(f'~point{i}_yaw', 0.0)
                ]
                points.append(point)
        else:
            rospy.logerr(f"{self.log_prefix} No target points.")
            exit(0)
        self.target_array = np.array(points)
        rospy.loginfo(f"{self.log_prefix} Target Array: \n {self.target_array}")
        # set target to first point
        self.target_p = self.target_array[self.target_count][:3]
        self.target_yaw = self.target_array[self.target_count][3]

    def trigger(self, msg: Bool):
        if msg.data:
            if self.mode_enable and not self.first_triggered:
                self.set_target()
                self.triggered = True
                self.first_triggered = True
                self.targets_publish()
                rospy.logwarn(f"{self.log_prefix} Start pos_yaw_ctrl for {len(self.target_array)} targets.")
            elif self.mode_enable and self.first_triggered:
                rospy.logwarn(
                        f"{self.log_prefix} Triggered! Continue to fly to target "
                        f"{self.target_count}: {self.target_array[self.target_count]}."
                    )
                self.triggered = True
            else:
                rospy.logwarn(f"{self.log_prefix} Mode not enabled, enable mode first and trigger again.")
        else:
            if self.triggered:
                rospy.logwarn(f"{self.log_prefix} Stop pos_yaw_ctrl.")
            self.triggered = False

    def reached_target(self):
        return np.linalg.norm(self.p - self.target_p) < self.target_dist_threshold and \
                    abs(self.rpy[2] - self.target_yaw) < self.target_yaw_threshold
    
    def process(self, ):
        if self.p is None:
            rospy.logwarn(f"{self.log_prefix} No odom!")
            return

        if self.mode_enable and self.triggered:
            ### update target ###
            if self.hover_enable:
                if self.reached_target():
                    self.hover_success_count += 1
                    if self.hover_success_count > 300:
                        self.success_pos_publish()
                        rospy.logerr(f"{self.log_prefix} reach hover target: {self.target_array[self.target_count]}")
                        self.task_success = True
                        return
            elif self.reached_target():
                if self.target_count == len(self.target_array) - 1:
                    self.task_success = True
                    self.success_pos_publish()
                    rospy.logerr(f"{self.log_prefix} reach final target: {self.target_array[self.target_count]}")
                    return
                else:
                    self.success_pos_publish()
                    rospy.logwarn(f"{self.log_prefix} reach target {self.target_count}: {self.target_array[self.target_count]}")
                    rospy.logwarn(f"{self.log_prefix} waiting for next trigger...")
                    self.target_count += 1
                    self.target_p = self.target_array[self.target_count][:3]
                    self.target_yaw = self.target_array[self.target_count][3]
                    self.triggered = False # wait for next trigger
                    return
            
            ### step ###
            if USE_LAST_ACTION:
                self.obs[0, -self.action_size:] = self.last_action

            self.obs[0, :12] = np.hstack(
                (
                    self.p - self.target_p,
                    self.rpy[0:2],
                    self.rpy[2] - self.target_yaw,
                    self.v,
                    self.w
                )
            )

            action, _ = self.model.predict(self.obs, deterministic=True)
            action = action[0]
            self.last_action = action

            if self.pub_mavros_cmd:
                ctbr_msg = AttitudeTarget()
                ctbr_msg.thrust = HOVER_THRUST * self.k_thrust * action[0]
                ctbr_msg.body_rate.x, \
                    ctbr_msg.body_rate.y, \
                        ctbr_msg.body_rate.z = action[1:4]
                ctbr_msg.type_mask = 128  # 128 = 0b10000000 (IGNORE_ATTITUDE)
                self.ctbr_pub.publish(ctbr_msg)
            else: 
                ctbr_msg = Command()
                ctbr_msg.thrust = HOVER_ACC + self.k_acc * action[0]
                ctbr_msg.angularVel.x, \
                    ctbr_msg.angularVel.y, \
                        ctbr_msg.angularVel.z = action[1:4]
                ctbr_msg.mode = Command.ANGULAR_MODE
                self.ctbr_pub.publish(ctbr_msg)

            # pub infos for debug
            obs_msg = Float32MultiArray()
            obs_msg.data = self.obs[0].astype(np.float32).tolist()
            action_msg = Float32MultiArray()
            action_msg.data = action.astype(np.float32).tolist()
            self.obs_pub.publish(obs_msg)    
            self.action_pub.publish(action_msg)
            self.traj_publish()

if __name__ == '__main__':
    rospy.init_node("pos_yaw_ctrl_node")
    log_prefix = "[py_ctrl]"
    agent = PosYawCtrl(log_prefix=log_prefix)
    rospy.loginfo(f"{log_prefix} running {__file__}.")
    rospy.loginfo(f"{log_prefix} Waiting for mode selection and trigger.")

    rate = rospy.Rate(30)  # rl ctrl freq

    try:
        while not agent.task_success and not rospy.is_shutdown():
            agent.process()
            rate.sleep()
    except rospy.ROSInterruptException:
        exit(0)