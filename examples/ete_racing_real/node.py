#!/usr/bin/env python3

import signal
from time import time
from tqdm import tqdm
import onnxruntime as ort

import numpy as np
import quaternion
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from quadrotor_msgs.msg import Command
from cv_bridge import CvBridge
from quadrotor_msgs.msg import TakeoffLand, SlowDown

no_odom = False
reparam = False
from_topic = True
in_features = 7 if no_odom else 10
sr = 2
rospy.init_node("e2e_planner")
target_speed = float(rospy.get_param("~target_speed", 5.0))


def quaternion_to_matrix(q):
    return np.array(
        [
            [
                1 - 2 * (q[2] ** 2 + q[3] ** 2),
                2 * (q[1] * q[2] - q[3] * q[0]),
                2 * (q[1] * q[3] + q[2] * q[0]),
            ],
            [
                2 * (q[1] * q[2] + q[3] * q[0]),
                1 - 2 * (q[1] ** 2 + q[3] ** 2),
                2 * (q[2] * q[3] - q[1] * q[0]),
            ],
            [
                2 * (q[1] * q[3] - q[2] * q[0]),
                2 * (q[2] * q[3] + q[1] * q[0]),
                1 - 2 * (q[1] ** 2 + q[2] ** 2),
            ],
        ],
        dtype=np.float32,
    )


class E2EPlanner:
    def __init__(self) -> None:
        weight = rospy.get_param(
            "~weight",
            "/home/zlz/AirsimDji/IntelligentUAVChampionshipSimulator/roswrapper/ros/src/e2e_planner/weight/weight_odom_0911.onnx",
        )
        self.extra = np.array([rospy.get_param("~margin", 0.2)], np.float32)

        self.bridge = CvBridge()
        self.model = ort.InferenceSession(weight)

        # preheating
        rospy.loginfo("[e2e_planner] preheating")
        for _ in range(6):
            self.model.run(
                None,
                {
                    "img": np.zeros((1, 1, 12 * sr, 16 * sr), dtype=np.float32),
                    "state": np.zeros((1, in_features), dtype=np.float32),
                    "h": np.zeros((1, 192), dtype=np.float32),
                },
            )

        self.p = None
        self.q = None
        self.v = None
        self.h = np.zeros((1, 192), dtype=np.float32)
        self.height = 0
        self.p_target = np.array([4000000.0, 0.0, 2], np.float32)

        if from_topic:
            rospy.Subscriber(
                "/airsim_node/drone_1/front_center/DepthPlanar",
                Image,
                self.update_depth,
                queue_size=10,
            )
        self.beep_pub = rospy.Publisher(
            "/mavros/beep", Bool, queue_size=10, tcp_nodelay=True
        )
        rospy.Subscriber(
            "/bfctrl/local_odom", Odometry, self.update_odometry, queue_size=10
        )

        rospy.Subscriber("/traj_start_trigger", Bool, self.trigger, queue_size=10)
        self.v_est_pub = rospy.Publisher("~v_est", Point, queue_size=10)
        self.a_set_pub = rospy.Publisher("~a_set", Point, queue_size=10)
        self.rpy_pub = rospy.Publisher(
            "/bfctrl/cmd", Command, queue_size=10, tcp_nodelay=True
        )
        self.shut_pub = rospy.Publisher(
            "/bfctrl/slow_down", SlowDown, queue_size=10, tcp_nodelay=True
        )
        self.plan_until_t = float(rospy.get_param("~plan_until_t", 0))
        self.pre_head_countdown = None
        self.pbar = None
        self.ready = False
        self.start_plan_t = 0.0
        # speed adjustment
        self.record_p = None
        self.is_down = False
        self.target_speed = target_speed
        self.stop_flag = False

    def close(self):
        if self.is_down:
            return
        self.is_down = True
        if self.record_p is not None:
            self.record_p.send_signal(signal.SIGINT)
            self.record_p.wait()

    def update_odometry(self, data: Odometry):
        p = data.pose.pose.position
        q = data.pose.pose.orientation
        v = data.twist.twist.linear
        self.p = (p.x, p.y, p.z)
        self.q = (q.w, q.x, q.y, q.z)
        self.v = (v.x, v.y, v.z)
        self.height = p.z

    def update_depth(self, data):
        depth = self.bridge.imgmsg_to_cv2(data)
        if time() > self.plan_until_t:
            return

        self.process(depth[None, None, ::20, ::20])

    def process(self, depth):
        if self.p is None:
            rospy.logwarn("No odom")
            return
        if self.stop_flag:
            return
        if self.p[0] > 35.1:
            shut = SlowDown()
            shut.x_acc = 4.0
            shut.y_acc = 4.0
            shut.header.frame_id = "world"
            shut.header.stamp = rospy.Time.now()
            self.plan_until_t = time() - 5
            self.shut_pub.publish(shut)
            self.stop_flag = True
            return

        t = time() - self.start_plan_t
        target_x = 4000
        target_y = 0  # 8 * math.sin(5 * t)
        target_z = 5.0
        self.p_target[0] = target_x
        self.p_target[1] = target_y
        self.p_target[2] = target_z
        p = np.array(self.p, dtype=np.float32)
        v = np.array(self.v, dtype=np.float32)
        env_R = quaternion_to_matrix(self.q)

        fwd = env_R[:, 0].copy()
        up = np.zeros_like(fwd)
        fwd[2] = 0
        up[2] = 1
        fwd = fwd / np.linalg.norm(fwd)
        R = np.stack([fwd, np.cross(up, fwd), up], -1)

        target_v = self.p_target - p
        target_v_norm = np.linalg.norm(target_v)
        target_v = target_v / target_v_norm * min(target_v_norm, self.target_speed)

        state = [target_v[None] @ R, env_R[None, 2], self.extra[None]]
        if not no_odom:
            state.insert(0, v[None] @ R)
        state = np.concatenate(state, -1)
        # state = (state - states_mean) / states_std
        act, self.h = self.model.run(None, {"img": depth, "state": state, "h": self.h})
        # act = act * action_std + action_mean
        v_setpoint, v_est = (R @ act.reshape(3, -1)).T

        # obtain acceleration setpoint
        a_setpoint = v_setpoint - v_est

        if reparam:
            x, y, z = a_setpoint
            z = np.log1p(np.exp(z + 9.8))  # z = softplus(z + 9.8)
            a_setpoint = np.array([x * z / 9.8, y * z / 9.8, z - 9.8], dtype=np.float32)
        a_set_debug = a_setpoint.copy()
        a_setpoint[2] += 9.8

        # convert acceleration setpoint to rpy throttle
        throttle = np.linalg.norm(a_setpoint)
        up_vec = a_setpoint / throttle

        # forward vector is the normalized moving average of target vector
        forward_vec = env_R[:, 0] * 5 + self.p_target - p
        forward_vec[2] = (
            forward_vec[0] * up_vec[0] + forward_vec[1] * up_vec[1]
        ) / -up_vec[2]
        forward_vec /= np.linalg.norm(forward_vec)
        left_vec = np.cross(up_vec, forward_vec)

        roll = np.arctan2(left_vec[2], up_vec[2])
        pitch = np.arcsin(-forward_vec[2])
        yaw = np.arctan2(forward_vec[1], forward_vec[0])

        if self.pre_head_countdown > 0:
            self.pre_head_countdown -= 1
            if self.pre_head_countdown == 0:
                self.h = np.zeros((1, 192), dtype=np.float32)
        else:
            msg = Command()
            msg.mode = Command().QUAT_MODE
            msg.thrust = throttle
            q = np.quaternion(1, 0, 0, 0)
            q_yaw = np.quaternion(np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
            q_pitch = np.quaternion(np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0)
            q_roll = np.quaternion(np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0)

            # 将这些旋转合成一个四元数
            q = q_yaw * q_pitch * q_roll

            msg.quat.w = q.w
            msg.quat.x = q.x
            msg.quat.y = q.y
            msg.quat.z = q.z

            self.rpy_pub.publish(msg)
            self.a_set_pub.publish(*a_set_debug.tolist())
            self.v_est_pub.publish(*v_est.tolist())
        self.pbar.update()

    def trigger(self, msg: Bool):
        if msg.data:
            rospy.loginfo(f"Start planning at speed {self.target_speed}")
            self.h = np.zeros((1, 192), dtype=np.float32)
            self.pre_head_countdown = 7
            self.pbar = tqdm()
            self.plan_until_t = time() + 30
        else:
            if isinstance(self.pbar, tqdm):
                self.pbar.close()
            self.plan_until_t = 0

    def trigger_outside(self, trigger):
        if trigger:
            rospy.loginfo(f"Start planning at speed {self.target_speed}")
            self.h = np.zeros((1, 192), dtype=np.float32)
            self.pre_head_countdown = 7
            self.pbar = tqdm()
            self.start_plan_t = time()
            self.plan_until_t = self.start_plan_t + 30.0
        else:
            if isinstance(self.pbar, tqdm):
                self.pbar.close()
            self.plan_until_t = 0


if __name__ == "__main__":

    planner = E2EPlanner()
    rospy.sleep(1)
    rospy.loginfo(f"[e2e_planner] running {__file__}.")
    rospy.loginfo("[e2e_planner] Waiting for trigger.")
    takeoff = TakeoffLand()
    takeoff.takeoff_land_cmd = TakeoffLand().TAKEOFF
    takeoff.takeoff_height = 1.5
    takeoffCmdPub = rospy.Publisher("/bfctrl/takeoff_land", TakeoffLand, queue_size=10)
    takeoffCmdPub.publish(takeoff)
    while planner.height < 0.3:
        rospy.sleep(0.01)
    rospy.sleep(5.0)
    planner.trigger_outside(True)
    land = TakeoffLand()
    land.takeoff_land_cmd = TakeoffLand().LAND
    if from_topic:
        while not rospy.is_shutdown():
            if planner.stop_flag:
                takeoffCmdPub.publish(land)

            rospy.sleep(0.066)
        exit(0)
    planner.close()
