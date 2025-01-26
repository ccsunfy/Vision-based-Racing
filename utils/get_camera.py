import habitat_sim
import numpy as np

# 创建模拟器配置
sim_cfg = habitat_sim.SimulatorConfiguration()
# sim_cfg.scene_id = "datasets/spy_datasets/configs/stages/frl_apartment_stage.stage_config.json"
sim_cfg.scene_id = "/home/suncc/SpySim6_24/datasets/spy_datasets/configs/stages/garage_v1.stage_config.json"
# 创建传感器配置
sensor_cfg = habitat_sim.CameraSensorSpec()
sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
sensor_cfg.resolution = [640, 480]
sensor_cfg.position = [0.0, 0.0, 0.0]  # 传感器位置
sensor_cfg.orientation = [0.0, 0.0, 0.0]  # 传感器方向

# 创建代理配置
agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [sensor_cfg]

# 创建模拟器
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

# 获取传感器内参
intrinsics = sensor_cfg.hfov  # 水平视场角
print(intrinsics)
width, height = sensor_cfg.resolution
fx = width / (2 * np.tan(np.deg2rad(float(intrinsics)) / 2))
fy = height / (2 * np.tan(np.deg2rad(float(intrinsics)) / 2))
cx = width / 2
cy = height / 2
camera_intrinsics = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

def quat_to_matrix(quat):
    """
    将四元数转换为旋转矩阵。
    :param quat: 四元数 [w, x, y, z]
    :return: 旋转矩阵 3x3
    """
    w, x, y, z = quat.w, quat.x, quat.y, quat.z
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    
# 获取传感器外参
camera_extrinsics = np.eye(4)
quat = habitat_sim.utils.quat_from_angle_axis(sensor_cfg.orientation[0], np.array([1.0, 0.0, 0.0])) * \
       habitat_sim.utils.quat_from_angle_axis(sensor_cfg.orientation[1], np.array([0.0, 1.0, 0.0])) * \
       habitat_sim.utils.quat_from_angle_axis(sensor_cfg.orientation[2], np.array([0.0, 0.0, 1.0]))
camera_extrinsics[:3, :3] = quat_to_matrix(quat)
camera_extrinsics[:3, 3] = sensor_cfg.position

print("Camera Intrinsics:\n", camera_intrinsics)
print("Camera Extrinsics:\n", camera_extrinsics)

# 关闭模拟器
sim.close()