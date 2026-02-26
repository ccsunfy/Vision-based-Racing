import numpy as np
import cv2
import json
import torch as th
from scipy.spatial.transform import Rotation as R

# file_path = 'datasets/spy_datasets/configs/special_circle/special_circle.scene_instance.json'
file_path = 'datasets/spy_datasets/configs/cross_circle/circle_debug.scene_instance.json'

def project_points(points, camera_matrix, dist_coeffs):
    # Project 3D points to 2D using the camera matrix and distortion coefficients
    points_2d, _ = cv2.projectPoints(points, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)

def draw_circle(image, points_2d):
    # Draw circles at the projected points
    points_2d = points_2d.astype(np.int32)
    for point in points_2d:
        center = tuple(point.astype(int))
        if 0 <= center[0] < image.shape[1] and 0 <= center[1] < image.shape[0]:
            # cv2.circle(image, center, 2, (255, 255, 255), -1)  # White color, filled circle
            cv2.polylines(image, [points_2d], isClosed=True, color=(255, 255, 255), thickness=1)

def world_to_camera(center, rvec, tvec):
    rotation = R.from_quat(rvec)
    r = rotation.as_matrix()
    camera_points = (r @ center.T).T + tvec
    return camera_points

def read_targets_from_json():
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    targets = []
    for obj in data.get("object_instances", []):
        if obj.get("template_name") == "red_circle":
            translation = obj.get("translation", [])
            target = [translation[0], translation[1], translation[2]]
            if target not in targets:
                targets.append(target)

    return targets

def main():
    circle_points = th.tensor(read_targets_from_json())
    # circle_points = th.tensor([[0.,1.,-3.],[0.,1.,-6.],[0.,1.,-9.]]) #直接用json里面的原始坐标即可
    print(circle_points)
    
    if len(circle_points) == 0:
        print("No targets found in the JSON file.")
        return

    center = circle_points.numpy()
        
    radius = 1.0  # 真实半径

    # 相机内参矩阵和畸变系数（示例值）
    camera_matrix = np.array([
        [42, 0, 42],
        [0, 42, 42],
        [0, 0, 1]
    ], dtype=np.float32)
    
    rvec = np.array([1, 0, 0, 0], dtype=np.float32)  # self.orientation
    tvec = np.array([0., 0.1, -1.], dtype=np.float32)  # self.position
    
    camera_point = world_to_camera(center, rvec, tvec)
    
    
    # 双球相机模型畸变
    # dist_coeffs = np.array([-0.2, 0.1, 0, 0], dtype=np.float32)  
    dist_coeffs = np.zeros(4)  # 假设没有畸变

    image = np.zeros((84, 84), dtype=np.uint8)

    # 生成并投影每个圆心的圆周上的点
    for c in camera_point:
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_points = np.vstack((radius * np.cos(theta) + c[0],
                                   radius * np.sin(theta) + c[1],
                                   np.ones_like(theta) * c[2])).T

        # 将圆周上的点投影到图像平面
        points_2d = project_points(circle_points, camera_matrix, dist_coeffs)

        # 在图像上绘制圆周上的点
        draw_circle(image, points_2d)

    # 显示图像
    cv2.imshow('Circle Projection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()