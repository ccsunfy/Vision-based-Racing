import json
import random

def add_obstacles(scene_file, num_obstacles=50):
    with open(scene_file, 'r') as f:
        scene = json.load(f)
    
    # 定义安全区域(赛道区域)
    # safe_zone = {
    #     'x_min': -1.5,
    #     'x_max': 1.5,
    #     'z_min': -13,
    #     'z_max': -2
    # }
    
    # # 定义可放置区域(场景边界)
    # bounds = {
    #     'x_min': -4,
    #     'x_max': 4,
    #     'z_min': -13,
    #     'z_max': -2
    # }
    
    # safe_zone = {
    #     'x_min': -3.5,
    #     'x_max': 3.5,
    #     'z_min': -10.5,
    #     'z_max': -1.0
    # }
    
    safe_zone = {
        'x_min': -4,
        'x_max': 4,
        'z_min': -7,
        'z_max': -1
    }
    
    # 定义可放置区域(场景边界)
    bounds = {
        'x_min': -6,
        'x_max': 6,
        'z_min': -10,
        'z_max': 0
    }
    
    
    new_obstacles = []
    for _ in range(num_obstacles):
        # 随机选择左侧或右侧区域
        if random.random() < 0.5:
            x = random.uniform(bounds['x_min'], safe_zone['x_min'])
        else:
            x = random.uniform(safe_zone['x_max'], bounds['x_max'])
            
        z = random.uniform(bounds['z_min'], bounds['z_max'])
        
        obstacle = {
            "template_name": "0.1_0.1_2cylinder",
            "translation": [x, 0.0, z],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "uniform_scale": 1.0,
            "motion_type": "STATIC",
            "translation_origin": "COM"
        }
        new_obstacles.append(obstacle)
    
    scene['object_instances'].extend(new_obstacles)
    
    with open(scene_file, 'w') as f:
        json.dump(scene, f, indent=4)

# 使用示例
# scene_file = "datasets/spy_datasets/configs/demo1_straight_ob_duo/racing_1.scene_instance.json"
scene_file = "datasets/spy_datasets/configs/demo2_3Dcircle_random_ob_video/racing_1.scene_instance.json"
# scene_file = "datasets/spy_datasets/configs/demo3_ellipse_ob_duo/racing_1.scene_instance.json"
# scene_file = "datasets/spy_datasets/configs/demo3_J_ob_duo/racing_1.scene_instance.json"
add_obstacles(scene_file, num_obstacles=10)