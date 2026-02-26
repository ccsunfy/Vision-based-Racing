import json
import random
import os

def regenerate_obstacles(file_path, num_obstacles=18):
    # 读取现有的 JSON 文件
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 获取现有的 object_instances
    object_instances = data["object_instances"]

    # 清空现有的障碍物
    # object_instances = [obj for obj in object_instances if obj["template_name"] != "0.1_0.1_2cylinder"]
    object_instances = [obj for obj in object_instances if obj["template_name"] != "0.15_0.15tree"]
    
    # #########################demo1############################
    
    # # # 定义需要避开的禁区点列表
    # exclusion_points = [(-1, 0, 0),(-5, 0, 0),(-9, 0, 0),(-12, 0, 0)]
    
    # # 生成第一段障碍物区域（y轴范围，z固定，x轴范围）
    # for _ in range(int(num_obstacles/3)):
    #     while True:
    #         y = random.uniform(-2., 2.)
    #         x = random.uniform(-6.5, -3.5)
    #         z = 0.0
    #         # 检查是否靠近任何禁区点
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:  # 距离平方小于0.5^2
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.15_0.15tree",
    #         "translation": [y, z, x],  # 原坐标顺序为[y, z, x]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
    
    # # 生成第二段障碍物区域（x轴范围，y固定，z轴范围）
    # for _ in range(int(num_obstacles/3)):
    #     while True:
    #         x = random.uniform(-2., 2.)
    #         z = random.uniform(-10.5, -7.5)
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.15_0.15tree",
    #         "translation": [x, y, z],  # 原坐标顺序为[x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
    
    # # 生成第三段障碍物区域（x轴范围，y固定，z轴范围）
    # for _ in range(int(num_obstacles/3)):
    #     while True:
    #         x = random.uniform(-2., 2.)
    #         z = random.uniform(-14.5, -11.5)
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.15_0.15tree",
    #         "translation": [x, y, z],  # 原坐标顺序为[x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)

#############################demo2#####################################
    # 随机生成障碍物的位置和旋转
    exclusion_points = [(-2, -3, 0),(-6, 3, 0),(-7, -2, 0),(-1, 2, 0)]
    
    for _ in range(int(num_obstacles/4)):
        while True:
            x = random.uniform(-3., -1.)  # x 坐标
            z = random.uniform(-7.5, -4.5)  # z 坐标
            y = 0.0
            valid = True
            for (px, py, pz) in exclusion_points:
                dx = x - px
                dy = y - py
                dz = z - pz
                if dx**2 + dy**2 + dz**2 < 0.25:
                    valid = False
                    break
            if valid:
                break
        obstacle = {
            "template_name": "0.1_0.1_2cylinder",
            "translation": [x, y, z],  # 原坐标顺序为[x, y, z]
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "uniform_scale": 1.0,
            "motion_type": "STATIC",
            "translation_origin": "COM"
        }
        object_instances.append(obstacle)
        
    for _ in range(int(num_obstacles/4)):
        while True:
            x = random.uniform(1., 3.) # x 坐标
            z = random.uniform(-7.5, -4.5)  # z 坐标
            y = 0.0
            valid = True
            for (px, py, pz) in exclusion_points:
                dx = x - px
                dy = y - py
                dz = z - pz
                if dx**2 + dy**2 + dz**2 < 0.25:
                    valid = False
                    break
            if valid:
                break
        obstacle = {
            "template_name": "0.1_0.1_2cylinder",
            "translation": [x, y, z],  # 原坐标顺序为[x, y, z]
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "uniform_scale": 1.0,
            "motion_type": "STATIC",
            "translation_origin": "COM"
        }
        object_instances.append(obstacle)
        
    for _ in range(int(num_obstacles/4)):
        while True:
            x = random.uniform(1., 3.)  # x 坐标
            z = random.uniform(-3.5, -0.5)   # z 坐标
            y = 0.0
            valid = True
            for (px, py, pz) in exclusion_points:
                dx = x - px
                dy = y - py
                dz = z - pz
                if dx**2 + dy**2 + dz**2 < 0.25:
                    valid = False
                    break
            if valid:
                break
        obstacle = {
            "template_name": "0.1_0.1_2cylinder",
            "translation": [x, y, z],  # 原坐标顺序为[x, y, z]
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "uniform_scale": 1.0,
            "motion_type": "STATIC",
            "translation_origin": "COM"
        }
        object_instances.append(obstacle)
        
    for _ in range(int(num_obstacles/4)):
        while True:
            x = random.uniform(-3., -1.)   # x 坐标
            z = random.uniform(-3.5, -0.5)   # z 坐标
            y = 0.0
            valid = True
            for (px, py, pz) in exclusion_points:
                dx = x - px
                dy = y - py
                dz = z - pz
                if dx**2 + dy**2 + dz**2 < 0.25:
                    valid = False
                    break
            if valid:
                break
        obstacle = {
            "template_name": "0.1_0.1_2cylinder",
            "translation": [x, y, z],  # 原坐标顺序为[x, y, z]
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "uniform_scale": 1.0,
            "motion_type": "STATIC",
            "translation_origin": "COM"
        }
        object_instances.append(obstacle)
############################################################################################################
    #         # 随机生成障碍物的位置和旋转 demo3
    # exclusion_points = [(-2, -3, 0),(-10, 3, 0),(-10, -3, 0),(-2, 3, 0)]
    # for _ in range(int(num_obstacles/6)):
    #     while True:
    #         x = random.uniform(-3.5, -1.)   # x 坐标
    #         z = random.uniform(-3.5, -1.)   # z 坐标
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.15_0.15tree",
    #         "translation": [x, y, z],  # 原坐标顺序为[x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
    
    # for _ in range(int(num_obstacles/6)):
    #     while True:
    #         x = random.uniform(-3.5, -1.)   # x 坐标
    #         z = random.uniform(-7.5, -4.5)   # z 坐标
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.15_0.15tree",
    #         "translation": [x, y, z],  # 原坐标顺序为[x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
    # for _ in range(int(num_obstacles/6)):
    #     while True:
    #         x = random.uniform(-3.5, -1.)   # x 坐标
    #         z = random.uniform(-11., -8.5)   # z 坐标
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.1_0.1_2cylinder",
    #         "translation": [x, y, z],  # 原坐标顺序为[x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
    
    # for _ in range(int(num_obstacles/6)):
    #     while True:
    #         x = random.uniform(1, 3.5)   # x 坐标
    #         z = random.uniform(-3.5, -1.)   # z 坐标
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.1_0.1_2cylinder",
    #         "translation": [x, y, z],  # 原坐标顺序为[x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
    
    # for _ in range(int(num_obstacles/6)):
    #     while True:
    #         x = random.uniform(1, 3.5)   # x 坐标
    #         z = random.uniform(-7.5, -4.5)   # z 坐标
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.15_0.15tree",
    #         "translation": [x, y, z],  # 原坐标顺序为[x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
    # for _ in range(int(num_obstacles/6)):
    #     while True:
    #         x = random.uniform(1, 3.5)   # x 坐标
    #         z = random.uniform(-11., -8.5)   # z 坐标
    #         y = 0.0
    #         valid = True
    #         for (px, py, pz) in exclusion_points:
    #             dx = x - px
    #             dy = y - py
    #             dz = z - pz
    #             if dx**2 + dy**2 + dz**2 < 0.25:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     obstacle = {
    #         "template_name": "0.15_0.15tree",
    #         "translation": [x, y, z],  # 原坐标顺序为[x, y, z]
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "uniform_scale": 1.0,
    #         "motion_type": "STATIC",
    #         "translation_origin": "COM"
    #     }
    #     object_instances.append(obstacle)
############################################################################################################
        
    # 更新 JSON 数据
    data["object_instances"] = object_instances

    return data


def generate_random_json_files(base_file_path, output_dir, num_files=4):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取基础 JSON 文件
    base_data = regenerate_obstacles(base_file_path)

    for i in range(num_files):
        # 为每个文件生成不同的障碍物数量n
        # num_obstacles = random.randint(5, 10)  # 随机选择障碍物数量
        num_obstacles = 16
        modified_data = regenerate_obstacles(base_file_path, num_obstacles)

        # 生成随机文件名
        random_filename = f"racing_{i+1}.scene_instance.json"
        output_file_path = os.path.join(output_dir, random_filename)

        # 将更新后的 JSON 数据写入新文件
        with open(output_file_path, 'w') as file:
            json.dump(modified_data, file, indent=4)

        print(f"文件 '{random_filename}' 已成功生成并添加到 '{output_dir}' 目录中。")

# example usage

# base_file_path = 'datasets/spy_datasets/configs/racing/racing_1.scene_instance.json'
# output_directory = 'datasets/spy_datasets/configs/racing'
# base_file_path = 'datasets/spy_datasets/configs/racing8_random_ob/racing_1.scene_instance.json'
# output_directory = 'datasets/spy_datasets/configs/racing8_random_ob'
# base_file_path = 'datasets/spy_datasets/configs/racing_straight/racing_1.scene_instance.json'
# output_directory = 'datasets/spy_datasets/configs/racing_straight'
# base_file_path = 'datasets/spy_datasets/configs/demo1_straight_ob1/racing_1.scene_instance.json'
# output_directory = 'datasets/spy_datasets/configs/demo1_straight_random_ob'
base_file_path = 'datasets/spy_datasets/configs/demo2_3Dcircle_ob1/racing_1.scene_instance.json'
output_directory = 'datasets/spy_datasets/configs/demo2_3Dcircle_random_ob'
# base_file_path = 'datasets/spy_datasets/configs/demo1_songjiang/racing_1.scene_instance.json'
# output_directory = 'datasets/spy_datasets/configs/demo1_songjiang'
# base_file_path = 'datasets/spy_datasets/configs/demo2_songjiang/racing_1.scene_instance.json'
# output_directory = 'datasets/spy_datasets/configs/demo2_songjiang'
# base_file_path = 'datasets/spy_datasets/configs/demo3_songjiang/racing_1.scene_instance.json'
# output_directory = 'datasets/spy_datasets/configs/demo3_songjiang'
generate_random_json_files(base_file_path, output_directory, num_files=5)

