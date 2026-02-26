import json
import random
import os

def regenerate_obstacles(file_path, num_obstacles=30):
    # 读取现有的 JSON 文件
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 获取现有的 object_instances
    object_instances = data["object_instances"]

    # 清空现有的障碍物
    object_instances = [obj for obj in object_instances if obj["template_name"] != "0.1_0.1_2cylinder"]

    for _ in range(int(num_obstacles/3)):
        obstacle = {
            "template_name": "0.1_0.1_2cylinder",
            "translation": [
                random.uniform(-1., 1.),  # x 
                random.uniform(0.0, 0.0),   # y 
                random.uniform(-5., -3.5)  # z 
            ],
            "rotation": [
                1.0,  # w
                0.0,  # x
                0.0,  # y
                0.0   # z
            ],
            "uniform_scale": 1.0,
            "motion_type": "STATIC",
            "translation_origin": "COM"
        }
        object_instances.append(obstacle)
        
    for _ in range(int(num_obstacles/3)):
        obstacle = {
            "template_name": "0.1_0.1_2cylinder",
            "translation": [
                random.uniform(-1., 1.),  # x 
                random.uniform(0.0, 0.0),   # y 
                random.uniform(-8., -6.5)  # z 
            ],
            "rotation": [
                1.0,  # w
                0.0,  # x
                0.0,  # y
                0.0   # z
            ],
            "uniform_scale": 1.0,
            "motion_type": "STATIC",
            "translation_origin": "COM"
        }
        object_instances.append(obstacle)
        
    for _ in range(int(num_obstacles/3)):
        obstacle = {
            "template_name": "0.1_0.1_2cylinder",
            "translation": [
                random.uniform(-1., 1.),  # x 
                random.uniform(0.0, 0.0),   # y 
                random.uniform(-11., -9.5)  # z 
            ],
            "rotation": [
                1.0,  # w
                0.0,  # x
                0.0,  # y
                0.0   # z
            ],
            "uniform_scale": 1.0,
            "motion_type": "STATIC",
            "translation_origin": "COM"
        }
        object_instances.append(obstacle)
        
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
        num_obstacles = 12
        modified_data = regenerate_obstacles(base_file_path, num_obstacles)

        # 生成随机文件名
        random_filename = f"racing_{i+1}.scene_instance.json"
        output_file_path = os.path.join(output_dir, random_filename)

        # 将更新后的 JSON 数据写入新文件
        with open(output_file_path, 'w') as file:
            json.dump(modified_data, file, indent=4)

        print(f"文件 '{random_filename}' 已成功生成并添加到 '{output_dir}' 目录中。")


base_file_path = 'datasets/spy_datasets/configs/racing_straight_random/racing_1.scene_instance.json'
output_directory = 'datasets/spy_datasets/configs/racing_straight_random'
generate_random_json_files(base_file_path, output_directory, num_files=20000)