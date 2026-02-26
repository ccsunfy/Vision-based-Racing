import json
import random
import os

def regenerate_obstacles(file_path, num_obstacles=30, avoid_areas=None):
    if avoid_areas is None:
        avoid_areas = [
            (-2, -2), (-10, 2), (-10, -2), (-2, 2),
            (-4,-2),(-2,0),(-4,2),(-8,2),(-10,0),(-8,-2)
        ]

    def is_in_avoid_area(x, z, areas, margin=0.8):
        for area in areas:
            if area[0] - margin <= x <= area[0] + margin and area[1] - margin <= z <= area[1] + margin:
                return True
        return False

    # 读取现有的 JSON 文件
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 获取现有的 object_instances
    object_instances = data["object_instances"]

    # 清空现有的障碍物
    object_instances = [obj for obj in object_instances if obj["template_name"] != "0.1_0.1_2cylinder"]

    # 随机生成障碍物的位置和旋转
    for _ in range(num_obstacles):
        while True:
            x = random.uniform(-4., 4.)
            z = random.uniform(-10., -1.)
            if not is_in_avoid_area(x, z, avoid_areas):
                break

        obstacle = {
            "template_name": "0.1_0.1_2cylinder",
            "translation": [x, 0.0, z],
            "rotation": [1.0, 0.0, 0.0, 0.0],
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

    for i in range(num_files):
        # 为每个文件生成不同的障碍物数量
        num_obstacles = 20
        modified_data = regenerate_obstacles(base_file_path, num_obstacles)

        # 生成随机文件名
        random_filename = f"racing_{i+1}.scene_instance.json"
        output_file_path = os.path.join(output_dir, random_filename)

        # 将更新后的 JSON 数据写入新文件
        with open(output_file_path, 'w') as file:
            json.dump(modified_data, file, indent=4)

        print(f"文件 '{random_filename}' 已成功生成并添加到 '{output_dir}' 目录中。")

# example usage
base_file_path = 'datasets/spy_datasets/configs/demo3_ellipse_no_ob/racing8.scene_instance.json'
output_directory = 'datasets/spy_datasets/configs/demo3_ellipse_ob1'

generate_random_json_files(base_file_path, output_directory, num_files=1)