import json
import random
import os

def regenerate_obstacles(file_path, num_obstacles=30, avoid_areas=None):
    if avoid_areas is None:
        avoid_areas = [
            (-2, 2), (-6, 2), (-6, -2), (-2, -2),  # Initial points
            (-4, -4), (0, -8), (4, -5), (1, -1)  # Gates
        ]

    def is_in_avoid_area(x, z, areas, margin=0.5):
        for area in areas:
            if len(area) == 2:
                if area[0] - margin <= x <= area[0] + margin and area[1] - margin <= z <= area[1] + margin:
                    return True
            elif len(area) == 3:
                if area[0] - margin <= x <= area[0] + margin and area[2] - margin <= z <= area[2] + margin:
                    return True
        return False

    # 读取现有的 JSON 文件
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 获取现有的 object_instances
    object_instances = data["object_instances"]

    # 清空现有的障碍物
    object_instances = [obj for obj in object_instances if obj["template_name"] != "0.1_0.1_2cylinder"]

    # 定义四个象限的边界，集中在通往门的道路上
    quadrants = [
        (-4, 0, -8, -4),  # 左下象限
        (0, 4, -8, -4),   # 右下象限
        (-4, 0, -4, 0),   # 左上象限
        (0, 4, -4, 0)     # 右上象限
    ]

    # 在每个象限内均匀生成障碍物
    obstacles_per_quadrant = num_obstacles // 4
    for quadrant in quadrants:
        x_min, x_max, z_min, z_max = quadrant
        for _ in range(obstacles_per_quadrant):
            while True:
                x = random.uniform(x_min, x_max)
                z = random.uniform(z_min, z_max)
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
base_file_path = 'datasets/spy_datasets/configs/demo2_3Dcircle_no_ob/no_ob.scene_instance.json'
output_directory = 'datasets/spy_datasets/configs/demo2_3Dcircle_ob1'

generate_random_json_files(base_file_path, output_directory, num_files=1)