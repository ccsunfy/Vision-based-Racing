import json

def read_targets_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    targets = []
    for obj in data.get("object_instances", []):
        if obj.get("template_name") == "red_circle":
            translation = obj.get("translation", [])
            target = [-translation[2],translation[0],translation[1]]
            if target not in targets:
                targets.append(target)
    
    return targets

if __name__ == "__main__":
    file_path = 'datasets/spy_datasets/configs/avoid_crosscircle_917/circle_avoid_debug.scene_instance.json'
    targets = read_targets_from_json(file_path)
    print("Targets:", targets)