from torch.utils.data import Dataset
import glob
import os
from typing import List, Union
from dataclasses import dataclass
import habitat_sim
import json
import numpy as np
import random
from utils.common import *
import argparse
from utils.common import std_to_habitat
import torch as th
# --------------------examples---------------------
# scene json
scene_example = {
    "stage_instance": {
        "template_name": "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    },
    # "collision_asset":"data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    "default_lighting": "data/scene_datasets/habitat-test-scenes/default_lighting.glb",
    "object_instances": [
        {
            "template_name": "data/objects/example_object.glb",
            "translation": [1.0,
                            0.0,
                            0.0],
            "rotation": [0.0,
                         0.0,
                         0.0,
                         1.0],
            "uniform_scale": 1.0,
        },
    ],
    "articulated_object_instances": [
        {
            "template_name": "fridge",
            "translation_origin": "COM",
            "fixed_base": True,
            "translation": [
                -2.1782121658325195,
                0.9755649566650391,
                3.2299728393554688
            ],
            "rotation": [
                1,
                0,
                0,
                0
            ],
            "motion_type": "DYNAMIC"
        },
    ],
    # "navmesh_instance": "empty_stage_navmesh",
    "default_lighting": "",
    "user_custom": {
        "bound": [
            [1, 1, 1],
            [10, 10, 10]
        ]
    }
}

empty_scene = {
    "stage_instance": {
        "template_name": "data/scene_datasets/habitat-test-scenes/empty_stage.glb",
    },
    "default_lighting": "",
    "object_instances": [],
    # "navmesh_instance": "empty_stage_navmesh"
}


# ----------------------end------------------------

@dataclass
class SceneGeneratorSetting:
    object_dense: float
    object_scale: float
    object_margin: tuple
    light_random: bool
    stage: str
    object_set: str


class SceneGenerator:
    def __init__(self,
                 path: str,
                 num: int,
                 name: str,
                 setting: SceneGeneratorSetting
                 ) -> None:
        self.path = path
        self.num = num
        self.name = name
        self.setting = setting

        self.object_root_path = os.path.join(path, f"configs/{setting.object_set}") # debug
        self.light_root_path = os.path.join(path, "configs/lights")
        self.stage_root_path = os.path.join(path, "configs/stages")
        self.navmesh_root_path = os.path.join(path, "navmeshes")
        self.save_path = os.path.join(path, f"configs/{name}")

        self.objects_path = self._get_all_chirldren_path(self.object_root_path)
        self.lights_path = self._get_all_chirldren_path(self.light_root_path)
        self.stages_path = self._get_all_chirldren_path(self.stage_root_path)
        self.navmesh_path = self._get_all_chirldren_path(self.navmesh_root_path)

        self._write_scene_dir_in_summary(name)

    def generate(self):
        return self._create_scene_json()
        # self._create_navmesh()

    def _write_scene_dir_in_summary(self, name):
        # load json file
        summary_path = os.path.join(self.path, "spy_datasets.scene_dataset_config.json")
        with open(summary_path, "r") as file:
            summary = json.load(file)
        # write
        if f"configs/{name}" not in summary["scene_instances"]["paths"][".json"]:
            summary["scene_instances"]["paths"][".json"].append(f"configs/{name}")
        with open(summary_path, "w") as file:
            json.dump(summary, file, indent=4)

    def _create_scene_json(self):
        scene_template_cache = None
        scene_save_paths = []
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        for id in range(self.num):
            scene_save_path = f"{self.save_path}/{self.name}_{id}.scene_instance.json"
            scene_json = scene_example.copy()
            scene_save_paths.append(scene_save_path)

            # stage
            scene_json["stage_instance"]["template_name"] = self._get_stage()

            # update stage range
            if scene_template_cache is None or scene_json["stage_instance"]["template_name"] != scene_template_cache:
                scene_template_cache = scene_json["stage_instance"]["template_name"]
                scene_json["user_custom"]["bound"] = self._get_stage_bound(scene_json["stage_instance"]["template_name"])

            # lighting
            if self.setting.light_random:
                pass
            else:
                if self.setting.stage == "garage":
                    scene_json["default_lighting"] = "lighting/garage_v1_0"
                else:
                    scene_json["default_lighting"] = "default"

            # objects

            scene_json["object_instances"] = self._create_objects(self.setting.object_dense,
                                                                  self.setting.object_scale,
                                                                  scene_json["user_custom"]["bound"])

            # articulated objects
            scene_json["articulated_object_instances"] = []
            scene_json["navmesh_instance"] =  "empty_stage_navmesh"

            # save
            self._save_json_file(scene_save_path, scene_json)
        print(f"{self.num} Files has been save in {os.getcwd()}/{self.save_path}" )
        return  scene_save_paths

    def _get_stage_bound(self, stage_name, debug_stage=""):
        """_summary_
            get stage range.
        Args:
            stage_name (_type_): _description_

        Returns:
            _type_: _description_
        """
        # scene_json = empty_scene.copy()
        # scene_json["stage_instance"]["template_name"] = stage_name
        # scene_save_path = f"{self.save_path}/temp.scene_instance.json"
        # self._save_json_file(scene_save_path, scene_json)
        # habitat_sim_cfg = habitat_sim.SimulatorConfiguration()
        # habitat_sim_cfg.scene_dataset_config_file = "datasets/spy_datasets/replicaCAD.scene_dataset_config.json"
        # habitat_sim_cfg.enable_physics = False
        # agent_cfg = habitat_sim.agent.AgentConfiguration()
        # sim = habitat_sim.Simulator(habitat_sim.Configuration(habitat_sim_cfg,[agent_cfg]))
        # bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        # sim.close()
        # return [[bb.min[0], bb.min[1], bb.min[2]], [bb.max[0], bb.max[1], bb.max[2]]]

        # return [[-7,0, -19],[7,5.5,1]]  # real
        # return ((-7, 7), (0, 5.5), (-18, 1))  # ((min_x, max_x), (min_y, max_y), (min_z, max_z))
        # debug
        return [[0, -7, 0], [18, 7, 5.]]

    def _get_stage(self):
        """_summary_
            get stage name.
        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if self.setting.stage == "random":
            pass
        elif self.setting.stage == "garage":
            return "stages/garage_v1"
        elif self.setting.stage == "box":
            return "stages/box_v1"
        elif self.setting.stage == "random":
            pass
        else:
            raise ValueError("stage setting error")

    def _create_objects(self,
                        density: float,
                        scale_randomness: float,
                        bounds: List[List]):
        """_summary_
            create random objects json in scenes.
        Args:
            density (float): _description_
            scale_randomness (float): _description_
            bounds (List[np.ndarray]): _description_

        Returns:
            _type_: _description_
        """
        # bounds = np.array(bounds)
        bounds = std_to_habitat(th.tensor(bounds), None)[0]
        margin = self.setting.object_margin
        bounds[:, 2] += np.array([-margin[0][0], margin[1][0]])
        bounds[:, 0] += np.array([-margin[0][1], margin[1][1]])
        bounds[:, 1] += np.array([margin[0][2], -margin[1][2]])

        # Determine the number of points to generate based on density and volume of bounds
        std_density_factor = 1
        volume = np.prod(np.abs(bounds[1,:] - bounds[0, :]))
        num_points = int(volume * density * std_density_factor)

        # Generate random positions within bounds
        positions = [np.random.uniform(bounds[0,:], bounds[1,:]) for _ in range(num_points)]

        # Generate random orientations (quaternions)
        orientations = []
        for _ in range(num_points):
            u1, u2, u3 = np.random.random(3)
            quat = np.array([
                np.sqrt(u1) * np.cos(2 * np.pi * u3),
                np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                np.sqrt(u1) * np.sin(2 * np.pi * u3),

            ])
            orientations.append(quat)

        # Generate sizes based on size_randomness
        max_scale = 2
        sizes = [1 + scale_randomness * np.random.uniform(0, max_scale - 1) for _ in range(num_points)]

        ids = [random.randint(0, len(self.objects_path) - 1) for _ in range(num_points)]

        object_instances = []

        for i in range(num_points):
            object_instance = scene_example["object_instances"][0].copy()
            object_instance["template_name"] = self.objects_path[ids[i]]
            object_instance["translation"] = positions[i].tolist()
            object_instance["rotation"] = orientations[i].tolist()
            object_instance["uniform_scale"] = sizes[i]
            object_instance["motion_type"] = "STATIC"
            object_instance["translation_origin"]="COM"
            object_instances.append(object_instance)

        return object_instances

    def _get_all_chirldren_path(self, root_path):
        file_paths = []
        for root, directories, files in os.walk(root_path):
            for filename in files:
                file_paths.append(str.split(filename, ".")[0])
        return file_paths

    def _save_json_file(self, file_path, data):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def _create_navmesh(self):
        pass


class StageGenerator:
    def __init__(self) -> None:
        pass


class LightGenerator:
    def __init__(self) -> None:
        pass


class ObjectGenerator:
    def __init__(self) -> None:
        pass


class ChildrenPathDataset(Dataset):
    def __init__(self, root_path, type="glb", semantic=False):
        """
        Args:
            strings (list): 一个包含字符串的列表.
        """
        self.root_path = root_path
        self.type = type

        self.paths = self._load_scene_path(semantic=semantic)

    def _load_scene_path(self, semantic=False):
        if "hm3d" in self.root_path.lower():
            # key = "*.basis.glb" if not semantic else "*.semantic.glb"
            key =  "*.semantic.glb"
        elif "mp3d" in self.root_path.lower():
            key = "*_semantic.ply"
        elif self.type == "json":
            key = "*.scene_instance.json"

        glb_files = []
        for root, dirs, files in os.walk(self.root_path):
            file_path = glob.glob(os.path.join(root, key))
            glb_files.extend(file_path)

        if not semantic:
            if "hm3d" in self.root_path.lower():
                glb_files = [glb_file[:-13]+glb_file[-4:] for glb_file in glb_files]
            elif "mp3d" in self.root_path.lower():
                glb_files = [glb_file[:-13]+".glb" for glb_file in glb_files]

        return glb_files

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, indice):
        """
        返回索引为 indice 的字符串.

        Args:
            indice (int): 数据的索引.

        Returns:
            string (str): 索引为 indice 的字符串.
        """
        return self.paths[indice]


def inverse_axes(path):
    pass


def parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", "-g", type=int, default=1, help="generate scenes")
    parser.add_argument("--render", "-r", type=int, default=0, help="render scenes")
    parser.add_argument("--quantity", "-q", type=int, default=1, help="generated quantity ")
    parser.add_argument("--name", "-n", type=str, default="garage_simple", help="name")
    parser.add_argument("--density", "-d", type=float, default=0.25, help="obstacle density")

    return parser


if __name__ == "__main__":
    args = parsers().parse_args()
    g = SceneGenerator(
        path="datasets/spy_datasets",
        num=args.quantity,
        name=args.name,
        setting=SceneGeneratorSetting(
            object_dense=args.density,
            object_scale=0,
            object_margin=np.array([[3,0,0],[7,0,1]]),  #  camera coordinate [[back, right, down],[front, left, up]]
            # object_margin=(0, 0, 0),
            light_random=False,
            stage="garage",
            object_set="objects2"
        )
    )
    if args.generate:
        scene_save_paths = g.generate()
    if args.render:
        os.system(f"python /home/lfx-desktop/files/habitat-sim/examples/viewer.py \
        --dataset datasets/spy_datasets/spy_datasets.scene_dataset_config.json \
        --scene {g.name}_0  --disable-physics")

        # --dataset datasets/spy_datasets/spy_datasets.scene_dataset_config.json \
        # --scene garage_simple_0  --disable-physics"

# f"python /home/suncc/habitat-sim/examples/viewer.py --dataset datasets/spy_datasets/spy_datasets.scene_dataset_config.json         --scene circle_debug  --disable-physics"
# f"python /home/lfx-desktop/files/habitat-sim/examples/viewer.py --dataset datasets/hssd-hab/hssd-hab.scene_dataset_config.json --scene 102343992 --disable-physics"
# python /home/suncc/habitat-sim/examples/viewer.py --dataset datasets/spy_datasets/spy_datasets.scene_dataset_config.json         --scene cross_and_avoid_debug  --disable-physics