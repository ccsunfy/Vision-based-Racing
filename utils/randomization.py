import torch as th
from .type import Uniform, Normal
from typing import Union, Optional, Dict
from .maths import Quaternion
from abc import abstractmethod


class StateRandomizer:
    def __init__(self,
                 position,
                 orientation,  # euler angle
                 velocity,
                 angular_velocity,
                 seed: int = 42,
                 is_collision_func: Optional[callable] = None,
                 scene_id: Optional[int] = None,
                 device: th.device = th.device("cpu")
                 ):

        self.position = position
        self.orientation = orientation
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.is_collision_func = is_collision_func
        self.device = device
        self.scene_id = scene_id
        # self.set_seed(seed)

    @abstractmethod
    def generate(self, num) -> tuple:
        pass

    def safe_generate(self, num=1):
        raw_pos, raw_ori, raw_vel, raw_ang_vel = self.generate(num)
        position = raw_pos
        orientation = raw_ori
        velocity = raw_vel
        angular_velocity = raw_ang_vel
        # position = (2 * th.rand(num, 3) - 1) * self.position.half + self.position.mean
        # orientation = (2 * th.rand(num, 3) - 1) * self.orientation.half + self.orientation.mean
        # velocity = (2 * th.rand(num, 3) - 1) * self.velocity.half + self.velocity.mean
        # angular_velocity = (2 * th.rand(num, 3) - 1) * self.angular_velocity.half + self.angular_velocity.mean

        if self.is_collision_func is not None:
            is_collision = self.is_collision_func(std_positions=position, scene_id=self.scene_id)
            while True:
                if not is_collision.any():
                    break
                raw_pos, raw_ori, raw_vel, raw_ang_vel = self.generate(is_collision.sum())
                position[is_collision, :] = raw_pos
                orientation[is_collision, :] = raw_ori
                velocity[is_collision, :] = raw_vel
                angular_velocity[is_collision, :] = raw_ang_vel
                # position[is_collision, :] = (2 * th.rand(is_collision.sum(), 3) - 1) * self.position.half + self.position.mean
                # orientation[is_collision, :] = (2 * th.rand(is_collision.sum(), 3) - 1) * self.orientation.half + self.orientation.mean
                # velocity[is_collision, :] = (2 * th.rand(is_collision.sum(), 3) - 1) * self.velocity.half + self.velocity.mean
                # angular_velocity[is_collision, :] = (2 * th.rand(is_collision.sum(), 3) - 1) * self.angular_velocity.half + self.angular_velocity.mean
                is_collision = self.is_collision_func(std_positions=position, scene_id=self.scene_id)

        orientation = Quaternion.from_euler(*orientation.T).toTensor().T
        return position.to(self.device), orientation.to(self.device), velocity.to(self.device), angular_velocity.to(self.device)

    def set_seed(self, seed=42):
        th.manual_seed(seed)

    def to(self, device):
        self.device = device
        return self


class UniformStateRandomizer(StateRandomizer):
    def __init__(self,
                 position={"mean": [0., 0., 0.], "half": [0., 0., 0.]},
                 orientation={"mean": [0., 0., 0.], "half": [0., 0., 0.]},  # euler angle
                 velocity={"mean": [0., 0., 0.], "half": [0., 0., 0.]},
                 angular_velocity={"mean": [0., 0., 0.], "half": [0., 0., 0.]},
                 seed: int = 42,
                 is_collision_func: Optional[callable] = None,
                 scene_id: Optional[int] = None,
                 device: th.device = th.device("cpu")
                 ):
        super().__init__(
            position=position,
            orientation=orientation,
            velocity=velocity,
            angular_velocity=angular_velocity,
            seed=seed,
            is_collision_func=is_collision_func,
            scene_id=scene_id,
            device=device
        )

        self.position = Uniform(**position)
        self.orientation = Uniform(**orientation)
        self.velocity = Uniform(**velocity)
        self.angular_velocity = Uniform(**angular_velocity)

    def generate(self, num) -> tuple:
        position = (2 * th.rand(num, 3) - 1) * self.position.half + self.position.mean
        orientation = (2 * th.rand(num, 3) - 1) * self.orientation.half + self.orientation.mean
        velocity = (2 * th.rand(num, 3) - 1) * self.velocity.half + self.velocity.mean
        angular_velocity = (2 * th.rand(num, 3) - 1) * self.angular_velocity.half + self.angular_velocity.mean
        return position, orientation, velocity, angular_velocity


class NormalStateRandomizer(StateRandomizer):
    def __init__(
            self,
            position={"mean": [0., 0., 0.], "std": [0., 0., 0.]},
            orientation={"mean": [0., 0., 0.], "std": [0., 0., 0.]},  # euler angle
            velocity={"mean": [0., 0., 0.], "std": [0., 0., 0.]},
            angular_velocity={"mean": [0., 0., 0.], "std": [0., 0., 0.]},
            seed: int = 42,
            is_collision_func: Optional[callable] = None,
            scene_id: Optional[int] = None,
            device: th.device = th.device("cpu")
    ):
        super().__init__(
            position=position,
            orientation=orientation,
            velocity=velocity,
            angular_velocity=angular_velocity,
            seed=seed,
            is_collision_func=is_collision_func,
            scene_id=scene_id,
            device=device
        )

        self.position = Normal(**position)

    def generate(self, num) -> tuple:
        position = th.randn(num, 3) * self.position.std + self.position.mean
        orientation = th.randn(num, 3) * self.orientation.std + self.orientation.mean
        velocity = th.randn(num, 3) * self.velocity.std + self.velocity.mean
        angular_velocity = th.randn(num, 3) * self.angular_velocity.std + self.angular_velocity.mean
        return position, orientation, velocity, angular_velocity


class UnionRandomizer:
    Randomizer_alias = {
        "Uniform": UniformStateRandomizer,
        "Normal": NormalStateRandomizer
    }

    def __init__(
            self,
            randomizers_kwargs: list,
            device,
            is_collision_func=None,
            scene_id=0,
    ):
        self.randomizers = []
        for randomizers in randomizers_kwargs:
            self.randomizers.append(
                self.Randomizer_alias[randomizers["class"]](
                    device=device,
                    is_collision_func=is_collision_func,
                    scene_id=scene_id,
                    **randomizers["kwargs"]
                )
            )

    def __Len__(self):
        return len(self.randomizers)

    def to(self, device):
        for randomizer in self.randomizers:
            randomizer.to(device)

    def generate(self, num) -> tuple:
        position, orientation, velocity, angular_velocity = [], [], [], []
        for randomizer in self.randomizers:
            pos, ori, vel, ang_vel = randomizer.generate(num)
            position.append(pos)
            orientation.append(ori)
            velocity.append(vel)
            angular_velocity.append(ang_vel)

        position, orientation, velocity, angular_velocity = th.stack(position), th.stack(orientation), th.stack(velocity), th.stack(angular_velocity)
        select_randomizer_index = th.randint(0, len(self.randomizers), (num,))
        row = th.arange(num)
        return position[row, select_randomizer_index], orientation[row, select_randomizer_index], velocity[row, select_randomizer_index], angular_velocity[row, select_randomizer_index]

    def safe_generate(self, num):
        position, orientation, velocity, angular_velocity = [], [], [], []
        for randomizer in self.randomizers:
            pos, ori, vel, ang_vel = randomizer.safe_generate(num)
            position.append(pos)
            orientation.append(ori)
            velocity.append(vel)
            angular_velocity.append(ang_vel)

        position, orientation, velocity, angular_velocity = th.stack(position), th.stack(orientation), th.stack(velocity), th.stack(angular_velocity)
        select_randomizer_index = th.randint(0, len(self.randomizers), (num,))
        row = th.arange(num)
        return position[select_randomizer_index, row, :], orientation[select_randomizer_index, row, :], velocity[select_randomizer_index, row, :], angular_velocity[select_randomizer_index, row, :]