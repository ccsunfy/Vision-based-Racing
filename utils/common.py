import numpy as np
from typing import Optional, Tuple, List
from torch import Tensor
import torch as  th
from utils.maths import Quaternion


def obs_list2array(obs_dict:List, row:int, column:int):
    obs_indice = 0
    obs_array = []
    for i in range(column):
        obs_row = []
        for j in range(row):
            obs_row.append(obs_dict[obs_indice]["depth"])
            obs_indice += 1
        obs_array.append(np.hstack(obs_row))
    return np.vstack(obs_array)

def depth2rgb(image):
    max_distance = 5.
    image = image / max_distance 
    image[image > 1] = 1
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 2:
        image = np.stack([image, image, image, np.full(image.shape, 255,dtype=np.uint8)], axis=-1)
    return image


def rgba2rgb(image):
    if isinstance(image, List):
        return [rgba2rgb(img) for img in image]
    else:
        return image[:,:,:3]

def habitat_to_std(habitat_pos:Optional[np.ndarray]=None, habitat_ori:Optional[np.ndarray]=None, format="enu"):
    """_summary_
        axes transformation, from habitat-sim to std

    Args:
        habitat_pos (_type_): _description_
        habitat_ori (_type_): _description_
        format (str, optional): _description_. Defaults to "enu".

    Returns:
        _type_: _description_
    """
    # habitat_pos, habitat_ori = np.atleast_2d(habitat_pos), np.atleast_2d(habitat_ori)
    assert format in ["enu"]

    if habitat_pos is None:
        std_pos = None
    else:
        # assert habitat_pos.shape[1] == 3
        std_pos =  th.as_tensor(
            np.atleast_2d(habitat_pos) @ np.array([[0, -1, 0],
                                            [0, 0, 1],
                                            [-1, 0, 0]])
        , dtype=th.float32)
        # if len(habitat_pos.shape) == 1:
        #     std_pos = habitat_pos
            
    if habitat_ori is None:
        std_ori = None
    else:
        # assert habitat_ori.shape[1] == 4
        std_ori = th.from_numpy(
            np.atleast_2d(habitat_ori) @ np.array(
            [[1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
                [0, -1, 0, 0]]
        )
        )
    return std_pos, std_ori

def std_to_habitat( std_pos:Optional[Tensor]=None, std_ori:Optional[Tensor]=None, format="enu")\
        -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """_summary_
        axes transformation, from std to habitat-sim

    Args:
        std_pos (_type_): _description_
        std_ori (_type_): _description_
        format (str, optional): _description_. Defaults to "enu".

    Returns:
        _type_: _description_
    """
    assert format in ["enu"]

    # Q = Quaternion(
    #     R.from_euler("ZYX", [-90, 0, 90], degrees=True).as_quat()
    # ).inverse()
    # std_pos_as_quat = [Quaternion(np.r_[std_pos_i, 0]) for std_pos_i in std_pos]
    # hab_pos = np.array([(Q * p * Q.inverse()).imag for p in std_pos_as_quat])
    # std_ori_as_quat = [Quaternion(q) for q in std_ori]
    # hab_ori = np.array(
    #     [(Q * std_ori_as_quat_i).numpy() for std_ori_as_quat_i in std_ori_as_quat]
    # )
    if std_ori is None:
        hab_ori = None
    else:
        hab_ori = std_ori.clone().detach().cpu().numpy() @ np.array(
            [[1, 0, 0, 0],
                [0, 0, 0, -1],
                [0, -1, 0, 0],
                [0, 0, 1, 0]]
        )

    if std_pos is None:
        hab_pos = None
    else:

        if len(std_pos.shape) == 1:
            hab_pos = (std_pos.clone().detach().cpu().unsqueeze(0).numpy() @ np.array([[0, 0, -1],
                                            [-1, 0, 0],
                                            [0, 1, 0]])).squeeze()
        elif std_pos.shape[1] == 3:
            hab_pos = std_pos.clone().detach().cpu().numpy() @ np.array([[0, 0, -1],
                                                                         [-1, 0, 0],
                                                                         [0, 1, 0]])
        else:
            raise ValueError("std_pos shape error")

    return hab_pos, hab_ori