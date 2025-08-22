from dataclasses import dataclass
from typing import Sequence, Optional
import json
from pathlib import Path

import numpy as np


@dataclass
class ItemPosition:
    name: str
    position: Sequence[float]
    # a grasp is defined by initial offset from position and rotation of the end effector around forward axis
    grasps_offset_rz: Sequence[tuple[Sequence[float], float]]

@dataclass
class ManipulatedObject:
    name: str
    grasp_width: Optional[float] = None

@dataclass
class MobileObstacle:
    name: str
    size: Sequence[float]
    initial_position: Sequence[float]

@dataclass
class SensingConfig:
    name: str
    robot_config: Sequence[float]
    visible_positions: Sequence[str]
    detection_probability: float


def save_grasps_to_json(position_name: str, grasps: list[tuple[list[float], float]],
                        output_dir: str = "grasps"):
    dir_path = Path(__file__).parent / output_dir
    dir_path.mkdir(parents=True, exist_ok=True)
    filepath = dir_path/ f"{position_name}_grasps.json"

    data = {
        "position_name": position_name,
        "grasps": [{"offset": list(g[0]), "rz": g[1]} for g in grasps]
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_grasps_from_json(position_name: str,
                          input_dir: str = "grasps")\
        -> list[tuple[list[float], float]]:

    dir_path = Path(__file__).parent / input_dir
    filepath = dir_path/ f"{position_name}_grasps.json"

    # handle failure to load file
    if not filepath.exists():
        print(f"\033[93m Warnning: File {filepath} does not exist \033[0m")
        return []
    with open(filepath) as f:
        data = json.load(f)
        return [(g["offset"], g["rz"]) for g in data["grasps"]]


# # # positions in the workspace where objects can be placed
positions = [
    ItemPosition("top_shelf_left", (-0.796, -0.189, 0.523), []),
    ItemPosition("top_shelf_right", (-0.796, 0.084, 0.523), []),
    ItemPosition("middle_shelf", (-0.781, -0.055, 0.283), []),
    ItemPosition("middle_shelf_left", (-0.676, -0.232, 0.283), []),
    ItemPosition("bottom_left", (-0.462, -0.214, 0.043), []),
    ItemPosition("bottom_right", (-0.495, 0.133, 0.043), []),
    ItemPosition("workspace_can", (0.55, 0.2, 0.043), []),
    ItemPosition("workspace_red_cup", (0.35, -0.3, 0.043), []),
    ItemPosition("workspace_blue_cup", (0.55, -0.3, 0.043), []),
]
for pos in positions:
    pos.grasps_offset_rz = load_grasps_from_json(pos.name)
positions_dict = {pos.name: pos for pos in positions}


# # # objects that can be manipulated
objects = [
    ManipulatedObject("red_cup"),
    ManipulatedObject("blue_cup"),
    ManipulatedObject("soda_can", grasp_width=57),
]
objects_dict = {obj.name: obj for obj in objects}


# # # obstacles that can be moved
mobile_obstacles = [
    MobileObstacle("mobile_obstacle_1", (0.12, 0.12, 0.48), initial_position=(-0.16, -0.36, 0.24)),
    MobileObstacle("mobile_obstacle_2", (0.12, 0.12, 0.24), initial_position=(-0.325, -0.04, 0.36)),
    MobileObstacle("mobile_obstacle_3", (0.24, 0.12, 0.48), initial_position=(-0.745, -0.36, 0.24)),
    MobileObstacle("mobile_obstacle_4", (0.32, 0.08, 0.16), initial_position=(-0.785, -0.1, 0.80)),
]
# inflate all obstacles by 0.5 cm
for mob in mobile_obstacles:
    mob.size = [s + 0.005 for s in mob.size]
mobile_obstacles_dict = {mob.name: mob for mob in mobile_obstacles}


# # # Sensing configurations
sensing_configs = [
    SensingConfig("sense_1", [-1.5728, -2.1127, -0.4491, 0.1095, -0.5790, -0.7206],
                  visible_positions=["top_shelf_left", "top_shelf_right"],
                  detection_probability=0.98),
    SensingConfig("sense_2", [-0.0549, -3.2338, 2.3415, -1.6529, -1.4437, -0.3197],
                  visible_positions=["middle_shelf_left", "bottom_left", "middle_shelf"],
                  detection_probability=0.95),
    SensingConfig("sense_3", [2.7262, -0.4723, -2.6765, -0.8842, 1.9666, 0.2401],
                    visible_positions=["bottom_right"],
                    detection_probability=0.999),
    SensingConfig("sense_4", [3.5249, -0.9331, -1.9401, -0.8049, 1.8051, 0.7575],
                    visible_positions=["middle_shelf", "middle_shelf_left", "top_shelf_left"],
                    detection_probability=0.95),
    SensingConfig("sense_5", [0.9672, -1.2414, 1.1472, -1.1855, -2.0487, 1.0435],
                    visible_positions=["bottom_left"],
                    detection_probability=0.999),
]
sensing_configs_dict = {sc.name: sc for sc in sensing_configs}


# # # other configs for manipulation
# pouring and trash configs/poses
pre_pour_config_red = [0.052, -1.856, -1.578, -2.088, 0.999, 2.659]
pour_pose_red = [0.411, -0.251, 0.212, 1.050, 2.335, 0.785]
pre_pour_config_blue = [0.273, -1.846, -1.868, -1.496, 0.539, 2.132]
pour_pose_blue = [0.533, -0.212, 0.212, -0.167, 2.949, 1.013]
trash_config = [-1.055, -2.258, -2.011, -0.384, 0, 1.6]
""" usage:
path = prm.find_path(r1_controller.getActualQ(), pre_pour_config_red)
r1_controller.move_path(path)
pre_pour_pose = r1_controller.getActualTCPPose()
r1_controller.moveL(pour_pose_red, speed=0.2, acceleration=0.1)
time.sleep(1)
r1_controller.moveL(pre_pour_pose, speed=0.2, acceleration=0.1)
"""

helper_idle_config = [-np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
