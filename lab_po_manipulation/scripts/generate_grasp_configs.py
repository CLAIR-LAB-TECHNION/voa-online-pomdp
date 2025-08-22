from copy import copy
import numpy as np
import typer
from klampt.math import so3

from lab_po_manipulation.env_configurations.objects_and_positions import positions_dict, objects_dict, \
    save_grasps_to_json, load_grasps_from_json, mobile_obstacles_dict
from lab_po_manipulation.poman_motion_planner import POManMotionPlanner
from lab_po_manipulation.prm import default_joint_limits_low, default_joint_limits_high
from lab_ur_stack.manipulation.manipulation_controller_2fg import ManipulationController2FG
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1
import json
from pathlib import Path


app = typer.Typer()



def test_execution_grasp_config(position_name, offset, ee_rz,
                                robot_controller: ManipulationController2FG):
    position = positions_dict[position_name]
    pos = list(position.position)
    robot_controller.pick_up_at_angle(pos, offset, ee_rz)
    robot_controller.put_down_at_angle(pos, offset, ee_rz)


if __name__ == "__main__":
    mp = POManMotionPlanner()
    mp.add_mobile_obstacle_by_name("mobile_obstacle_1")
    mp.add_mobile_obstacle_by_name("mobile_obstacle_2")
    mp.add_mobile_obstacle_by_name("mobile_obstacle_3")
    mp.add_mobile_obstacle_by_name("mobile_obstacle_4")

    gt = GeometryAndTransforms.from_motion_planner(mp)
    r1_controller = ManipulationController2FG(ur5e_1["ip"], ur5e_1["name"], mp, gt)
    r1_controller.speed, r1_controller.acceleration = 0.75, .75

    grasp_configs = [
        # ([-0.09, 0.014, 0.01], 0),
        ([-0.08, 0.00, 0.01], 0)
    ]

    for offset, ee_rz in grasp_configs:
        test_execution_grasp_config("workspace_can", offset, ee_rz, r1_controller)

    # save_grasps_to_json("", grasp_configs)
