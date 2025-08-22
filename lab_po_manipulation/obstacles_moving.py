import numpy as np
from sympy.physics.units import acceleration

from lab_po_manipulation.env_configurations.objects_and_positions import mobile_obstacles_dict, helper_idle_config
from lab_ur_stack.manipulation.manipulation_controller_vg import ManipulationControllerVG



def move_obstacle_1(robot_controller: ManipulationControllerVG):
    obstacle_data = mobile_obstacles_dict["mobile_obstacle_1"]
    x,y,z = obstacle_data.initial_position
    robot_controller.pick_up(x, y, 0, 0.51)
    robot_controller.put_down(x-0.25, y-0.6, 0, 0.51)
    robot_controller.moveL_relative([0.1, 0.1, 0.02], speed=0.1)
    robot_controller.plan_and_moveJ(helper_idle_config)


def move_obstacle_2(robot_controller: ManipulationControllerVG):
    robot_controller.plan_and_moveJ([-1.9180, -2.2754, -0.2052, -2.0970, 1.6764, -1.2454])
    robot_controller.moveUntilContact(xd=[0, 0, -0.03, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
    robot_controller.moveL_relative([0, 0, -0.01], speed=0.1)
    robot_controller.grasp()
    robot_controller.moveL_relative([0, 0, 0.02], speed=0.05, acceleration=0.05)
    robot_controller.moveJ([-2.1029, -1.6270, -0.2055, -2.4398, 1.7708, -1.2492], acceleration=0.3)
    robot_controller.moveJ([-3.2223, -1.6669, -0.4591, -2.5828, 1.5657, -1.2057])
    robot_controller.moveJ([-3.9901, -1.6350, -1.3587, -1.6978, 1.5735, -1.6894])
    robot_controller.moveUntilContact(xd=[0, 0, -0.05, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
    robot_controller.release_grasp()
    robot_controller.moveL_relative([0, 0, 0.1], speed=0.1)
    robot_controller.plan_and_moveJ(helper_idle_config)


def move_obstacle_3(robot_controller: ManipulationControllerVG):
    robot_controller.plan_and_moveJ([1.2978, -1.8859, 1.5597, -2.7861, -1.1940, -0.7651])
    robot_controller.moveL_relative([0, -0.035, 0], speed=0.1, acceleration=0.1)
    robot_controller.grasp()
    robot_controller.moveL_relative([0, 0.12, 0.06], speed=0.1, acceleration=0.1)
    robot_controller.moveJ([-0.4379, -1.6530, 1.4768, -2.5453, -1.0443, -0.7711], acceleration=0.3)
    robot_controller.moveJ([-1.3565, -1.3222, 1.7949, -2.0230, -1.5285, -2.1419])
    robot_controller.moveUntilContact(xd=[0, 0, -0.05, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
    robot_controller.release_grasp()
    robot_controller.moveL_relative([0, 0, 0.1], speed=0.1)
    robot_controller.plan_and_moveJ(helper_idle_config)

def move_obstacle_4(robot_controller: ManipulationControllerVG):
    robot_controller.plan_and_moveJ([1.3666,-1.9827,1.4911,-1.1744,-1.5647,-1.0474])
    robot_controller.moveUntilContact(xd=[0, 0, -0.03, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
    robot_controller.moveL_relative([0, 0, -0.01], speed=0.1)
    robot_controller.grasp()
    robot_controller.moveL_relative([0, 0.08, 0.05], speed=0.05, acceleration=0.05)
    robot_controller.moveJ([-0.9476,-1.9498,1.4806,-1.1418,-1.4703,-1.7247], acceleration=0.3)
    robot_controller.moveUntilContact(xd=[0, 0, -0.05, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
    robot_controller.release_grasp()
    robot_controller.moveL_relative([0, 0, 0.15], speed=0.1)
    robot_controller.plan_and_moveJ(helper_idle_config)

