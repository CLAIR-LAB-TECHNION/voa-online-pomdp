from lab_po_manipulation.env_configurations.objects_and_positions import helper_idle_config
from lab_po_manipulation.obstacles_moving import move_obstacle_1, move_obstacle_2, move_obstacle_3, move_obstacle_4
from lab_po_manipulation.poman_motion_planner import POManMotionPlanner
from lab_ur_stack.manipulation.manipulation_controller_vg import ManipulationControllerVG
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_2


mp = POManMotionPlanner()
mp.add_mobile_obstacle_by_number(1)
mp.add_mobile_obstacle_by_number(2)
mp.add_mobile_obstacle_by_number(3)
mp.add_mobile_obstacle_by_number(4)
gt = GeometryAndTransforms(mp)
r2_controller = ManipulationControllerVG(ur5e_2["ip"], ur5e_2["name"], mp, gt)
r2_controller.speed, r2_controller.acceleration = 0.5, .5

r2_controller.plan_and_move_home()
move_obstacle_1(r2_controller)
