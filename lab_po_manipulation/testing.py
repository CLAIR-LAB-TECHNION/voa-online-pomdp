from lab_po_manipulation.env_configurations.objects_and_positions import (positions, objects_dict,
                                                                          mobile_obstacles_dict, mobile_obstacles,
                                                                          positions_dict, helper_idle_config)
from lab_po_manipulation.poman_motion_planner import POManMotionPlanner
from lab_po_manipulation.utils import plan_on_prm_and_pick_up, plan_on_prm_and_put_down
from lab_ur_stack.manipulation.manipulation_controller_vg import ManipulationControllerVG
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.manipulation.manipulation_controller_2fg import ManipulationController2FG
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from klampt.math import so3, se3
from lab_po_manipulation.manipulation_prm import ManipulationPRM


mp = POManMotionPlanner()
prm = ManipulationPRM.load_roadmap(mp, "roadmap_ur5e_1_with_grasps.npy")
gt = GeometryAndTransforms.from_motion_planner(mp)

for mob_obs in mobile_obstacles:
    obj = {"name": mob_obs.name, "scale": mob_obs.size, "coordinates": mob_obs.initial_position,
           "color": [0.8, 0.3, 0.3, 0.75], "geometry_file": "../lab_ur_stack/motion_planning/objects/cube.off",
           "angle": so3.identity()}
    mp.add_object_to_world(mob_obs.name, obj)
mp.visualize()
#
r1_controller = ManipulationController2FG(ur5e_1["ip"], ur5e_1["name"], mp, gt)
r2_controller = ManipulationControllerVG(ur5e_2["ip"], ur5e_2["name"], mp, gt)
r1_controller.speed, r1_controller.acceleration = 0.8, .75
r2_controller.speed, r2_controller.acceleration = 0.75, .75

try:
    r1_controller.plan_and_move_home()
except:
    r1_controller.moveL_relative([0, 0, 0.1], speed=0.05, acceleration=0.1)
    r1_controller.plan_and_move_home()

r2_controller.plan_and_moveJ(helper_idle_config)

cur_vid = prm.home_cfg_idx
cur_vid = plan_on_prm_and_pick_up(r1_controller, prm, positions_dict["top_shelf_right"], cur_vid, grasp_width=57)
cur_vid = plan_on_prm_and_put_down(r1_controller, prm, positions_dict["workspace_can"], cur_vid)
cur_vid = plan_on_prm_and_pick_up(r1_controller, prm, positions_dict["bottom_left"], cur_vid)
cur_vid = plan_on_prm_and_put_down(r1_controller, prm, positions_dict["workspace_red_cup"], cur_vid)
cur_vid = plan_on_prm_and_pick_up(r1_controller, prm, positions_dict["top_shelf_left"], cur_vid)
cur_vid = plan_on_prm_and_put_down(r1_controller, prm, positions_dict["workspace_blue_cup"], cur_vid)

path = prm.find_path_by_vertices_id(cur_vid, prm.home_cfg_idx)
r1_controller.move_path(path)