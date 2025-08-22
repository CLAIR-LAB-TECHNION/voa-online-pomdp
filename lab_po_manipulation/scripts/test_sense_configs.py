from matplotlib import pyplot as plt

from lab_po_manipulation.manipulation_prm import ManipulationPRM
from lab_po_manipulation.poman_motion_planner import POManMotionPlanner
from lab_ur_stack.camera.realsense_camera import RealsenseCamera
from lab_ur_stack.manipulation.manipulation_controller_2fg import ManipulationController2FG
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1

mp = POManMotionPlanner()
mp.add_mobile_obstacle_by_number(1)
mp.add_mobile_obstacle_by_number(2)
mp.add_mobile_obstacle_by_number(3)
mp.add_mobile_obstacle_by_number(4)
mp.visualize()
gt = GeometryAndTransforms.from_motion_planner(mp)

camera = RealsenseCamera()

r1_controller = ManipulationController2FG(ur5e_1["ip"], ur5e_1["name"], mp, gt)
r1_controller.plan_and_move_home()

prm = ManipulationPRM.load_roadmap(mp, "../env_configurations/prms/prm_all_obs.npy")

interesting_configs = prm.get_all_interesting_configs_as_dict()

curr_vid = prm.home_cfg_idx
for config_name, config_ids in interesting_configs.items():
    if not config_name.startswith("sense"):
        continue

    path = prm.find_path_by_vertices_id(curr_vid, config_ids[0])
    r1_controller.move_path(path)

    rgb, depth = camera.get_frame_rgb()
    plt.imshow(rgb)
    plt.show()

    curr_vid = config_ids[0]

