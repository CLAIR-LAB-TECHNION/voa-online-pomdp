import time

from lab_po_manipulation.env_configurations.objects_and_positions import ItemPosition, positions_dict
from lab_po_manipulation.manipulation_prm import ManipulationPRM
from lab_po_manipulation.poman_motion_planner import POManMotionPlanner
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms


mp = POManMotionPlanner()
# mp.add_mobile_obstacle_by_number(1)
mp.add_mobile_obstacle_by_number(2)
mp.add_mobile_obstacle_by_number(3)
mp.add_mobile_obstacle_by_number(4)
mp.visualize()

prm = ManipulationPRM.load_roadmap(mp, "../env_configurations/prms/prm_no_obs1_y_up.npy")
gt = GeometryAndTransforms.from_motion_planner(mp)


def plan_with_prm_and_animate(prm: ManipulationPRM, pick_position: ItemPosition, init_vertex_id=None):
    """
    returns current vertex id
    """
    if init_vertex_id is None:
        init_vertex_id = prm.home_cfg_idx

    path, vertex_id, metadata = prm.get_shortest_path_to_position(init_vertex_id, pick_position.name)

    if not path:
        print(f"Failed to find path to {pick_position.name}")
        return init_vertex_id

    print(f"path length to {pick_position.name}: {len(path)}")

    prm.mp.animate_path("ur5e_1", path)
    time.sleep(0.5)

    return vertex_id

vid = plan_with_prm_and_animate(prm, pick_position=positions_dict["bottom_right"])
vid = plan_with_prm_and_animate(prm, pick_position=positions_dict["workspace_red_cup"], init_vertex_id=vid)
vid = plan_with_prm_and_animate(prm, pick_position=positions_dict["bottom_left"], init_vertex_id=vid)
vid = plan_with_prm_and_animate(prm, pick_position=positions_dict["top_shelf_right"], init_vertex_id=vid)

path = prm.find_path_by_vertices_id(vid, prm.interest_configs[1][1])
vid = prm.interest_configs[1][1]
prm.mp.animate_path("ur5e_1", path)

vid = plan_with_prm_and_animate(prm, pick_position=positions_dict["workspace_blue_cup"], init_vertex_id=vid)
vid = plan_with_prm_and_animate(prm, pick_position=positions_dict["bottom_left"], init_vertex_id=vid)
vid = plan_with_prm_and_animate(prm, pick_position=positions_dict["workspace_red_cup"], init_vertex_id=vid)
vid = plan_with_prm_and_animate(prm, pick_position=positions_dict["middle_shelf"], init_vertex_id=vid)

path = prm.find_path_by_vertices_id(vid, prm.interest_configs[2][1])
vid = prm.interest_configs[2][1]
prm.mp.animate_path("ur5e_1", path)

