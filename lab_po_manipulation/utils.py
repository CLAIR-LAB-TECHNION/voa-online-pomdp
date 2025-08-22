import numpy as np

from lab_po_manipulation.env_configurations.objects_and_positions import ItemPosition
from lab_po_manipulation.manipulation_prm import ManipulationPRM
from lab_ur_stack.manipulation.manipulation_controller_2fg import ManipulationController2FG


def plan_on_prm_and_pick_up(robot_controller: ManipulationController2FG,
                            prm: ManipulationPRM,
                            pick_position: ItemPosition,
                            init_vertex_id=None,
                            grasp_width=None,
                            ):
    """
    returns current vertex id
    """
    robot_controller.release_grasp()

    if init_vertex_id is None:
        q = robot_controller.getActualQ()
        init_vertex_id = prm.add_vertex(q)


    path, vertex_id, metadata = prm.get_shortest_path_to_position(init_vertex_id, pick_position.name)
    grasp_offset = np.array(metadata["grasp_offset"])

    robot_controller.move_path(path)
    robot_controller.moveL_relative(-grasp_offset, speed=0.1, acceleration=0.1)
    robot_controller.grasp(width=grasp_width)
    robot_controller.moveL_relative(grasp_offset, speed=0.1, acceleration=0.1)

    return vertex_id

def plan_on_prm_and_put_down(robot_controller: ManipulationController2FG,
                             prm: ManipulationPRM,
                             put_position: ItemPosition,
                             init_vertex_id=None,
                             ):
    """
    returns current vertex id
    """
    if init_vertex_id is None:
        q = robot_controller.getActualQ()
        init_vertex_id = prm.add_vertex(q)

    path, vertex_id, metadata = prm.get_shortest_path_to_position(init_vertex_id, put_position.name)
    grasp_offset = np.array(metadata["grasp_offset"])

    robot_controller.move_path(path)
    robot_controller.moveL_relative(-grasp_offset, speed=0.1, acceleration=0.1)
    robot_controller.release_grasp()
    robot_controller.moveL_relative(grasp_offset, speed=0.1, acceleration=0.1)

    return vertex_id

