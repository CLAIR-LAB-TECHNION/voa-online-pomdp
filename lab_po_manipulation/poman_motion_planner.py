import os

from klampt import Geometry3D
from klampt.math import so3
from klampt.model import collide
from klampt.model.geometry import box
from lab_po_manipulation.env_configurations.objects_and_positions import MobileObstacle, mobile_obstacles_dict, \
    positions_dict
from lab_ur_stack.motion_planning.motion_planner import MotionPlanner
import time


class POManMotionPlanner(MotionPlanner):
    def __init__(self, add_obstacles_for_objects_in_ws=True):
        self.grasped_object_attachment_geom_id = None
        super().__init__()
        if add_obstacles_for_objects_in_ws:
            self.add_obstacle_for_objects_in_ws()

    def _get_klampt_world_path(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        world_path = os.path.join(dir, "stat_object_klampt_world.xml")
        return world_path

    def add_obstacle_for_objects_in_ws(self,):
        positions_to_protect = [positions_dict["workspace_blue_cup"],
                                positions_dict["workspace_red_cup"],
                                positions_dict["workspace_can"],
                                positions_dict["bottom_left"],
                                positions_dict["bottom_right"],
                                positions_dict["middle_shelf_left"],]

        size = [0.065, 0.065, 0.2]

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "../lab_ur_stack/motion_planning/objects/cube.off")

        for position in positions_to_protect:
            obj = {"name": position.name, "scale": size,
                   "coordinates": position.position,
                   "color": [0.5, 0.5, 0.0, 0.75],
                   "geometry_file": path,
                   "angle": so3.identity()}
            self.add_object_to_world(position.name, obj)


    def add_mobile_obstacle(self, mobile_obstacle: MobileObstacle):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "../lab_ur_stack/motion_planning/objects/cube.off")
        obj = {"name": mobile_obstacle.name, "scale": mobile_obstacle.size,
               "coordinates": mobile_obstacle.initial_position,
               "color": [0.8, 0.3, 0.3, 0.75],
               "geometry_file": path,
               "angle": so3.identity()}
        self.add_object_to_world(mobile_obstacle.name, obj)

    def add_mobile_obstacle_by_name(self, name: str):
        self.add_mobile_obstacle(mobile_obstacles_dict[name])

    def add_mobile_obstacle_by_number(self, number: int):
        self.add_mobile_obstacle_by_name(f"mobile_obstacle_{number}")

    def add_grasped_object_attachment(self):
        if self.grasped_object_attachment_geom_id is None:
            n_elements = self.ur5e_1.link("ee_link").geometry().numElements()
            self.grasped_object_attachment_geom_id = n_elements

        geom_id = self.grasped_object_attachment_geom_id
        grasped_obj_obj = box(0.07, 0.16, 0.07, center=[0, -0.02, 0.13 - self.ee_offset])
        grasped_obj_geom = Geometry3D()
        grasped_obj_geom.set(grasped_obj_obj)

        self.ur5e_1.link("ee_link").geometry().setElement(geom_id, grasped_obj_geom)

    def remove_grasped_object_attachment(self):
        if self.grasped_object_attachment_geom_id is not None:
            # dummy small box
            grasped_obj_obj = box(0.001, 0.001, 0.001, center=[0, 0, 0])
            grasped_obj_geom = Geometry3D()
            grasped_obj_geom.set(grasped_obj_obj)
            self.ur5e_1.link("ee_link").geometry().setElement(self.grasped_object_attachment_geom_id, grasped_obj_geom)

    def _add_attachments(self, robot, attachments):
        super()._add_attachments(robot, attachments)
        self.add_grasped_object_attachment()


if __name__ == "__main__":
    planner = POManMotionPlanner()
    for i in range(1, 5):
        planner.add_mobile_obstacle_by_number(i)
    planner.visualize()

    time.sleep(100)
