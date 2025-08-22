import time

import numpy as np
import pomdp_py
import os
from datetime import datetime
import pickle
import json
import cv2
from pathlib import Path

from matplotlib import pyplot as plt

from lab_po_manipulation.env_configurations.objects_and_positions import helper_idle_config, objects, \
    pre_pour_config_red, pour_pose_red, pour_pose_blue, positions, objects_dict, pre_pour_config_blue, \
    sensing_configs_dict
from lab_po_manipulation.image_state_estimation import ObjectPositionEstimator
from lab_po_manipulation.manipulation_prm import ManipulationPRM
from lab_po_manipulation.obstacles_moving import move_obstacle_1, move_obstacle_2, move_obstacle_3, move_obstacle_4
from lab_po_manipulation.pomdp_model.pomanipulation_problem import Action, BaseObservation, POMRewardModel, \
    PutDownAction, PickUpAction, MoveAction, PourAction, TrashAction, SenseAction, SensingObservation
from lab_ur_stack.camera.realsense_camera import RealsenseCamera
from lab_ur_stack.manipulation.manipulation_controller_2fg import ManipulationController2FG
from lab_ur_stack.manipulation.manipulation_controller_vg import ManipulationControllerVG

detection_classes = ('green soda can', 'red cup with black strap', 'blue cup')
detection_classes_to_actual_names = {'green soda can': 'soda_can', 'red cup with black strap': 'red_cup',
                                     'blue cup': 'blue_cup'}


class LabPOMANEnv:
    def __init__(self,
                 r1_controller: ManipulationController2FG,
                 r2_controller: ManipulationControllerVG,
                 agent: pomdp_py.Agent,
                 initial_position_to_objects,
                 plot_detections=False):
        self.r1_controller = r1_controller
        self.r2_controller = r2_controller
        self.agent = agent
        self.plot_detections = plot_detections

        self.reward_model: POMRewardModel = agent.reward_model
        self.prm: ManipulationPRM = agent.policy_model.prm
        self.prm_y_up: ManipulationPRM = agent.policy_model.prm_y_up
        self.camera = RealsenseCamera()
        self.position_estimator = ObjectPositionEstimator(detection_classes, positions, r1_controller.gt)

        self.max_steps = 60
        self.step_count = 0

        # state data
        self.positions_to_objects = initial_position_to_objects
        self.current_cfg_id = self.prm.home_cfg_idx
        self.state_red_cup_full = False
        self.state_blue_cup_full = False
        self.held_object = None

        # Set up logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(f"lab_runs/{timestamp}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize history
        self.history = {
            'actions': [],
            'observations': [],
            'rewards': [],
            'positions_to_objects': [],
            'robot_states': [],
            'step_info': []
        }

    def _save_step_data(self, action, observation, reward, info, annotated_img=None, depth_vis=None):
        """Helper method to save data for each step"""
        # Store robot state
        robot_state = {
            'current_cfg_id': self.current_cfg_id,
            'robot_config': self.r1_controller.getActualQ(),
        }

        # Update history
        self.history['actions'].append({
            'step': self.step_count,
            'action_type': action.__class__.__name__,
            'action_params': action.__dict__
        })

        self.history['observations'].append({
            'step': self.step_count,
            'observation': observation.__dict__
        })

        self.history['rewards'].append(reward)
        self.history['positions_to_objects'].append(dict(self.positions_to_objects))
        self.history['robot_states'].append(robot_state)
        self.history['step_info'].append(info)

        # Save images if provided (sensing action)
        if annotated_img is not None and depth_vis is not None:
            cv2.imwrite(
                str(self.run_dir / f"step_{self.step_count}_detections.png"),
                cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                str(self.run_dir / f"step_{self.step_count}_depth.png"),
                depth_vis
            )

        # Save current history state
        with open(self.run_dir / "history.pkl", "wb") as f:
            pickle.dump(self.history, f)

        # Update and save summary
        summary = {
            'total_steps': self.step_count,
            'final_state': {
                'positions_to_objects': self.positions_to_objects,
                'red_cup_full': self.state_red_cup_full,
                'blue_cup_full': self.state_blue_cup_full,
                'held_object': self.held_object
            }
        }
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def reset(self):
        self.r1_controller.plan_and_move_home()
        self.r2_controller.plan_and_moveJ(helper_idle_config)
        self.step_count = 0

        # Clear history when environment is reset
        self.history = {
            'actions': [],
            'observations': [],
            'rewards': [],
            'positions_to_objects': [],
            'robot_states': [],
            'step_info': []
        }

    def helper_move_obstacle(self, obstacle_number):
        if obstacle_number == 1:
            move_obstacle_1(self.r2_controller)
        elif obstacle_number == 2:
            move_obstacle_2(self.r2_controller)
        elif obstacle_number == 3:
            move_obstacle_3(self.r2_controller)
        elif obstacle_number == 4:
            move_obstacle_4(self.r2_controller)
        else:
            raise ValueError("Invalid obstacle number")

    def step(self, action: Action) -> (BaseObservation, float, bool, dict):
        current_prm = self.prm if self.held_object != 'soda_can' else self.prm_y_up
        reward = 0
        info = {}
        done = self.step_count >= self.max_steps
        observation = None
        annotated_img = None
        depth_vis = None

        self.step_count += 1

        if isinstance(action, MoveAction):
            config = current_prm.vertices[action.cfg_id]
            self.r1_controller.moveJ(config)
            self.current_cfg_id = action.cfg_id

        elif isinstance(action, PickUpAction):
            grasp_config = current_prm.grasp_configs.get(self.current_cfg_id, None)
            self.grasp_and_update_state(grasp_config)

            if self.held_object == 'soda_can':
                config = self.prm.vertices[self.current_cfg_id]
                self.current_cfg_id = self.prm_y_up.vertex_to_id[tuple(config)]

        elif isinstance(action, PutDownAction):
            move_prm = False
            if self.held_object == 'soda_can':
                move_prm = True
            grasp_config = current_prm.grasp_configs.get(self.current_cfg_id, None)
            self.put_and_update_state(grasp_config)

            if move_prm:
                config = self.prm_y_up.vertices[self.current_cfg_id]
                self.current_cfg_id = self.prm.vertex_to_id[tuple(config)]


        elif isinstance(action, PourAction):
            pre_pour_pose = self.r1_controller.getActualTCPPose()
            curr_config = np.array(self.r1_controller.getActualQ())
            pour_red_dist = np.linalg.norm(curr_config - np.array(pre_pour_config_red))
            pour_blue_dist = np.linalg.norm(curr_config - np.array(pre_pour_config_blue))
            if pour_red_dist < pour_blue_dist:
                self.r1_controller.moveL(pour_pose_red, speed=0.2, acceleration=0.1)
                self.state_red_cup_full = True
            else:
                self.r1_controller.moveL(pour_pose_blue, speed=0.2, acceleration=0.1)
                self.state_blue_cup_full = True

            time.sleep(0.5)
            self.r1_controller.moveL(pre_pour_pose, speed=0.2, acceleration=0.1)

        elif isinstance(action, TrashAction):
            self.r1_controller.release_grasp()
            self.r1_controller.plan_and_move_home()
            done = True

        elif isinstance(action, SenseAction):
            rgb, depth = self.camera.get_frame_rgb()
            results_dict, (annotated_img, depth_vis) = (
                self.position_estimator.estimate_positions(rgb, depth, self.r1_controller.getActualQ()))

            mapped_positions = results_dict['mapped_positions']
            obs_positions_to_objects = {}
            for object_name, position in mapped_positions.items():
                obs_positions_to_objects[position["position"].name] = detection_classes_to_actual_names[object_name]

            # avoid particle deprivation that can happen by returning observation that is not modeled:
            sense_config_name = self.agent.policy_model.vid_to_sensing_config_name[self.current_cfg_id]
            sense_config_visible_position = sensing_configs_dict[sense_config_name].visible_positions
            for pos, object_name in obs_positions_to_objects.items():
                if pos not in sense_config_visible_position:
                    del obs_positions_to_objects[pos]

            observation = SensingObservation(obs_positions_to_objects,
                                             robot_cfg_id=self.current_cfg_id,
                                             held_object=self.held_object,
                                             red_cup_full=self.state_red_cup_full,
                                             blue_cup_full=self.state_blue_cup_full)

            if self.plot_detections:
                plt.imshow(depth_vis)
                plt.show()
                plt.imshow(annotated_img)
                plt.show()

        else:
            raise ValueError("Invalid action")

        if observation is None:
            observation = BaseObservation(held_object=self.held_object,
                                          red_cup_full=self.state_red_cup_full,
                                          blue_cup_full=self.state_blue_cup_full)


        # Save step data
        self._save_step_data(action, observation, reward, info, annotated_img, depth_vis)

        return observation, reward, done, info

    def grasp_and_update_state(self, grasp_config):
        if grasp_config is None:
            raise ValueError("Grasp config not found")
        self.r1_controller.release_grasp()

        position_name = grasp_config["position_name"]
        object_name = self.positions_to_objects[position_name]
        object_data = objects_dict[object_name]

        grasp_width = object_data.grasp_width
        grasp_offset = np.array(grasp_config["grasp_offset"])

        self.r1_controller.moveL_relative(-grasp_offset, speed=0.1, acceleration=0.1)
        self.r1_controller.grasp(width=grasp_width)
        self.r1_controller.moveL_relative(grasp_offset, speed=0.1, acceleration=0.1)
        if position_name in ['top_shelf_right', 'middle_shelf', 'top_shelf_left']:
            # for safety
            self.r1_controller.moveL_relative(0.3*grasp_offset, speed=0.1, acceleration=0.1)
            if grasp_offset[2] > 0.011:
                self.r1_controller.moveL_relative(0.4 * grasp_offset, speed=0.1, acceleration=0.1)

        self.held_object = object_name
        self.positions_to_objects[position_name] = 'empty'

    def put_and_update_state(self, grasp_config):
        if grasp_config is None:
            raise ValueError("Grasp config not found")

        position_name = grasp_config["position_name"]
        object_name = self.held_object
        grasp_offset = np.array(grasp_config["grasp_offset"])

        self.r1_controller.moveL_relative(-grasp_offset, speed=0.1, acceleration=0.1)
        self.r1_controller.release_grasp()
        self.r1_controller.moveL_relative(grasp_offset, speed=0.1, acceleration=0.1)

        self.held_object = None
        self.positions_to_objects[position_name] = object_name