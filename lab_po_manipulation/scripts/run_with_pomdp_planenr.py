import numpy as np
import pomdp_py

from lab_po_manipulation.env_configurations.objects_and_positions import (positions, objects_dict,
                                                                          mobile_obstacles_dict, mobile_obstacles,
                                                                          positions_dict, helper_idle_config)
from lab_po_manipulation.generate_prms import load_prm
from lab_po_manipulation.lab_poman_env import LabPOMANEnv
from lab_po_manipulation.pomdp_model.pomanipulation_problem import POMProblem, POMActionPrior, POMRolloutPolicy
from lab_ur_stack.manipulation.manipulation_controller_vg import ManipulationControllerVG
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.manipulation.manipulation_controller_2fg import ManipulationController2FG
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1, ur5e_2


without_obstacle = 1

# initial_positions = {"top_shelf_right": "red_cup", "middle_shelf": "blue_cup", "bottom_left": "soda_can"}
all_initial_positions = {'top_shelf_left': None,
 'top_shelf_right': 'red_cup',
 'middle_shelf': 'blue_cup',
 'middle_shelf_left': None,
 'bottom_left': 'soda_can',
 'bottom_right': None,
 'workspace_can': None,
 'workspace_red_cup': None,
 'workspace_blue_cup': None}


prm = load_prm(obstacles_to_exclude=[without_obstacle], constrained=False)
prm_y_up = load_prm(obstacles_to_exclude=[without_obstacle], constrained=True)

mp = prm.mp
gt = GeometryAndTransforms.from_motion_planner(mp)

r1_controller = ManipulationController2FG(ur5e_1["ip"], ur5e_1["name"], mp, gt)
r1_controller.speed, r1_controller.acceleration = 0.4, .75
r2_controller = ManipulationControllerVG(ur5e_2["ip"], ur5e_2["name"], mp, gt)
r2_controller.speed, r2_controller.acceleration = 0.75, .75


# sample_state = POMProblem.sample_init_state(prm)
# all_initial_positions = sample_state.positions_to_objects
# for key in all_initial_positions:
#     all_initial_positions[key] = None
# for key in initial_positions:
#     all_initial_positions[key] = initial_positions[key]


agent = POMProblem.build_agent(prm, prm_y_up)

env = LabPOMANEnv(r1_controller, r2_controller, agent, all_initial_positions, plot_detections=True)

env.reset()

# setup POMCP planner
action_prior = POMActionPrior(prm, prm_y_up, agent.policy_model.vid_to_sensing_config_name)
rollout_policy = POMRolloutPolicy(prm, prm_y_up, agent.policy_model.vid_to_sensing_config_name,
                                  action_prior)
pomcp = pomdp_py.POMCP(
        num_sims=5000,
        max_depth=35,
        discount_factor=0.98,
        exploration_const=np.sqrt(2),
        action_prior=action_prior,
        rollout_policy=rollout_policy ,
        show_progress=True
    )

for step in range(60):
    action = pomcp.plan(agent)

    obs, reward, done, info = env.step(action)
    pomcp.update(agent, action, obs)
    agent.update_history(action, obs)

    if done:
        break