import time
from typing import Dict
import pomdp_py
import numpy as np
import copy
from lab_po_manipulation.generate_prms import load_prm
from lab_po_manipulation.pomdp_model.full_info_planning import plan_full_info_manipulation
from lab_po_manipulation.pomdp_model.pomanipulation_problem import (
    POMProblem, POMRolloutPolicy, POMActionPrior
)


class PRMCache:
    def __init__(self):
        self._prms = {}
        self._action_priors = {}

    def get_prms(self, obstacles_to_exclude):
        key = tuple(sorted(obstacles_to_exclude))  # Make hashable
        if key not in self._prms:
            self._prms[key] = (
                load_prm(obstacles_to_exclude=obstacles_to_exclude, constrained=False),
                load_prm(obstacles_to_exclude=obstacles_to_exclude, constrained=True)
            )
        return self._prms[key]

    def get_action_prior(self, obstacles_to_exclude, vid_to_sensing_config_name):
        key = tuple(sorted(obstacles_to_exclude))
        if key not in self._action_priors:
            prms = self.get_prms(obstacles_to_exclude)
            self._action_priors[key] = POMActionPrior(
                prms[0], prms[1], vid_to_sensing_config_name
            )
        return self._action_priors[key]



# Global cache instance
prm_cache = PRMCache()


def h_vd_results(problem: POMProblem, obstacles_to_exclude: list, vd_table, n_states) -> tuple[float, float]:
    """
    Baseline heuristic using subset of empirical value differences.
    Returns: (value_difference, computation_time)
    """
    help_id = obstacles_to_exclude[0] - 1
    vd_subset = vd_table[vd_table['help_id'] == help_id]

    # Take a random subset of n_states_to_use states
    if len(vd_subset) < n_states:
        raise ValueError(f"Not enough states in table. Have {len(vd_subset)}, need {n_states}")

    sample = vd_subset.sample(n=n_states)
    return (
        sample['value_diff'].mean(),
        sample['computation_time'].sum()  # Once you add computation time to the vd_table
    )


def plan_one_step_and_get_tree_values(problem: POMProblem, pomcp_params: dict = None) -> dict:
    """
    Plan one step ahead and return the tree values for all actions.
    Returns dict: {action: value}
    """
    if pomcp_params is None:
        pomcp_params = {}

    # Create a copy of the problem
    pomcp = pomdp_py.POMCP(
        num_sims=pomcp_params.get('num_sims', 2000),
        max_depth=pomcp_params.get('max_depth', 35),
        discount_factor=pomcp_params.get('discount_factor', 0.98),
        exploration_const=pomcp_params.get('exploration_const', np.sqrt(2)),
        action_prior=pomcp_params.get('action_prior', None),
        rollout_policy=pomcp_params.get('rollout_policy', problem.agent.policy_model),
        show_progress=False
    )

    action = pomcp.plan(problem.agent)
    tree = problem.agent.tree

    action_values = {action: child.value for action, child in tree.children.items()}
    return action_values


def first_step_planning_value(problem: POMProblem, obstacles_to_exclude, n_sims, max_depth) -> float:
    """Get value from one-step planning with POMCP"""
    action_prior = prm_cache.get_action_prior(obstacles_to_exclude,
                                              problem.agent.policy_model.vid_to_sensing_config_name)

    rollout_policy = POMRolloutPolicy(
        problem.agent.policy_model.prm,
        problem.agent.policy_model.prm_y_up,
        problem.agent.policy_model.vid_to_sensing_config_name,
        action_prior
    )

    pomcp_params = {
        "action_prior": action_prior,
        "rollout_policy": rollout_policy,
        "num_sims": n_sims,
        "max_depth": max_depth
    }

    action_values = plan_one_step_and_get_tree_values(problem, pomcp_params)
    return max(action_values.values())


def h_first_step_planning_value_diff(problem: POMProblem, obstacles_to_exclude: list,
                                     n_sims=8000, max_depth=30, n_trials=1) -> (float, float):
    """
    Compute value difference between problem with and without help using first-step planning
    """
    prm_helped, prm_y_up_helped = prm_cache.get_prms(obstacles_to_exclude)
    prm_no_help, prm_y_up_no_help = prm_cache.get_prms([])

    comp_time = 0

    vdiffs = []
    for _ in range(n_trials):
        # create new problems at each iteration as agent tree is modified
        problem = POMProblem(prm_no_help, prm_y_up_no_help)
        problem_helped = POMProblem(prm_helped, prm_y_up_helped)

        t = time.time()
        vdiff = first_step_planning_value(problem_helped, obstacles_to_exclude, n_sims=n_sims, max_depth=max_depth) \
                - first_step_planning_value(problem, [], n_sims=n_sims, max_depth=max_depth)
        vdiffs.append(vdiff)
        comp_time += time.time() - t
    return sum(vdiffs) / n_trials, comp_time


def perform_rollout(problem: POMProblem, rollout_policy: POMRolloutPolicy, max_steps: int,
                    discount_factor: float) -> float:
    """Perform a single rollout with given policy"""
    total_discounted_reward = 0
    state = copy.deepcopy(problem.env.state)
    history = []

    for i in range(max_steps):
        action = rollout_policy.rollout(state, history)
        reward = problem.env.state_transition(action, execute=True)
        obs = problem.env.provide_observation(problem.agent.observation_model, action)

        history.append((action, obs))
        total_discounted_reward += reward * (discount_factor ** i)

        if problem.env.state.terminal or problem.env.state.can_in_trash:
            break

    return total_discounted_reward


def h_rollout_policy_value(problem: POMProblem, obstacles_to_exclude: list, n_rollouts=10, max_steps=70)\
        -> tuple[float, float]:
    """
    Compute value difference using simplified policy rollouts
    Returns: (value_difference, computation_time)
    """
    # Get PRMs and action priors from cache
    prm_no_help, prm_y_up_no_help = prm_cache.get_prms([])
    prm_helped, prm_y_up_helped = prm_cache.get_prms(obstacles_to_exclude)

    action_prior_no_help = prm_cache.get_action_prior([], problem.agent.policy_model.vid_to_sensing_config_name)
    problem_helped = POMProblem(prm_helped, prm_y_up_helped)
    action_prior_helped = prm_cache.get_action_prior(obstacles_to_exclude,
                                                     problem_helped.agent.policy_model.vid_to_sensing_config_name)

    rollout_policy_no_help = POMRolloutPolicy(prm_no_help, prm_y_up_no_help,
                                              action_prior_no_help.vid_to_sensing_config_name, action_prior_no_help)
    rollout_policy_helped = POMRolloutPolicy(prm_helped, prm_y_up_helped,
                                             action_prior_helped.vid_to_sensing_config_name, action_prior_helped)

    vdiffs = []
    comp_time = 0
    for _ in range(n_rollouts):
        # create new problem and helped problem at every step
        init_state_no_help = POMProblem.sample_init_state(prm_no_help)
        init_state_helped = copy.deepcopy(init_state_no_help)
        init_state_helped.robot_cfg_id = prm_helped.home_cfg_idx
        problem_nohelp = POMProblem(prm_no_help, prm_y_up_no_help, init_state=init_state_no_help)
        problem_helped = POMProblem(prm_helped, prm_y_up_helped, init_state=init_state_helped)

        start_time = time.time()
        vdiff = perform_rollout(problem_helped, rollout_policy_helped, max_steps, discount_factor=0.98) \
                - perform_rollout(problem_nohelp, rollout_policy_no_help, max_steps, discount_factor=0.98)

        comp_time += time.time() - start_time
        vdiffs.append(vdiff)

    return sum(vdiffs) / n_rollouts, comp_time


def h_full_info_planning_value_diff(problem: POMProblem, obstacles_to_exclude: list, n_states=120,
                                    discount_factor: float = 0.98) -> tuple[float, float]:
    """
    Compute value difference using full info planning.
    Returns (value_difference, computation_time)
    """

    # Get PRMs with and without help
    prm_helped, prm_y_up_helped = prm_cache.get_prms(obstacles_to_exclude)
    prm_no_help, prm_y_up_no_help = prm_cache.get_prms([])

    start_time = time.time()

    vdiffs = []
    for _ in range(n_states):
        # create new problem and helped problem at every step
        init_state_no_help = POMProblem.sample_init_state(prm_no_help)
        init_state_helped = copy.deepcopy(init_state_no_help)
        init_state_helped.robot_cfg_id = prm_helped.home_cfg_idx

        value_without_help = plan_full_info_manipulation(prm_no_help, prm_y_up_no_help, init_state_no_help,
                                                         discount_factor)
        value_with_help = plan_full_info_manipulation(prm_helped, prm_y_up_helped, init_state_helped, discount_factor)

        assert value_without_help != float('-inf') and value_with_help != float('-inf')
        assert not np.isnan(value_with_help) and not np.isnan(value_without_help)

        vdiffs.append(value_with_help - value_without_help)

    computation_time = time.time() - start_time
    return sum(vdiffs) / n_states, computation_time


if __name__ == "__main__":
    h_full_info_planning_value_diff(None, [1], n_states=1)