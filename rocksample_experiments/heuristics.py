from itertools import product
from typing import Dict
import pomdp_py
from rocksample_experiments.full_info_planning import get_full_info_value
from rocksample_experiments.help_actions import push_rocks
from rocksample_experiments.preferred_actions import RSActionPrior, CustomRSPolicyModel
from rocksample_experiments.rocksample_problem import RockSampleProblem, RockType
from rocksample_experiments.full_info_planning import get_full_info_value, get_full_info_value_greedy
from copy import deepcopy


def h_vd_results(problem: RockSampleProblem, help_config, vd_table, n_states_to_use) -> float:
    """
    this one is for sanity checks and to see how many samples are enough. We use the already
    made rollout with the actual policy. This is actually not heuristic but the same way we compute
    empriical VOA, possibly with fewer samples
    """
    # take entries from vd table with same problem params and same help config:
    rover_position_str = str(list(problem.env.state.position))
    rock_locs = problem.rock_locs
    rock_locs = {str(k): v for k, v in rock_locs.items()}
    rock_locs_str = str(rock_locs)
    help_config_str = {str(k): v for k, v in help_config.items()}
    help_config_str = str(help_config_str)

    vd_table_for_problem_help_pair = vd_table[vd_table['rover_position'] == rover_position_str]
    vd_table_for_problem_help_pair = \
        vd_table_for_problem_help_pair[vd_table_for_problem_help_pair['rock_locations'] == rock_locs_str]
    vd_table_for_problem_help_pair = \
        vd_table_for_problem_help_pair[vd_table_for_problem_help_pair['help_actions'] == help_config_str]
    value_diffs = vd_table_for_problem_help_pair['value_diff'].values
    value_diffs = value_diffs[:n_states_to_use]
    return sum(value_diffs) / len(value_diffs)



def plan_one_step_and_get_tree_values(problem: RockSampleProblem, pomcp_params: dict = None) -> dict:
    """
    Plan one step ahead and return the tree values for all actions.
    returns dict: {action: value}
    """
    if pomcp_params is None:
        pomcp_params = {}

    # don't modify the original problem
    problem = deepcopy(problem)

    pomcp = pomdp_py.POMCP(num_sims=pomcp_params.get('num_sims', 2000),
                           max_depth=pomcp_params.get('max_depth', 20),
                           discount_factor=pomcp_params.get('discount_factor', 0.95),
                           exploration_const=pomcp_params.get('exploration_const', 5),
                           action_prior=pomcp_params.get('action_prior', None),
                           rollout_policy=pomcp_params.get('rollout_policy', problem.agent.policy_model),
                           show_progress=False)

    action = pomcp.plan(problem.agent)
    tree = problem.agent.tree

    action_values = {action: child.value for action, child in tree.children.items()}
    return action_values


def first_step_planning_value(problem: RockSampleProblem, n_sims, max_depth) -> float:
    action_prior = RSActionPrior(problem.n, problem.k, problem.rock_locs)
    pomcp_params = {"action_prior": action_prior,
                    "rollout_policy": CustomRSPolicyModel(problem.n, problem.k, actions_prior=action_prior),
                    "num_sims": n_sims,
                    "max_depth": max_depth}
    action_values = plan_one_step_and_get_tree_values(problem, pomcp_params)
    action_values = action_values.values()
    return max(action_values)


def h_first_step_planning_value_diff(problem: RockSampleProblem, help_config, n_sims=2000,
                                     max_depth=20, n_trials=1) -> float:
    problem_helped, _ = push_rocks(problem, help_config, deepcopy_belief=True)

    vdiffs = []
    for _ in range(n_trials):
        vdiff = first_step_planning_value(problem_helped, n_sims=n_sims, max_depth=max_depth) \
                - first_step_planning_value(problem, n_sims=n_sims, max_depth=max_depth)
        vdiffs.append(vdiff)
    return sum(vdiffs) / n_trials


def perform_rollout(problem: RockSampleProblem, rollout_policy: CustomRSPolicyModel, max_stpes: int,
                    discount_factor) -> float:
    # create copy of the problem since we will be modifying it, dont use deepcopy since copying the belief
    # takes time and we don't use it here
    total_discounted_reward = 0
    state = problem.env.state
    history = []
    for i in range(max_stpes):
        action = rollout_policy.rollout(state, history)
        reward = problem.env.state_transition(action, execute=True)
        obs = problem.env.provide_observation(problem.agent.observation_model, action)

        history.append((action, obs))
        total_discounted_reward += reward * discount_factor ** i

    return total_discounted_reward


def h_rollout_policy_value(problem: RockSampleProblem, help_config, n_rollouts=10) -> float:
    action_prior = RSActionPrior(problem.n, problem.k, problem.rock_locs)
    rollout_policy = CustomRSPolicyModel(problem.n, problem.k, actions_prior=action_prior)

    vdiffs = []

    for i in range(n_rollouts):
        # create new problem but sample rock types randomly
        rocktypes = [RockType.random() for _ in range(problem.k)]
        init_state = deepcopy(problem.env.state)
        init_state.rocktypes = rocktypes
        curr_problem = RockSampleProblem(n=problem.n, k=problem.k, rock_locs=problem.rock_locs,
                                         init_state=init_state, init_belief=problem.agent.init_belief,
                                         half_efficiency_dist=problem.agent.observation_model._half_efficiency_dist)


        problem_helped, _ = push_rocks(curr_problem, help_config, deepcopy_belief=False)

        vd = perform_rollout(problem_helped, rollout_policy, max_stpes=100, discount_factor=0.95) \
             - perform_rollout(curr_problem, rollout_policy, max_stpes=100, discount_factor=0.95)
        vdiffs.append(vd)

    return sum(vdiffs) / n_rollouts


def h_full_info_planning_value_diff(problem: RockSampleProblem, help_config: Dict, n_states=1) -> float:
    """
    Compute VOA heuristic based on difference in full information planning values.
    """

    heuristic_function = get_full_info_value if problem.k <=9 else get_full_info_value_greedy

    vd = []
    for _ in range(n_states):
        rocktypes = [RockType.random() for _ in range(problem.k)]
        curr_init_state = deepcopy(problem.env.state)
        curr_init_state.rocktypes = rocktypes
        curr_problem = RockSampleProblem(n=problem.n, k=problem.k, rock_locs=problem.rock_locs,
                                            init_state=curr_init_state, init_belief=problem.agent.init_belief)

        curr_problem_helped, _ = push_rocks(curr_problem, help_config, deepcopy_belief=False)

        value_without_help = heuristic_function(curr_problem)
        value_with_help = heuristic_function(curr_problem_helped)
        vd.append(value_with_help - value_without_help)
    return sum(vd) / n_states


def h_full_info_planning_value_all_states(problem: RockSampleProblem, help_config: Dict) -> float:
    """
    Compute VOA heuristic based on difference in full information planning values.
    """

    heuristic_function = get_full_info_value if problem.k <=9 else get_full_info_value_greedy

    options = ['bad', 'good']
    combinations = list(product(options, repeat=11))
    all_rock_types = [list(combo) for combo in combinations]

    vd = []
    for rocktypes in all_rock_types:
        curr_init_state = deepcopy(problem.env.state)
        curr_init_state.rocktypes = rocktypes
        curr_problem = RockSampleProblem(n=problem.n, k=problem.k, rock_locs=problem.rock_locs,
                                            init_state=curr_init_state, init_belief=problem.agent.init_belief)

        curr_problem_helped, _ = push_rocks(curr_problem, help_config, deepcopy_belief=False)

        value_without_help = heuristic_function(curr_problem)
        value_with_help = heuristic_function(curr_problem_helped)
        vd.append(value_with_help - value_without_help)
    return sum(vd) / len(vd)


if __name__ == '__main__':
    from rocksample_experiments.utils import sample_problem_from_voa_row, get_help_action_from_row
    import pandas as pd

    row = pd.Series({
        'env_instance_id': 1,
        'help_config_id': 1,
        'help_actions': "{'10': [-1, -8], '7': [-5, -4], '1': [0, -8]}",
        'rover_position': "[0, 10]",
        'rock_locations': "{'(5, 4)': 0, '(2, 9)': 1, '(7, 4)': 2, '(9, 7)': 3, '(10, 9)': 4, '(8, 2)': 5, '(5, 3)': 6, '(6, 6)': 7, '(8, 0)': 8, '(10, 1)': 9, '(3, 10)': 10}",
        'empirical_voa': -9.262964,
        'empirical_voa_variance': 42.025261,
        'n_states': 40,
        'baseline_value': 14.601503,
        'std_error': 1.025003
    })

    for i in range(20):
        problem = sample_problem_from_voa_row(row, n=10)
        help_config = get_help_action_from_row(row)
        h = h_rollout_policy_value(problem, help_config, n_rollouts=100)
