from copy import deepcopy
import pomdp_py

from rocksample_experiments.preferred_actions import RSActionPrior, CustomRSPolicyModel
from rocksample_experiments.rocksample_problem import RockSampleProblem


class PathTrackingPOMCP(pomdp_py.POMCP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_history = []
        self.current_simulation = []
        self.simulation_count = 0
        self.max_simulations_to_track = 10
        self.track_action = "check-7"
        self.is_tracking = False  # Flag to track full paths

    def _simulate(self, state, history, root, parent, observation, depth):
        if self.simulation_count >= self.max_simulations_to_track:
            return super()._simulate(state, history, root, parent, observation, depth)

        current_action = root.argmax() if root else None

        # Start tracking if we see check-7 at depth 0
        if depth == 0:
            self.is_tracking = str(current_action) == self.track_action

        # Record info if we're tracking this path
        if self.is_tracking:
            step_info = {
                'depth': depth,
                'state': str(state),
                'action': str(current_action),
                'parent_value_before': parent.value if parent else None,
                'parent_visits': parent.num_visits if parent else None
            }

            self.current_simulation.append(step_info)

        # Run simulation
        result = super()._simulate(state, history, root, parent, observation, depth)

        # Update info if we're tracking
        if self.is_tracking:
            # Calculate value update carefully avoiding division by zero
            if parent and parent.num_visits > 0:
                value_update = (result - parent.value) / parent.num_visits
            else:
                value_update = None

            step_info.update({
                'immediate_reward': result,
                'parent_value_after': parent.value if parent else None,
                'value_update': value_update
            })

            if depth == 0:
                if len(self.current_simulation) > 0:  # Only save if we tracked something
                    self.path_history.append(self.current_simulation)
                    self.simulation_count += 1
                self.current_simulation = []
                self.is_tracking = False

        return result

    def clear_path_history(self):
        self.path_history = []
        self.current_simulation = []
        self.simulation_count = 0
        self.is_tracking = False

        
def run_tracking_test(problem: RockSampleProblem):
    problem = deepcopy(problem)

    action_prior = RSActionPrior(problem.n, problem.k, problem.rock_locs)
    pomcp = PathTrackingPOMCP(
        num_sims=2000,
        max_depth=20,
        discount_factor=0.95,
        exploration_const=5,
        action_prior=action_prior,
        rollout_policy=CustomRSPolicyModel(problem.n, problem.k, actions_prior=action_prior),
        show_progress=False
    )

    # Run planning
    pomcp.clear_path_history()
    action = pomcp.plan(problem.agent)

    # Print detailed simulation sequences
    print("\nDetailed Simulation Sequences for check-7:")
    for sim_idx, simulation in enumerate(pomcp.path_history):
        print(f"\nSimulation {sim_idx + 1}:")
        print("Depth | Action | Reward | Parent Value (before -> after) | Visits | Value Update")
        print("-" * 90)
        for step in simulation:
            depth_str = str(step['depth'])
            action_str = str(step['action']) if step['action'] is not None else "None"
            reward_str = f"{step['immediate_reward']:.3f}" if step['immediate_reward'] is not None else "None"
            val_before = f"{step['parent_value_before']:.3f}" if step['parent_value_before'] is not None else "None"
            val_after = f"{step['parent_value_after']:.3f}" if step['parent_value_after'] is not None else "None"
            visits = str(step['parent_visits']) if step['parent_visits'] is not None else "None"
            update = f"{step['value_update']:.3f}" if step['value_update'] is not None else "None"

            print(f"{depth_str:>5} | {action_str:<15} | {reward_str:>6} | "
                  f"{val_before:>6} -> {val_after:>6} | {visits:>6} | {update:>6}")

    return pomcp.path_history


# Run the test
if __name__ == '__main__':
    from rocksample_experiments.utils import sample_problem_from_voa_row
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

    problem = sample_problem_from_voa_row(row, 11)
    simulation_history = run_tracking_test(problem)