import copy
import gc
import json
import os
import pickle
import random
import sys
import time
from contextlib import redirect_stdout
import io
from itertools import combinations, permutations
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pomdp_py
import typer
from lab_po_manipulation.generate_prms import load_prm
from lab_po_manipulation.pomdp_model.pomanipulation_problem import (
    POMProblem, POMRolloutPolicy, POMActionPrior, State
)


app = typer.Typer()


def run_pomcp_manipulation_instance(
        problem: POMProblem,
        pomcp_params: dict,
        max_steps: int = 50,
        verbose: int = 0,
) -> dict:
    """
    Run a single PO Manipulation experiment with given problem instance and POMCP parameters.

    Args:
        problem: Initialized POMProblem instance
        pomcp_params: Dictionary containing POMCP configuration parameters
        max_steps: Maximum number of steps to run
        verbose: Verbosity level (0: none, 1: summary, 2+: detailed)

    Returns:
        dict: Results dictionary containing:
            - total_reward: float
            - total_discounted_reward: float
            - history: dict with lists of states, actions, rewards, observations
            - final_state_info: dict with final state details
            - num_steps: int
            - terminated: bool (whether reached terminal state vs timeout)
    """

    def create_results(states, actions, rewards, observations, total_reward, total_discounted_reward, start_time):
        final_state = problem.env.state
        run_time = time.time() - start_time

        return {
            'total_reward': total_reward,
            'total_discounted_reward': total_discounted_reward,
            'num_steps': len(states),
            'run_time': run_time,
            'terminated': problem.env.state.terminal,
            'history': {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'observations': observations
            },
            'final_state_info': {
                'red_cup_full': final_state.red_cup_full,
                'blue_cup_full': final_state.blue_cup_full,
                'red_cup_in_position': final_state.positions_to_objects.get('workspace_red_cup') == 'red_cup',
                'blue_cup_in_position': final_state.positions_to_objects.get('workspace_blue_cup') == 'blue_cup',
                'can_in_trash': final_state.can_in_trash
            }
        }

    pomcp = pomdp_py.POMCP(
        num_sims=pomcp_params.get('num_sims', 2000),
        max_depth=pomcp_params.get('max_depth', 35),
        discount_factor=pomcp_params.get('discount_factor', 0.99),
        exploration_const=pomcp_params.get('exploration_const', np.sqrt(2)),
        action_prior=pomcp_params.get('action_prior', None),
        rollout_policy=pomcp_params.get('rollout_policy', problem.agent.policy_model),
        show_progress=verbose > 1
    )

    if verbose > 0:
        print("Initial state:")
        print(problem.env.state)

    total_reward = 0
    total_discounted_reward = 0
    gamma = 1.0
    discount = pomcp_params.get('discount_factor', 0.99)

    states = []
    actions = []
    rewards = []
    observations = []

    start_time = time.time()

    for step in range(max_steps):
        if verbose > 1:
            print(f"\nStep {step + 1}")

        states.append(copy.deepcopy(problem.env.state))

        # Plan and execute action
        action = pomcp.plan(problem.agent)
        actions.append(action)
        reward = problem.env.state_transition(action, execute=True)
        rewards.append(reward)

        # Get observation and update agent
        observation = problem.env.provide_observation(
            problem.agent.observation_model,
            action
        )
        observations.append(observation)

        try:
            pomcp.update(problem.agent, action, observation)
            problem.agent.update_history(action, observation)
        except ValueError as e:
            if "Particle deprivation" in str(e):
                if verbose > 1:
                    print("\nParticle deprivation occurred. Returning partial results.")
                return create_results(states, actions, rewards, observations, total_reward, total_discounted_reward,
                                      start_time)
            raise

        total_reward += reward
        total_discounted_reward += reward * gamma
        gamma *= discount

        if verbose > 1:
            print(f"Action: {action}")
            print(f"Observation: {observation}")
            print(f"Reward: {reward}")
            print(f"Total Reward: {total_reward}")
            print(f"Total Discounted Reward: {total_discounted_reward}")
            print(f"Current state: {problem.env.state}")

        # Check if terminal (can in trash)
        if problem.env.state.terminal:
            if verbose > 1:
                print("\nReached terminal state! Can is in trash. Terminating...")
            break

    return create_results(states, actions, rewards, observations, total_reward, total_discounted_reward, start_time)


def generate_all_initial_states() -> List[Dict]:
    """Generate all possible initial object configurations."""
    shelf_positions = [
        "top_shelf_left", "top_shelf_right", "middle_shelf",
        "middle_shelf_left", "bottom_left", "bottom_right"
    ]
    workspace_positions = ["workspace_red_cup", "workspace_blue_cup"]
    objects = ["red_cup", "blue_cup", "soda_can"]

    # Generate all possible position combinations for 3 objects
    position_combinations = list(combinations(shelf_positions, len(objects)))

    # For each combination, generate all possible object arrangements
    initial_states = []
    for pos_combo in position_combinations:
        for obj_perm in permutations(objects):
            # Initialize all positions including workspace positions as None
            positions_to_objects = {pos: None for pos in shelf_positions + workspace_positions}
            # Map selected positions to objects
            for pos, obj in zip(pos_combo, obj_perm):
                positions_to_objects[pos] = obj
            initial_states.append(positions_to_objects)

    return initial_states


@app.command()
def generate_experiment_configs(
        output_file: str = typer.Option(..., help="Output JSON file path"),
):
    """Generate experiment configurations for all possible initial states and help actions."""
    # Generate all possible initial states
    initial_states = generate_all_initial_states()

    metadata = {
        "num_initial_states": len(initial_states),
        "num_help_actions": 5,  # no help + 4 obstacles that can be removed
        "total_experiments": len(initial_states) * 5,
        "help_actions": ["none", "remove_obs1", "remove_obs2", "remove_obs3", "remove_obs4"],
        "positions": [
            "top_shelf_left", "top_shelf_right", "middle_shelf",
            "middle_shelf_left", "bottom_left", "bottom_right",
            "workspace_red_cup", "workspace_blue_cup"
        ]
    }

    experiments = []
    experiment_id = 0

    # Generate experiments for each initial state and help action
    for state_id, positions_to_objects in enumerate(initial_states):
        for help_id in range(5):  # 0: no help, 1-4: remove obstacle 1-4
            experiments.append({
                "experiment_id": experiment_id,
                "state_id": state_id,
                "help_id": help_id - 1,  # -1 represents no help
                "obstacles_to_exclude": [] if help_id == 0 else [help_id],
                "initial_positions": positions_to_objects,
                "total_reward": None,
                "total_discounted_reward": None,
                "run_time": None,
                "task_progress": {
                    "red_cup_full": None,
                    "blue_cup_full": None,
                    "red_cup_in_position": None,
                    "blue_cup_in_position": None,
                    "can_in_trash": None,
                    "terminated": None,
                    "episode_length": None
                }
            })
            experiment_id += 1

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save configurations
    with open(output_file, 'w') as f:
        json.dump({"metadata": metadata, "experiments": experiments}, f, indent=2)


def save_intermediate_results(exp_id: int, help_id: int, results: dict, config_file: str):
    """Save detailed experiment results to a separate file."""
    # Create intermediate_states directory next to the config file
    config_path = Path(config_file)
    results_dir = config_path.parent / "intermediate_states"
    results_dir.mkdir(exist_ok=True)

    filename = results_dir / f"{exp_id}_{help_id}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(results, f)


def run_experiment_from_config(exp: dict, params: dict) -> tuple[int, Optional[dict]]:
    """Run a single experiment from configuration."""
    try:
        # Set random seeds for this experiment
        seed = params.get('seed', None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Load appropriate PRMs based on help action
        prm = load_prm(obstacles_to_exclude=exp['obstacles_to_exclude'], constrained=False)
        prm_y_up = load_prm(obstacles_to_exclude=exp['obstacles_to_exclude'], constrained=True)

        # Create initial state
        init_state = State(
            robot_cfg_id=prm.home_cfg_idx,
            positions_to_objects=exp['initial_positions'].copy(),
            held_object=None,
            red_cup_full=False,
            blue_cup_full=False,
            can_in_trash=False
        )

        # Initialize problem
        problem = POMProblem(prm, prm_y_up, init_state=init_state)

        # Setup POMCP parameters
        action_prior = POMActionPrior(
            prm, prm_y_up,
            problem.agent.policy_model.vid_to_sensing_config_name
        )
        rollout_policy = POMRolloutPolicy(
            prm, prm_y_up,
            problem.agent.policy_model.vid_to_sensing_config_name,
            action_prior
        )

        pomcp_params = {
            'num_sims': params['n_sims'],
            'max_depth': params['max_depth'],
            'discount_factor': params['discount'],
            'rollout_policy': rollout_policy,
            'action_prior': action_prior
        }

        # Run experiment
        results = run_pomcp_manipulation_instance(
            problem=problem,
            pomcp_params=pomcp_params,
            max_steps=params.get('max_steps', 50),
            verbose=params.get('verbose', 0)
        )

        save_intermediate_results(exp['experiment_id'], exp['help_id'], results, params['config_file'])

        return exp['experiment_id'], results

    except Exception as e:
        print(f"~~~~~~~~~~ Error in experiment {exp['experiment_id']}: {str(e)}")
        return exp['experiment_id'], None


def run_experiment_wrapper(args):
    exp, params = args
    exp_id, result = run_experiment_from_config(exp, params)
    return exp_id, result  # result is already the dictionary we want


@app.command()
def run_experiments_from_file_parallel(
        config_file: str = typer.Option(..., help="Path to configuration file"),
        n_processes: int = typer.Option(2, help="Number of parallel processes"),
        n_sims: int = typer.Option(5000, help="Number of simulations for POMCP"),
        max_steps: int = typer.Option(50, help="Maximum steps per experiment"),
        max_depth: int = typer.Option(30, help="Maximum depth for POMCP"),
        discount: float = typer.Option(0.98, help="Discount factor"),
        verbose: int = typer.Option(0, help="Verbosity level"),
        batch_size: int = typer.Option(200, help="Number of experiments per batch"),
        n_experiment_repeats: int = typer.Option(3, help="Number of times to repeat each experiment"),
        base_seed: int = typer.Option(42, help="Base seed for random number generation")
):
    """Run experiments from configuration file in parallel."""

    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

    problem_params = {
        'n_sims': n_sims,
        'max_steps': max_steps,
        'max_depth': max_depth,
        'discount': discount,
        'verbose': verbose,
        'config_file': config_file
    }

    # Initialize experiment results if needed
    for exp in config['experiments']:
        if exp['total_reward'] is None:
            exp['total_reward'] = []
            exp['total_discounted_reward'] = []
            exp['run_time'] = []
            exp['task_progress'] = {k: [] for k in exp['task_progress'].keys()}

    # Get experiments that need more runs
    experiments_to_run = [
        exp for exp in config['experiments']
        if len(exp['total_reward']) < n_experiment_repeats
    ]

    total_runs = sum(
        n_experiment_repeats - len(exp['total_reward'])
        for exp in experiments_to_run
    )

    print(f"~~~~~~~ Total runs needed: {total_runs}")
    if total_runs == 0:
        return

    n_done = 0
    for batch_start in range(0, len(experiments_to_run), batch_size):
        t_start = time.time()
        batch = experiments_to_run[batch_start:min(batch_start + batch_size, len(experiments_to_run))]

        # Create all needed runs for this batch
        experiment_args = []
        for exp in batch:
            runs_needed = n_experiment_repeats - len(exp['total_reward'])
            for run_idx in range(runs_needed):
                # Create unique seed for this experiment run
                exp_seed = base_seed + exp['experiment_id'] * n_experiment_repeats + run_idx
                params = problem_params.copy()
                params['seed'] = exp_seed
                experiment_args.append((exp, params))

        print(f"Starting batch with {len(experiment_args)} runs")

        with mp.Pool(n_processes) as pool:
            for exp_id, result in pool.imap_unordered(run_experiment_wrapper, experiment_args):
                if result is not None:  # result is now directly the dictionary from run_pomcp_manipulation_instance
                    # Update experiment results in config
                    for exp in config['experiments']:
                        if exp['experiment_id'] == exp_id:
                            exp['total_reward'].append(result['total_reward'])
                            exp['total_discounted_reward'].append(result['total_discounted_reward'])
                            exp['run_time'].append(time.time() - t_start)

                            for k, v in result['final_state_info'].items():
                                exp['task_progress'][k].append(v)
                            exp['task_progress']['terminated'].append(result['terminated'])
                            exp['task_progress']['episode_length'].append(result['num_steps'])
                            break

                    # Save after each successful run
                    with open(config_file, 'w') as f:
                        json.dump(config, f, indent=2)

                    print(f"\033[94m~~~~  Progress: {n_done + 1}/{total_runs} runs - Completed experiment {exp_id} \033[0m")
                else:
                    print(f"\033[91m~~~~  Progress: {n_done + 1}/{total_runs} runs - Failed experiment {exp_id}\033[0m")

                n_done += 1

        print(f"Batch completed in {time.time() - t_start:.2f} seconds")
        gc.collect()
        time.sleep(1)

    print("All experiments completed")


if __name__ == "__main__":
    app()