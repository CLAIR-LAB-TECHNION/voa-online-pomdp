import gc
import random
import platform
import multiprocessing as mp
import sys

import numpy as np
import pomdp_py
import json
from rocksample_experiments.help_actions import push_rocks
from rocksample_experiments.preferred_actions import CustomRSPolicyModel, RSActionPrior
from rocksample_experiments.rocksample_problem import RockSampleProblem, State
import typer
from typing import Optional
import os
import time

app = typer.Typer()


def run_pomcp_rocksample_instance(
        problem: RockSampleProblem,
        pomcp_params: dict,
        max_steps: int = 100,
        verbose: int = 0,
) -> tuple[float, float, float]:
    """
    Run a single RockSample experiment with given problem instance and POMCP parameters.

    Args:
        problem: Initialized RockSampleProblem instance
        pomcp_params: Dictionary containing POMCP configuration parameters
        max_steps: Maximum number of steps to run
        verbose: Verbosity level (0: none, 1: summary, 2+: detailed)

    Returns:
        tuple[float, float, float]: (total_reward, total_discounted_reward, run_time)
    """
    start_time = time.time()

    # Initialize POMCP planner
    pomcp = pomdp_py.POMCP(
        num_sims=pomcp_params.get('num_sims', 2000),
        max_depth=pomcp_params.get('max_depth', 20),
        discount_factor=pomcp_params.get('discount_factor', 0.95),
        exploration_const=pomcp_params.get('exploration_const', 5),
        action_prior=pomcp_params.get('action_prior', None),
        rollout_policy=pomcp_params.get('rollout_policy', problem.agent.policy_model),
        show_progress=verbose > 1
    )

    if verbose > 0:
        print("Initial state:")
        problem.print_state()

    # Run simulation
    total_reward = 0
    total_discounted_reward = 0
    gamma = 1.0
    discount = pomcp_params.get('discount_factor', 0.95)

    for step in range(max_steps):
        if verbose > 1:
            print(f"\nStep {step + 1}")

        # Plan and execute action
        action = pomcp.plan(problem.agent)
        reward = problem.env.state_transition(action, execute=True)

        # Get observation and update agent
        observation = problem.env.provide_observation(
            problem.agent.observation_model,
            action
        )
        problem.agent.update_history(action, observation)
        pomcp.update(problem.agent, action, observation)

        # Update rewards
        total_reward += reward
        total_discounted_reward += reward * gamma
        gamma *= discount

        if verbose > 1:
            print(f"Action: {action}")
            print(f"Observation: {observation}")
            print(f"Reward: {reward}")
            print(f"Total Reward: {total_reward}")
            print(f"Total Discounted Reward: {total_discounted_reward}")
            problem.print_state()

        # Check if reached exit
        if problem.in_exit_area(problem.env.state.position):
            if verbose > 1:
                print("\nReached exit area! Terminating...")
            break

    run_time = time.time() - start_time

    if verbose > 0:
        print(f"\nExperiment completed:")
        print(f"Total Reward: {total_reward}")
        print(f"Total Discounted Reward: {total_discounted_reward}")
        print(f"Run Time: {run_time:.2f} seconds")

    return total_reward, total_discounted_reward, run_time


@app.command()
def run_rocksample_experiment(
        grid_size: int = typer.Option(11, help="Size of the grid (nxn)"),
        num_rocks: int = typer.Option(11, help="Number of rocks in the environment"),
        max_steps: int = typer.Option(100, help="Maximum number of steps to run"),
        n_sims: int = typer.Option(2000, help="Number of simulations for POMCP planner"),
        max_depth: int = typer.Option(20, help="Maximum depth for POMCP planner"),
        discount: float = typer.Option(0.95, help="Discount factor"),
        exploration_const: float = typer.Option(5.0, help="Exploration constant for POMCP"),
        half_efficiency_dist: float = typer.Option(20.0, help="Half efficiency distance for sensing"),
        num_particles: int = typer.Option(1000, help="Number of particles for belief representation"),
        use_preferred_actions: bool = typer.Option(True, help="Whether to use preferred actions in planning"),
        preferred_actions_v_init: float = typer.Option(10.0, help="Initial value for preferred actions"),
        preferred_actions_n_visits_init: int = typer.Option(10, help="Initial visit count for preferred actions"),
        seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
        verbose: int = typer.Option(1, help="Verbosity level (0: none, 1: summary, 2+: detailed)"),
        results_dir: str = typer.Option("", help="Directory to save results")
):
    """
    Run a single RockSample experiment with specified parameters.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Set up results directory
    if results_dir == "":
        results_dir = os.path.join(os.path.dirname(__file__), "experiments")

    datetime_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(results_dir, datetime_stamp)
    os.makedirs(experiment_dir, exist_ok=True)

    # Generate initial state and belief
    init_state, rock_locs = RockSampleProblem.generate_instance(grid_size, num_rocks)

    with open(f'rocksample_experiments/configurations/particles_k{num_rocks}.json', 'r') as f:
        rocktypes = json.load(f)
        if len(rocktypes) < num_particles:
            ValueError("Not enough particles in file")
    particles = [State(position=init_state.position, rocktypes=rocktypes[i], terminal=False)
                 for i in range(num_particles)]
    init_belief = pomdp_py.Particles(particles)

    # Set up action prior if using preferred actions
    action_prior = None
    rollout_policy = None
    if use_preferred_actions:
        action_prior = RSActionPrior(
            grid_size,
            num_rocks,
            rock_locs,
            v_init=preferred_actions_v_init,
            n_visits_init=preferred_actions_n_visits_init
        )
        rollout_policy = CustomRSPolicyModel(grid_size, num_rocks, action_prior)

    # Create problem instance
    problem = RockSampleProblem(
        n=grid_size,
        k=num_rocks,
        init_state=init_state,
        rock_locs=rock_locs,
        init_belief=init_belief,
        half_efficiency_dist=half_efficiency_dist
    )

    # rocks_to_push = {0: (0, 1), 7: (1, 1)}
    # problem_help, budget_used = push_rocks(problem, rocks_to_push)
    # problem = problem_help

    # Set up POMCP parameters
    pomcp_params = {
        'num_sims': n_sims,
        'max_depth': max_depth,
        'discount_factor': discount,
        'exploration_const': exploration_const,
        'action_prior': action_prior,
        'rollout_policy': rollout_policy if rollout_policy is not None else problem.agent.policy_model,
    }

    # Run experiment
    total_reward, total_discounted_reward, time_compute = run_pomcp_rocksample_instance(
        problem=problem,
        pomcp_params=pomcp_params,
        max_steps=max_steps,
        verbose=verbose,
    )

    # Save results
    results = {
        'parameters': {
            'grid_size': grid_size,
            'num_rocks': num_rocks,
            'max_steps': max_steps,
            'n_sims': n_sims,
            'max_depth': max_depth,
            'discount': discount,
            'exploration_const': exploration_const,
            'half_efficiency_dist': half_efficiency_dist,
            'num_particles': num_particles,
            'use_preferred_actions': use_preferred_actions,
            'preferred_actions_v_init': preferred_actions_v_init,
            'preferred_actions_n_visits_init': preferred_actions_n_visits_init,
            'seed': seed
        },
        'results': {
            'total_reward': total_reward,
            'total_discounted_reward': total_discounted_reward
        }
    }

    with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    if verbose > 0:
        print(f"\nResults saved to: {experiment_dir}")
        print(f"Total Reward: {total_reward}")
        print(f"Total Discounted Reward: {total_discounted_reward}")


def run_experiment_from_config(experiment_config: dict, problem_params: dict) -> tuple[
    Optional[float], Optional[float], Optional[float]]:  # Added Optional[float] for run_time
    """
    Run a single experiment using configuration from the JSON file.

    Args:
        experiment_config: Dictionary containing experiment configuration
        problem_params: Optional override parameters for the experiment

    Returns:
        tuple[Optional[float], Optional[float], Optional[float]]: (total_reward, total_discounted_reward, run_time),
                                                               returns (None, None, None) if particle deprivation occurs
    """
    try:
        # Redirect stdout to devnull in child processes
        if mp.current_process().name != 'MainProcess':
            sys.stdout = open(os.devnull, 'w')

        # Convert string tuples in rock_locations back to actual tuples
        rock_locs = {}
        for loc_str, rock_id in experiment_config['rock_locations'].items():
            loc = eval(loc_str)  # Safe here as we trust our config file
            rock_locs[loc] = rock_id

        # Create initial state
        init_state = State(
            position=tuple(experiment_config['rover_position']),
            rocktypes=tuple(experiment_config['rock_types']),
            terminal=False
        )

        # Create initial belief
        with open(f'rocksample_experiments/configurations/particles_k{len(rock_locs)}.json', 'r') as f:
            rocktypes = json.load(f)
            if len(rocktypes) < problem_params['num_particles']:
                ValueError("Not enough particles in file")
        particles = [State(position=init_state.position, rocktypes=rocktypes[i], terminal=False)
                     for i in range(problem_params['num_particles'])]
        init_belief = pomdp_py.Particles(particles)

        # Create base problem instance
        problem = RockSampleProblem(
            n=problem_params.get('grid_size', 11),
            k=len(rock_locs),
            init_state=init_state,
            rock_locs=rock_locs,
            init_belief=init_belief,
            half_efficiency_dist=problem_params['half_efficiency_dist']
        )

        # Apply help actions if present
        if experiment_config['help_actions']:
            rocks_to_push = {int(k): tuple(v) for k, v in experiment_config['help_actions'].items()}
            problem, push_budget_used = push_rocks(problem, rocks_to_push)

        # Setup POMCP parameters
        action_prior = None
        rollout_policy = None
        if problem_params['use_preferred_actions']:
            action_prior = RSActionPrior(
                problem_params.get('grid_size', 11),
                len(rock_locs),
                rock_locs,
                v_init=problem_params['preferred_actions_v_init'],
                n_visits_init=problem_params['preferred_actions_n_visits_init']
            )
            rollout_policy = CustomRSPolicyModel(problem_params.get('grid_size', 11),
                                                 len(rock_locs),
                                                 action_prior)

        pomcp_params = {
            'num_sims': problem_params['n_sims'],
            'max_depth': problem_params['max_depth'],
            'discount_factor': problem_params['discount'],
            'exploration_const': problem_params['exploration_const'],
            'action_prior': action_prior,
            'rollout_policy': rollout_policy if rollout_policy is not None else problem.agent.policy_model,
        }

        # Run experiment
        return run_pomcp_rocksample_instance(
            problem=problem,
            pomcp_params=pomcp_params,
            max_steps=problem_params.get('max_steps', 100),
            verbose=problem_params.get('verbose', 0)
        )

    except ValueError as e:
        if "Particle deprivation" in str(e):
            return None, None, None
        raise
    except Exception as e:
        print(f"Unexpected error in experiment {experiment_config['experiment_id']}: {str(e)}")
        return None, None, None

@app.command()
def run_experiments_from_file(
        config_file: str = typer.Option(..., help="Path to configuration file"),
        max_steps: int = typer.Option(100, help="Maximum steps per experiment"),
        n_sims: int = typer.Option(2000, help="Number of simulations for POMCP planner"),
        num_particles: int = typer.Option(1000, help="Number of particles for belief representation"),
        max_depth: int = typer.Option(20, help="Maximum depth for POMCP planner"),
        verbose: int = typer.Option(0, help="Verbosity level")
):
    """
    Run experiments from configuration file and save results back to the same file.
    """
    # Read configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract problem parameters from metadata
    problem_params = {
        'grid_size': config['metadata']['grid_size'],
        'max_steps': max_steps,
        'n_sims': n_sims,
        'num_particles': num_particles,
        'max_depth': max_depth,
        'verbose': verbose,
        # Add other parameters with default values
        'half_efficiency_dist': 20.0,
        'discount': 0.95,
        'exploration_const': 5.0,
        'use_preferred_actions': True,
        'preferred_actions_v_init': 10.0,
        'preferred_actions_n_visits_init': 10
    }

    # Run experiments
    for experiment in config['experiments']:
        # Skip if already done
        if experiment['total_reward'] is not None:
            continue

        print(f"Running experiment {experiment['experiment_id']}/{len(config['experiments'])}")

        total_reward, total_discounted_reward, time_compute = run_experiment_from_config(
            experiment,
            problem_params
        )

        # Update results
        experiment['total_reward'] = total_reward
        experiment['total_discounted_reward'] = total_discounted_reward

        # Save after each experiment
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Completed experiment {experiment['experiment_id']}: "
              f"reward={total_reward}, discounted_reward={total_discounted_reward}")


def run_experiment_wrapper(args):
    exp, params = args
    result = run_experiment_from_config(exp, params)
    return exp['experiment_id'], result


@app.command()
def run_experiments_from_file_parallel(
        config_file: str = typer.Option(..., help="Path to configuration file"),
        n_processes: int = typer.Option(2, help="Number of parallel processes"),
        max_steps: int = typer.Option(100, help="Maximum steps per experiment"),
        n_sims: int = typer.Option(2000, help="Number of simulations for POMCP planner"),
        num_particles: int = typer.Option(1000, help="Number of particles for belief representation"),
        max_depth: int = typer.Option(20, help="Maximum depth for POMCP planner"),
        verbose: int = typer.Option(0, help="Verbosity level"),
        batch_size: int = typer.Option(50, help="Number of experiments to run before recreating worker pool"),
        n_experiment_repeats: int = typer.Option(3, help="Number of times to repeat each experiment")
):
    """
    Run experiments from configuration file in parallel and save results back to the same file.
    """
    # Set up multiprocessing
    if platform.system() == 'Linux':
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

    # Read configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract problem parameters from metadata and CLI args
    problem_params = {
        'grid_size': config['metadata']['grid_size'],
        'max_steps': max_steps,
        'n_sims': n_sims,
        'num_particles': num_particles,
        'max_depth': max_depth,
        'verbose': verbose,
        'half_efficiency_dist': 20.0,
        'discount': 0.95,
        'exploration_const': 5.0,
        'use_preferred_actions': True,
        'preferred_actions_v_init': 10.0,
        'preferred_actions_n_visits_init': 10
    }

    # Convert single numbers to lists if needed
    for exp in config['experiments']:
        if exp['total_reward'] is not None:
            if not isinstance(exp['total_reward'], list):
                exp['total_reward'] = [exp['total_reward']]
            if not isinstance(exp['total_discounted_reward'], list):
                exp['total_discounted_reward'] = [exp['total_discounted_reward']]
            if not isinstance(exp.get('run_time'), list):
                exp['run_time'] = [exp.get('run_time')]

    # Get experiments that need more runs
    experiments_to_run = [exp for exp in config['experiments']
                          if exp['total_reward'] is None
                          or len(exp['total_reward']) < n_experiment_repeats]

    # Count total runs needed
    total_runs = sum(n_experiment_repeats - (len(exp['total_reward']) if exp['total_reward'] is not None else 0)
                     for exp in experiments_to_run)

    print(f"Total runs needed: {total_runs}")
    if total_runs == 0:
        print("No more runs needed")
        return

    n_done = 0
    for batch_start in range(0, len(experiments_to_run), batch_size):
        t_start = time.time()
        batch = experiments_to_run[batch_start:min(batch_start + batch_size, len(experiments_to_run))]

        # Create all needed runs for this batch
        experiment_args = []
        for exp in batch:
            runs_needed = n_experiment_repeats
            if exp['total_reward'] is not None:
                runs_needed -= len(exp['total_reward'])
            experiment_args.extend([(exp, problem_params) for _ in range(runs_needed)])

        print(f"Starting batch with {len(experiment_args)} runs")

        # Dictionary to collect results by experiment ID
        results_by_exp = {}

        with mp.Pool(n_processes) as pool:
            for exp_id, result in pool.imap_unordered(run_experiment_wrapper, experiment_args):
                if result[0] is not None:
                    print(f"Completed run for experiment {exp_id}: "
                          f"reward={result[0]}, discounted_reward={result[1]}, "
                          f"run_time={result[2]:.2f}s")

                    # Find and update experiment in config
                    for exp in config['experiments']:
                        if exp['experiment_id'] == exp_id:
                            if exp['total_reward'] is None:
                                exp['total_reward'] = []
                                exp['total_discounted_reward'] = []
                                exp['run_time'] = []
                            exp['total_reward'].append(result[0])
                            exp['total_discounted_reward'].append(result[1])
                            exp['run_time'].append(result[2])
                            break

                    # Save after each successful run
                    with open(config_file, 'w') as f:
                        json.dump(config, f, indent=2)
                else:
                    print(f"Run failed for experiment {exp_id}")

                n_done += 1
                print(f"Progress: {n_done}/{total_runs} runs")

        print(f"---Batch completed in {time.time() - t_start:.2f} seconds")
        gc.collect()
        time.sleep(10)

    print("All experiments completed")


if __name__ == "__main__":
    app()
