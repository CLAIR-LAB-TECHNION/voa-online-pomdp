import typer
import json
import os
from typing import Dict, List, Tuple, Optional
import random
from rocksample_experiments.rocksample_problem import RockSampleProblem, RockType, State
from rocksample_experiments.help_actions import is_legal_rock_push
from rocksample_experiments.clustering_help_util import (
    generate_cluster_help_configs,
    convert_to_movement_config
)

app = typer.Typer()


def generate_help_config(problem: RockSampleProblem, budget: int, max_rocks_to_push: int) -> dict:
    """
    Generate a single valid help configuration within budget by:
    1. Randomly selecting number of rocks to push (1 to max_rocks_to_push)
    2. Randomly dividing budget between selected rocks
    3. For each rock, generating dx, dy that sum to its allocated budget
    """
    while True:  # Main generation loop
        # Random number of rocks to push (1 to max_rocks_to_push)
        max_rocks = min(len(problem.rock_locs), max_rocks_to_push)
        num_rocks_to_push = random.randint(1, max_rocks)
        rocks_to_push = random.sample(list(problem.rock_locs.values()), num_rocks_to_push)
        rock_positions = {rock_id: pos for pos, rock_id in problem.rock_locs.items()}

        # Randomly divide budget between rocks
        rock_budgets = {}
        remaining_budget = budget

        for rock_id in rocks_to_push[:-1]:
            if remaining_budget <= 0:
                break
            rock_budgets[rock_id] = random.randint(1, remaining_budget)
            remaining_budget -= rock_budgets[rock_id]

        if remaining_budget > 0:
            rock_budgets[rocks_to_push[-1]] = remaining_budget

        help_config = {}
        total_used_budget = 0
        config_valid = True  # Flag to track if we need to retry entire configuration

        # For each rock, generate valid dx, dy considering grid boundaries
        for rock_id, rock_budget in rock_budgets.items():
            current_pos = rock_positions[rock_id]

            # Calculate maximum possible movements considering grid boundaries
            max_dx_neg = -current_pos[0]
            max_dx_pos = problem.n - 1 - current_pos[0]
            max_dy_neg = -current_pos[1]
            max_dy_pos = problem.n - 1 - current_pos[1]

            # Try to generate valid movement within boundaries
            valid_move_found = False
            attempts = 0
            while attempts < 10:
                dx_budget = random.randint(0, rock_budget)
                dy_budget = rock_budget - dx_budget

                dx = dx_budget * random.choice([-1, 1])
                dy = dy_budget * random.choice([-1, 1])

                if max_dx_neg <= dx <= max_dx_pos and max_dy_neg <= dy <= max_dy_pos:
                    help_config[rock_id] = (dx, dy)
                    total_used_budget += abs(dx) + abs(dy)
                    valid_move_found = True
                    break

                attempts += 1

            if not valid_move_found:
                config_valid = False  # Mark configuration as invalid
                break  # Break out of rock loop

        # Only check final conditions if all moves were valid
        if config_valid and total_used_budget == budget:
            if is_legal_rock_push(problem, help_config):
                return help_config


def generate_help_config_set(problem: RockSampleProblem, n_configs: int, budget: int, max_rocks_to_push: int) -> List[
    Dict]:
    """Generate n_configs unique valid help configurations"""
    configs = set()
    while len(configs) < n_configs:
        config = generate_help_config(problem, budget, max_rocks_to_push)
        # Convert dict to tuple for hashing
        config_tuple = tuple(sorted((k, v[0], v[1]) for k, v in config.items()))
        configs.add(config_tuple)

    # Convert back to list of dicts
    return [dict((k, (dx, dy)) for k, dx, dy in config) for config in configs]


def test_help_config_set(problem: RockSampleProblem, n_configs: int, budget: int, max_rocks_to_push: int):
    """
    Test help config set generation by validating:
    1. All configs are unique
    2. All configs use exactly the budget amount
    3. All configs are legal according to is_legal_rock_push
    """
    help_configs = generate_help_config_set(problem, n_configs, budget, max_rocks_to_push)

    # Test we got correct number of configs
    assert len(help_configs) == n_configs, f"Generated {len(help_configs)} configs instead of {n_configs}"

    # Test uniqueness
    config_tuples = [tuple(sorted((k, v[0], v[1]) for k, v in config.items())) for config in help_configs]
    assert len(set(config_tuples)) == n_configs, "Not all configurations are unique"

    for idx, config in enumerate(help_configs):
        # Test budget use
        total_budget_used = sum(abs(dx) + abs(dy) for dx, dy in config.values())
        assert total_budget_used == budget, f"Config {idx} uses {total_budget_used} budget instead of {budget}"

        # Test legality
        assert is_legal_rock_push(problem, config), f"Config {idx} is not legal"

        # Test number of rocks pushed doesn't exceed max
        assert len(
            config) <= max_rocks_to_push, f"Config {idx} pushes {len(config)} rocks, exceeding max of {max_rocks_to_push}"

        print(f"Config {idx}: {config}")

    print(f"All tests passed! Generated {n_configs} valid configurations.")


@app.command()
def test_generate_help_config_set():
    n, k = 15, 15
    init_state, rock_locs = RockSampleProblem.generate_instance(n, k)
    problem = RockSampleProblem(
        n=n,
        k=k,
        init_state=init_state,
        rock_locs=rock_locs,
        init_belief=None
    )

    # Test with different parameters
    test_help_config_set(problem, n_configs=100, budget=10, max_rocks_to_push=5)


def generate_single_push_configs(problem: RockSampleProblem, single_push_distance: int) -> List[Dict]:
    """
    Generate help configs where each config pushes one rock by up to single_push_distance steps
    in one of four directions. If maximum distance is illegal, try shorter distances.
    """
    configs = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # N,S,E,W

    for rock_id in problem.rock_locs.values():
        for dx_unit, dy_unit in directions:
            # Try distances from max to 1
            for distance in range(single_push_distance, 0, -1):
                dx = dx_unit * distance
                dy = dy_unit * distance
                config = {rock_id: (dx, dy)}

                # If legal, add this config and move to next direction
                if is_legal_rock_push(problem, config):
                    configs.append(config)
                    break

    return configs


def generate_cluster_configs(problem: RockSampleProblem, max_rocks_per_cluster: int, window_size: int) -> List[Dict]:
    """
    Generate help configs for clustering mode and convert them to movement format.
    Returns list of movement configs {rock_id: (dx, dy)}
    """
    cluster_configs = generate_cluster_help_configs(problem, window_size, max_rocks_per_cluster)
    return [convert_to_movement_config(problem, config) for config in cluster_configs]


@app.command()
def generate_initial_belief(
        k: int = typer.Option(..., help="Number of rocks"),
        num_particles: int = typer.Option(..., help="Number of particles to generate"),
        output_file: str = typer.Option(..., help="Output JSON file path")
):
    """
    Generate initial belief particles and save to file.
    Saved as list which each entry is a particle.
    Each particle is k-length list of rock types.
    """
    particles = []
    for _ in range(num_particles):
        particles.append([RockType.random() for _ in range(k)])

    # Create outputfile dir if not exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(particles, f, indent=2)



@app.command()
def generate_experiment_configs(
        n: int = typer.Option(..., help="Grid size"),
        k: int = typer.Option(..., help="Number of rocks"),
        n_problem_instances: int = typer.Option(..., help="Number of problem instances"),
        n_actual_states: int = typer.Option(..., help="Number of states per instance"),
        help_mode: str = typer.Option("budget", help="Help mode: 'budget', 'single_push', or 'cluster'"),
        # Parameters for budget mode
        n_help_configs: Optional[int] = typer.Option(None, help="Number of help configurations per state (budget mode only)"),
        push_help_budget: Optional[int] = typer.Option(None, help="Budget for push help actions (budget mode only)"),
        max_rocks_to_push: Optional[int] = typer.Option(None, help="Maximum number of rocks to push (budget mode only)"),
        # Parameters for single push mode
        single_push_distance: Optional[int] = typer.Option(None, help="Maximum distance for single rock push (single_push mode only)"),
        # Parameters for cluster mode
        window_size: Optional[int] = typer.Option(None, help="Window size for clustering mode"),
        max_rocks_per_cluster: Optional[int] = typer.Option(None, help="Maximum number of rocks per cluster (cluster mode only)"),
        ###
        max_help_configs: Optional[int] = typer.Option(0, help="Maximum number of help configurations, for all help modes."
                                                               "used for value function learning where we want sparse instances"),
        output_file: str = typer.Option(..., help="Output JSON file path")
):
    """Generate experiment configurations and save to JSON file"""

    # Validate parameters based on mode
    if help_mode not in ["budget", "single_push", "cluster"]:
        raise ValueError("help_mode must be either 'budget', 'single_push', or 'cluster'")

    if help_mode == "budget":
        if any(param is None for param in [n_help_configs, push_help_budget, max_rocks_to_push]):
            raise ValueError("In budget mode, n_help_configs, push_help_budget, and max_rocks_to_push must be provided")
    elif help_mode == "single_push":
        if single_push_distance is None:
            raise ValueError("In single_push mode, single_push_distance must be provided")
    else:  # cluster mode
        if any(param is None for param in [window_size, max_rocks_per_cluster]):
            raise ValueError("In cluster mode, window_size and max_rocks_per_cluster must be provided")

    metadata = {
        "grid_size": n,
        "num_rocks": k,
        "num_problem_instances": n_problem_instances,
        "num_states_per_instance": n_actual_states,
        "help_mode": help_mode
    }

    if help_mode == "budget":
        metadata.update({
            "num_help_configs": n_help_configs,
            "push_help_budget": push_help_budget,
            "max_rocks_to_push": max_rocks_to_push
        })
    elif help_mode == "single_push":
        metadata.update({
            "single_push_distance": single_push_distance
        })
    else:  # cluster mode
        metadata.update({
            "window_size": window_size,
            "max_rocks_per_cluster": max_rocks_per_cluster
        })

    experiments = []
    experiment_id = 0

    for env_idx in range(n_problem_instances):
        init_state, rock_locs = RockSampleProblem.generate_instance(n, k)
        rover_position = init_state.position

        # Create problem instance for help config generation
        problem = RockSampleProblem(
            n=n,
            k=k,
            init_state=init_state,
            rock_locs=rock_locs,
            init_belief=None
        )

        # Generate help configs based on mode
        if help_mode == "budget":
            help_configs = generate_help_config_set(problem, n_help_configs, push_help_budget, max_rocks_to_push)
        elif help_mode == "single_push":
            help_configs = generate_single_push_configs(problem, single_push_distance)
        else:  # cluster mode
            help_configs = generate_cluster_configs(problem, max_rocks_per_cluster, window_size)

        if max_help_configs > 0:
            help_configs = random.sample(help_configs, min(max_help_configs, len(help_configs)))

        # For each instance, generate different states (rock types)
        for state_idx in range(n_actual_states):
            rocktypes = tuple(RockType.random() for _ in range(k))

            # Add experiment with no help
            experiments.append({
                "experiment_id": experiment_id,
                "env_instance_id": env_idx,
                "state_id": state_idx,
                "help_config_id": -1,
                "rover_position": list(rover_position),
                "rock_locations": {str(pos): rock_id for pos, rock_id in rock_locs.items()},
                "rock_types": list(rocktypes),
                "help_actions": {},
                "total_reward": None,
                "total_discounted_reward": None
            })
            experiment_id += 1

            # Add experiments with help configurations
            for help_idx, help_config in enumerate(help_configs):
                experiments.append({
                    "experiment_id": experiment_id,
                    "env_instance_id": env_idx,
                    "state_id": state_idx,
                    "help_config_id": help_idx,
                    "rover_position": list(rover_position),
                    "rock_locations": {str(pos): rock_id for pos, rock_id in rock_locs.items()},
                    "rock_types": list(rocktypes),
                    "help_actions": {str(k): list(v) for k, v in help_config.items()},
                    "total_reward": None,
                    "total_discounted_reward": None
                })
                experiment_id += 1

    # Create outputfile dir if not exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump({"metadata": metadata, "experiments": experiments}, f, indent=2)


@app.command()
def merge_experiment_results(
        file1: str = typer.Option(..., help="First results file path"),
        file2: str = typer.Option(..., help="Second results file path"),
        output_file: str = typer.Option(..., help="Output merged file path")
):
    """Merge two experiment results files"""

    # Load both files
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)

    # Validate metadata matches
    metadata_keys_to_check = [
        "grid_size", "num_rocks", "help_mode", "window_size",
        "max_rocks_per_cluster", "num_states_per_instance"
    ]
    for key in metadata_keys_to_check:
        if data1["metadata"][key] != data2["metadata"][key]:
            raise ValueError(f"Metadata mismatch for {key}")

    # Create new metadata with combined problem instances
    new_metadata = data1["metadata"].copy()
    new_metadata["num_problem_instances"] = (
            data1["metadata"]["num_problem_instances"] +
            data2["metadata"]["num_problem_instances"]
    )

    # Create map of old env_instance_ids to new ones
    max_env_id = max(exp["env_instance_id"] for exp in data1["experiments"])
    env_id_map_2 = {
        old_id: old_id + max_env_id + 1
        for old_id in set(exp["env_instance_id"] for exp in data2["experiments"])
    }

    # Combine experiments, updating IDs for second file
    new_experiments = data1["experiments"].copy()
    next_exp_id = max(exp["experiment_id"] for exp in new_experiments) + 1

    for exp in data2["experiments"]:
        new_exp = exp.copy()
        new_exp["experiment_id"] = next_exp_id
        new_exp["env_instance_id"] = env_id_map_2[exp["env_instance_id"]]
        next_exp_id += 1
        new_experiments.append(new_exp)

    # Validate merged results
    validate_merged_results(new_experiments, new_metadata["num_states_per_instance"])

    # Save merged results
    merged_data = {
        "metadata": new_metadata,
        "experiments": new_experiments
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)


def validate_merged_results(experiments, num_states_per_instance):
    """Validate the merged experiments for consistency"""
    by_env = {}
    for exp in experiments:
        env_id = exp["env_instance_id"]
        if env_id not in by_env:
            by_env[env_id] = []
        by_env[env_id].append(exp)

    for env_id, env_exps in by_env.items():
        # Check rock locations
        rock_locs = env_exps[0]["rock_locations"]
        if not all(exp["rock_locations"] == rock_locs for exp in env_exps):
            raise ValueError(f"Inconsistent rock locations in env_instance {env_id}")

        # Get the actual max help_config_id
        max_help_id = max(exp["help_config_id"] for exp in env_exps)

        # Check help configs are consistent
        help_configs = {}
        for exp in env_exps:
            if exp["help_config_id"] != -1:
                help_id = exp["help_config_id"]
                if help_id in help_configs:
                    if exp["help_actions"] != help_configs[help_id]:
                        raise ValueError(f"Inconsistent help config {help_id} in env_instance {env_id}")
                else:
                    help_configs[help_id] = exp["help_actions"]

        # Check all states for this env exist
        states = set((exp["state_id"], exp["help_config_id"]) for exp in env_exps)
        expected_states = set(
            (state_id, help_id)
            for state_id in range(num_states_per_instance)
            for help_id in range(-1, max_help_id + 1)  # From -1 (no help) to max_help_id
        )
        if not expected_states.issubset(states):
            missing = expected_states - states
            raise ValueError(f"Missing experiments in env_instance {env_id}: {missing}")


@app.command()
def add_states_to_experiments(
        input_file: str = typer.Option(..., help="Input experiment results file"),
        additional_states: int = typer.Option(..., help="Number of additional states to add"),
        output_file: str = typer.Option(..., help="Output file path")
):
    """Add more states to existing experiment configurations"""

    print(f"\nLoading experiments from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Store original data for validation
    original_experiments = data["experiments"]
    original_count = len(original_experiments)
    experiments_with_results = sum(1 for exp in original_experiments
                                   if exp["total_reward"] is not None)

    print(f"\nOriginal experiment stats:")
    print(f"Total experiments: {original_count}")
    print(f"Experiments with results: {experiments_with_results}")

    # Update metadata
    new_metadata = data["metadata"].copy()
    original_states = data["metadata"]["num_states_per_instance"]
    new_metadata["num_states_per_instance"] = original_states + additional_states

    print(f"\nAdding {additional_states} states to each environment instance")
    print(f"States per instance: {original_states} -> {original_states + additional_states}")

    # Group experiments by env_instance_id
    by_env = {}
    for exp in data["experiments"]:
        env_id = exp["env_instance_id"]
        if env_id not in by_env:
            by_env[env_id] = []
        by_env[env_id].append(exp)

    print(f"\nFound {len(by_env)} environment instances")

    # Create new experiments list
    new_experiments = []
    experiment_id = 0
    preserved_results = 0
    new_experiments_added = 0
    total_new_experiments_expected = 0

    # Process each environment instance
    for env_id in sorted(by_env.keys()):
        env_exps = by_env[env_id]
        print(f"\nProcessing environment instance {env_id}:")
        print(f"Original experiments in this instance: {len(env_exps)}")

        # Get environment details from first experiment
        rover_position = env_exps[0]["rover_position"]
        rock_locations = env_exps[0]["rock_locations"]

        # Get help configurations
        help_configs = {}
        for exp in env_exps:
            if exp["help_config_id"] != -1:
                help_configs[exp["help_config_id"]] = exp["help_actions"]

        exps_per_state = len(help_configs) + 1  # +1 for no-help case
        print(f"Found {len(help_configs)} help configurations ({exps_per_state} experiments per state)")

        # Calculate expected new experiments for this env
        env_new_exps = additional_states * exps_per_state
        total_new_experiments_expected += env_new_exps

        # Add existing states first
        for state_idx in range(original_states):
            state_exps = [e for e in env_exps if e["state_id"] == state_idx]
            for exp in state_exps:
                new_exp = exp.copy()
                new_exp["experiment_id"] = experiment_id
                new_experiments.append(new_exp)
                experiment_id += 1
                if exp["total_reward"] is not None:
                    preserved_results += 1

        # Add new states
        for state_idx in range(original_states, original_states + additional_states):
            # Generate new rock types
            rocktypes = tuple(RockType.random() for _ in range(data["metadata"]["num_rocks"]))

            # Add experiment with no help
            new_experiments.append({
                "experiment_id": experiment_id,
                "env_instance_id": env_id,
                "state_id": state_idx,
                "help_config_id": -1,
                "rover_position": rover_position,
                "rock_locations": rock_locations,
                "rock_types": list(rocktypes),
                "help_actions": {},
                "total_reward": None,
                "total_discounted_reward": None
            })
            experiment_id += 1
            new_experiments_added += 1

            # Add experiments with help configurations
            for help_idx in sorted(help_configs.keys()):
                new_experiments.append({
                    "experiment_id": experiment_id,
                    "env_instance_id": env_id,
                    "state_id": state_idx,
                    "help_config_id": help_idx,
                    "rover_position": rover_position,
                    "rock_locations": rock_locations,
                    "rock_types": list(rocktypes),
                    "help_actions": help_configs[help_idx],
                    "total_reward": None,
                    "total_discounted_reward": None
                })
                experiment_id += 1
                new_experiments_added += 1

    # Validate results
    print("\nValidating results:")

    # Check all original results were preserved
    assert preserved_results == experiments_with_results, \
        f"Lost some results! Preserved: {preserved_results}, Original: {experiments_with_results}"
    print("✓ All original results preserved")

    # Validate experiment count
    expected_new_count = original_count + total_new_experiments_expected
    assert len(new_experiments) == expected_new_count, \
        f"Expected {expected_new_count} experiments, got {len(new_experiments)}"
    print("✓ Total experiment count matches expected")

    # Validate experiment IDs are sequential
    exp_ids = [exp["experiment_id"] for exp in new_experiments]
    assert exp_ids == list(range(len(new_experiments))), "Experiment IDs not sequential"
    print("✓ Experiment IDs are sequential")

    # Validate help configs are consistent within each env instance
    for env_id in by_env:
        env_exps = [exp for exp in new_experiments if exp["env_instance_id"] == env_id]
        help_actions = {}
        for exp in env_exps:
            if exp["help_config_id"] != -1:
                help_id = exp["help_config_id"]
                if help_id in help_actions:
                    assert exp["help_actions"] == help_actions[help_id], \
                        f"Inconsistent help config {help_id} in env {env_id}"
                else:
                    help_actions[help_id] = exp["help_actions"]
    print("✓ Help configurations are consistent within environments")

    print("\nSummary:")
    print(f"Original experiments: {original_count}")
    print(f"New experiments added: {new_experiments_added}")
    print(f"Total experiments: {len(new_experiments)}")
    print(f"Results preserved: {preserved_results}")

    # Save to output file
    print(f"\nSaving to: {output_file}")
    output_data = {
        "metadata": new_metadata,
        "experiments": new_experiments
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    app()
