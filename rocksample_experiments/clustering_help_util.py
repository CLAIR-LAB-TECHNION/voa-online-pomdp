import typer
import json
import os
from typing import Dict, List, Tuple, Set, Optional
import random
from itertools import combinations
from rocksample_experiments.rocksample_problem import RockSampleProblem, RockType, State
from rocksample_experiments.help_actions import is_legal_rock_push, push_rocks

app = typer.Typer()


def get_cluster_positions(n: int) -> List[Tuple[int, int]]:
    """Return the three fixed cluster positions at column 2"""
    return [
        (2, n // 2),    # middle
        (2, n // 4),    # upper quarter
        (2, 3 * n // 4)  # lower quarter
    ]


def get_available_positions_around(center: Tuple[int, int], n: int, occupied_positions: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Get available positions around center point, considering grid bounds and occupied positions"""
    x, y = center
    relative = [
        (0, 0),  # center
        (-1, 0),  # left
        (0, -1),  # up (negative y)
        (0, 1),  # down (positive y)
        (1, 0),  # right
        (-1, -1),  # diagonal up-left
        (-1, 1),  # diagonal down-left
        (1, -1),  # diagonal up-right
        (1, 1)  # diagonal down-right
    ]

    positions = []
    for dx, dy in relative:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < n and 0 <= new_y < n and (new_x, new_y) not in occupied_positions:
            positions.append((new_x, new_y))

    return positions

def get_cluster_center(positions: List[Tuple[int, int]], n: int) -> Tuple[int, int]:
    """
    Given positions of rocks in a cluster, determine which cluster center they belong to.
    """
    cluster_centers = get_cluster_positions(n)
    center_pos = min(positions, key=lambda pos: min(
        abs(pos[0] - c[0]) + abs(pos[1] - c[1])
        for c in cluster_centers
    ))
    return min(cluster_centers, key=lambda c:
    abs(c[0] - center_pos[0]) + abs(c[1] - center_pos[1]))


def config_to_action_tuple(config: Dict, n: int) -> Tuple:
    """
    Convert config to tuple for comparison.
    Two configs are the same if they move the same rocks to the same cluster.
    """
    rock_ids = sorted(config.keys())
    positions = [config[rid] for rid in rock_ids]
    cluster_center = get_cluster_center(positions, n)

    # The identifying tuple is just (sorted rock ids, cluster center)
    return (tuple(rock_ids), cluster_center)


def is_subset_config(config1: Dict, config2: Dict, n: int) -> bool:
    """
    Check if config1 is a subset of config2.
    Configs are considered for the same action if they move rocks to the same cluster.
    """
    rocks1 = set(config1.keys())
    rocks2 = set(config2.keys())
    if len(rocks1) >= len(rocks2):
        return False

    # If it's a subset of rocks, check if they're going to the same cluster
    if rocks1.issubset(rocks2):
        positions1 = [config1[rid] for rid in config1.keys()]
        positions2 = [config2[rid] for rid in rocks1]  # Only positions of common rocks
        return get_cluster_center(positions1, n) == get_cluster_center(positions2, n)
    return False


def generate_cluster_configuration(problem: RockSampleProblem, rock_ids: List[int], cluster_center: Tuple[int, int]) -> \
Optional[Dict]:
    """
    Generate a configuration that clusters given rocks around the center position.
    Returns None if valid configuration cannot be created.
    """
    n = problem.n
    occupied_positions = set(problem.rock_locs.keys()) - {pos for pos, rid in problem.rock_locs.items() if
                                                          rid in rock_ids}

    # Try to place rocks in order (random order each time)
    rock_order = list(rock_ids)
    random.shuffle(rock_order)

    config = {}
    for rock_id in rock_order:
        # Get available positions
        positions = get_available_positions_around(cluster_center, n, occupied_positions)
        if not positions:
            return None

        # Choose position closest to center
        pos = min(positions, key=lambda p: abs(p[0] - cluster_center[0]) + abs(p[1] - cluster_center[1]))
        config[rock_id] = pos
        occupied_positions.add(pos)

    return config


def remove_duplicate_and_subset_configs(configs: List[Dict], n: int) -> List[Dict]:
    """Remove duplicate configurations and subset configurations"""
    # First remove duplicates
    unique_configs = []
    seen_actions = set()

    for config in configs:
        action_tuple = config_to_action_tuple(config, n)
        if action_tuple not in seen_actions:
            unique_configs.append(config)
            seen_actions.add(action_tuple)

    # Then remove subsets
    final_configs = []
    for i, config in enumerate(unique_configs):
        is_subset = False
        for j, other_config in enumerate(unique_configs):
            if i != j and is_subset_config(config, other_config, n):
                is_subset = True
                break
        if not is_subset:
            final_configs.append(config)

    return final_configs


def generate_cluster_help_configs(problem: RockSampleProblem, window_size: int, max_rocks_per_cluster: int) -> List[
    Dict]:
    """Generate all help configurations for clustering rocks near exit"""
    n = problem.n
    clusters = get_cluster_positions(n)
    configs = []

    # Get all lxl windows
    for x in range(n - window_size + 1):
        for y in range(n - window_size + 1):
            # Find rocks in this window
            rocks_in_window = []
            for pos, rock_id in problem.rock_locs.items():
                if x <= pos[0] < x + window_size and y <= pos[1] < y + window_size:
                    rocks_in_window.append(rock_id)

            # Generate combinations of rocks
            num_rocks = min(len(rocks_in_window), max_rocks_per_cluster)
            if num_rocks > 0:  # Only if we have any rocks
                for rock_combo in combinations(rocks_in_window, num_rocks):
                    # For each cluster position
                    for cluster_pos in clusters:
                        config = generate_cluster_configuration(problem, list(rock_combo), cluster_pos)
                        if config is not None:
                            configs.append(config)

    # Remove duplicates and subsets
    return remove_duplicate_and_subset_configs(configs, n)


def convert_to_movement_config(problem: RockSampleProblem, cluster_config: Dict) -> Dict:
    """Convert cluster configuration to movement configuration (dx, dy format)"""
    movements = {}
    rock_positions = {rock_id: pos for pos, rock_id in problem.rock_locs.items()}

    for rock_id, new_pos in cluster_config.items():
        orig_pos = rock_positions[rock_id]
        dx = new_pos[0] - orig_pos[0]
        dy = new_pos[1] - orig_pos[1]
        movements[rock_id] = (dx, dy)

    return movements


@app.command()
def test_cluster_help_generation(
        n: int = typer.Option(..., help="Grid size"),
        k: int = typer.Option(..., help="Number of rocks"),
        w: int = typer.Option(5, help="Window size"),
        max_rocks: int = typer.Option(3, help="Maximum rocks per cluster")
):
    """Test help config generation with given parameters"""

    # Create test problem instance
    init_state, rock_locs = RockSampleProblem.generate_instance(n, k)
    problem = RockSampleProblem(
        n=n,
        k=k,
        init_state=init_state,
        rock_locs=rock_locs,
        init_belief=None
    )

    # Generate configs
    cluster_configs = generate_cluster_help_configs(problem, w, max_rocks)
    movement_configs = [convert_to_movement_config(problem, config) for config in cluster_configs]

    # Print results and statistics
    print(f"\nParameters:")
    print(f"Grid size: {n}x{n}")
    print(f"Number of rocks: {k}")
    print(f"Window size: {w}")
    print(f"Max rocks per cluster: {max_rocks}")
    print(f"\nGenerated {len(cluster_configs)} unique help configurations")

    # Print rock distribution statistics
    rocks_used = [len(config) for config in cluster_configs]
    if rocks_used:
        print(f"\nRocks per configuration:")
        print(f"Min: {min(rocks_used)}")
        print(f"Max: {max(rocks_used)}")
        print(f"Average: {sum(rocks_used) / len(rocks_used):.2f}")

    # Print a few example configs (first 5)
    print("\nInitial state:")
    problem.print_state()

    print("\nSample configurations (first 5):")
    for i, (cluster_config, movement_config) in enumerate(zip(cluster_configs[:5], movement_configs[:5])):
        print(f"\nConfig {i}:")
        # Create new problem instance with rocks moved according to config
        new_problem, _ = push_rocks(problem, movement_config)
        new_problem.print_state()

    # Validate all configurations are legal
    print("\nValidating configurations...")
    all_legal = True
    for i, movement_config in enumerate(movement_configs):
        if not is_legal_rock_push(problem, movement_config):
            print(f"Warning: Configuration {i} is not legal!")
            all_legal = False

    if all_legal:
        print("All configurations are legal!")


if __name__ == "__main__":
    app()