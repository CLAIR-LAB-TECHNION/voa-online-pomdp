from itertools import permutations

import networkx as nx
from rocksample_experiments.rocksample_problem import RockSampleProblem


def manhattan_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_path_value(problem: RockSampleProblem, rock_order, discount_factor: float = 0.95):
    """Compute discounted reward for a specific path through rocks."""
    value = 0
    current_pos = problem.env.state.position
    steps = 0

    # Visit each rock in order
    for rock_pos in rock_order:
        steps += manhattan_dist(current_pos, rock_pos)
        reward = 10 * (discount_factor ** steps)
        value += reward
        steps += 1  # sampling action takes 1 step
        current_pos = rock_pos

    steps += problem.n - current_pos[0]  # horizontal distance to exit
    if steps < problem.n:  # if we can reach exit
        value += 10 * (discount_factor ** steps)  # exit reward

    return value


def get_full_info_value(problem: RockSampleProblem, discount_factor: float = 0.95):
    """Find optimal path through good rocks and compute its value."""
    good_rocks = [
        loc for loc, rock_id in problem.rock_locs.items()
        if problem.env.state.rocktypes[rock_id] == "good"
    ]

    # Try all possible orderings of rocks
    best_value = 0
    for rock_order in permutations(good_rocks):
        value = get_path_value(problem, rock_order, discount_factor)
        best_value = max(best_value, value)

    return best_value


def get_full_info_value_greedy(problem: RockSampleProblem, discount_factor: float = 0.95):
    """Find near-optimal path through good rocks using greedy approach."""
    # Get good rocks
    good_rocks = [
        loc for loc, rock_id in problem.rock_locs.items()
        if problem.env.state.rocktypes[rock_id] == "good"
    ]

    value = 0
    current_pos = problem.env.state.position
    steps = 0
    rocks_left = set(good_rocks)

    while rocks_left:
        best_next_value = -float('inf')
        best_rock = None

        for rock in rocks_left:
            dist = manhattan_dist(current_pos, rock)
            steps_to_rock = steps + dist
            rock_value = 10 * (discount_factor ** steps_to_rock)  # value of sampling this rock
            # Consider remaining path length to exit through rightmost remaining rock
            rightmost_remaining = max(r[0] for r in (rocks_left - {rock})) if len(rocks_left) > 1 else rock[0]
            future_discount = discount_factor ** (steps_to_rock + 1 + (problem.n - rightmost_remaining))
            value_estimate = rock_value + (10 * future_discount if rightmost_remaining < problem.n else 0)

            if value_estimate > best_next_value:
                best_next_value = value_estimate
                best_rock = rock

        # Collect best rock
        steps += manhattan_dist(current_pos, best_rock)  # steps to reach rock
        value += 10 * (discount_factor ** steps)  # sampling reward
        steps += 1  # sampling action
        current_pos = best_rock
        rocks_left.remove(best_rock)

    # Add exit reward if we can reach it
    steps_to_exit = problem.n - current_pos[0]
    if steps + steps_to_exit < problem.n:
        value += 10 * (discount_factor ** (steps + steps_to_exit))

    return value

#
# def value_given_reward_times(reward_times, discount_factor=0.95):
#     # reward_times = [r-1 for r in reward_times]
#     return sum([10 * discount_factor ** t for t in reward_times])
#
# print(value_given_reward_times([3, 10, 12, 14, 16, 18]))
# print(value_given_reward_times([5, 7, 9, 11, 19, 25]))


