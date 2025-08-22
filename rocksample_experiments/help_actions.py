from copy import deepcopy
from rocksample_experiments.rocksample_problem import RockSampleProblem, State


def is_legal_rock_push(problem: RockSampleProblem, rocks_to_push: dict) -> bool:
    """
    Check if pushing rocks according to rocks_to_push is legal:
    - All rocks remain within grid bounds
    - No rocks overlap after pushing

    Args:
        problem: Current RockSample problem instance
        rocks_to_push: Dict mapping rock_id to (push_x, push_y) displacement

    Returns:
        bool: True if push configuration is legal
    """
    n = problem.n  # Grid size
    new_positions = {}

    rock_id_to_pos = {rock_id: pos for pos, rock_id in problem.rock_locs.items()}

    # Calculate new positions
    for rock_id, pos in rock_id_to_pos.items():
        if rock_id in rocks_to_push:
            push_x, push_y = rocks_to_push[rock_id]
            new_x = pos[0] + push_x
            new_y = pos[1] + push_y

            # Check grid bounds
            if new_x < 0 or new_x >= n or new_y < 0 or new_y >= n:
                return False

            new_positions[rock_id] = (new_x, new_y)
        else:
            new_positions[rock_id] = pos

    # Check for overlaps
    seen_positions = set()
    for pos in new_positions.values():
        if pos in seen_positions:
            return False
        seen_positions.add(pos)

    return True


def push_rocks(problem: RockSampleProblem, rocks_to_push: dict, deepcopy_belief=True) -> (RockSampleProblem, int):
    """
    Create new RockSample instance with rocks pushed according to rocks_to_push

    Args:
        problem: Current RockSample problem instance
        rocks_to_push: Dict mapping rock_id to (push_x, push_y) displacement

    Returns:
        RockSampleProblem: New instance with pushed rocks

    Raises:
        AssertionError: If resulting push configuration is illegal
    """
    # Verify push is legal
    assert is_legal_rock_push(problem, rocks_to_push), "Illegal rock push configuration"

    # Calculate total Manhattan distance of pushes
    total_push_dist = sum(abs(dx) + abs(dy) for dx, dy in rocks_to_push.values())

    # Create new rock locations
    new_rock_locs = {}
    rock_id_to_pos = {rock_id: pos for pos, rock_id in problem.rock_locs.items()}

    for rock_id, pos in rock_id_to_pos.items():
        if rock_id in rocks_to_push:
            push_x, push_y = rocks_to_push[rock_id]
            new_pos = (pos[0] + push_x, pos[1] + push_y)
        else:
            new_pos = pos
        new_rock_locs[new_pos] = rock_id

    init_belief = deepcopy(problem.agent.cur_belief) if deepcopy_belief else problem.agent.cur_belief
    # Create new instance with same parameters but updated rock locations
    new_problem = RockSampleProblem(
        n=problem.n,
        k=problem.k,
        init_state=deepcopy(problem.env.state),
        rock_locs=new_rock_locs,
        init_belief=init_belief,
        half_efficiency_dist=problem.agent.observation_model._half_efficiency_dist
    )

    return new_problem, total_push_dist


def calculate_num_push_help_actions(k: int, m: int) -> int:
    """
    Calculate the total number of possible help actions for k rocks with budget m.

    For k rocks, we have 2k dimensions (x and y for each rock) where we can allocate m units of movement.
    This is equivalent to placing m identical balls into 2k distinct boxes, where boxes can be empty
    or receive multiple balls (combinations with repetition). The formula for this is:
    C(n+r-1,r) where n=2k (dimensions) and r=m (budget).
    Then for each dimension that received an allocation, we can choose positive or negative direction.
    This means multiplying by 2^(2k) for all possible sign combinations.

    The complete formula is:
    ((2k+m-1)! / (m! * (2k-1)!)) * 2^(2k)

    Args:
        k: Number of rocks
        m: Manhattan distance budget

    Returns:
        int: Number of possible help actions (ignoring illegal configurations)
    """

    def factorial(n: int) -> int:
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    # Calculate C(2k+m-1,m) = (2k+m-1)! / (m! * (2k-1)!)
    numerator = factorial(2 * k + m - 1)
    denominator = factorial(m) * factorial(2 * k - 1)
    combinations = numerator // denominator

    # Multiply by 2^(2k) for sign choices
    sign_combinations = 2 ** (2 * k)

    return combinations * sign_combinations

# print(calculate_num_push_help_actions(7, 5))