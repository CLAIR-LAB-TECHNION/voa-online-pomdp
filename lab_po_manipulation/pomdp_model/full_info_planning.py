import itertools
import math
import time
import copy
from typing import List, Tuple, Optional
from lab_po_manipulation.manipulation_prm import ManipulationPRM
from lab_po_manipulation.pomdp_model.pomanipulation_problem import State, POMProblem


def get_path_value(prm: ManipulationPRM, start_vid: int, end_vid: int,
                   action_reward: float = 0, discount_factor: float = 0.98,
                   current_steps: int = 0) -> Tuple[float, Optional[List], int]:
    """
    Compute discounted reward for path between vertices including an optional action at the end.
    Returns (total_value, path, end_steps)
    """
    path = prm.find_path_by_vertices_id(start_vid, end_vid)
    if not path:
        return float('-inf'), None, current_steps

    steps = len(path) - 1  # -1 because path includes start
    movement_reward = steps * -0.5  # movement cost for each step
    discounted_action_reward = action_reward * (discount_factor ** (current_steps + steps)) if action_reward else 0

    total_value = movement_reward + discounted_action_reward
    return total_value, path, current_steps + steps


def evaluate_pour_sequence(prm_y_up: ManipulationPRM, start_vid: int,
                           current_steps: int, discount_factor: float) -> Tuple[float, int]:
    """
    Try all possible pour sequences if reachable.
    Returns best (value, end_steps)
    """
    pour_configs = {
        "red": prm_y_up.get_all_interesting_configs_as_dict().get("pre_pour_red", []),
        "blue": prm_y_up.get_all_interesting_configs_as_dict().get("pre_pour_blue", [])
    }

    # Check if pour configurations are reachable
    reachable_pours = {}
    for cup, pour_vids in pour_configs.items():
        for vid in pour_vids:
            if prm_y_up.find_path_by_vertices_id(start_vid, vid):
                reachable_pours[cup] = vid
                break

    if not reachable_pours:
        # If no pour configs reachable, just count steps until episode end
        remaining_steps = 70 - current_steps
        return remaining_steps * -0.5, current_steps + remaining_steps

    best_value = float('-inf')
    best_steps = current_steps

    # Try all pour orders
    for pour_order in itertools.permutations(reachable_pours.items()):
        value = 0
        steps = current_steps
        curr_vid = start_vid

        for cup, pour_vid in pour_order:
            # Path to pour config
            path_value, path, new_steps = get_path_value(
                prm_y_up, curr_vid, pour_vid,
                action_reward=10,  # fill_cup_reward
                discount_factor=discount_factor,
                current_steps=steps
            )
            if path_value == float('-inf'):
                continue

            value += path_value
            steps = new_steps + 1  # +1 for pour action
            curr_vid = pour_vid

        # After pouring, try to reach trash
        trash_vid = prm_y_up.get_all_interesting_configs_as_dict().get("trash", [None])[0]
        if trash_vid:
            trash_value, trash_path, final_steps = get_path_value(
                prm_y_up, curr_vid, trash_vid,
                action_reward=1,  # trash_reward
                discount_factor=discount_factor,
                current_steps=steps
            )
            if trash_value != float('-inf'):
                value += trash_value
                steps = final_steps + 1  # +1 for trash action
            else:
                # If trash unreachable, just count remaining steps
                remaining_steps = 70 - steps
                value += remaining_steps * -0.5
                steps += remaining_steps

        if value > best_value and not math.isnan(value):  # Added NaN check
            best_value = value
            best_steps = steps

    if best_value == float('-inf'):
        remaining_steps = 70 - current_steps
        return remaining_steps * -0.5, current_steps + remaining_steps

    return best_value, best_steps


def evaluate_cup_sequence(prm: ManipulationPRM, prm_y_up: ManipulationPRM, start_vid: int,
                          cup_order: List[Tuple[str, str]], can_position: str,
                          discount_factor: float = 0.98) -> float:
    """
    Evaluate one specific sequence of cup placements followed by can pickup and pouring.
    Returns total discounted reward for best execution of this sequence.
    """
    cups_value = 0
    current_steps = 0
    current_vid = start_vid

    # Place cups in specified order
    for cup_name, start_pos in cup_order:
        pickup_grasp_configs = prm.get_grasp_configs_for_position(start_pos)
        target_pos = f"workspace_{cup_name}"

        best_cup_stage_value = float('-inf')
        best_steps = 0
        best_putdown_vid = None

        # Try all valid grasp configurations for pickup
        for pickup_vid, _, _ in pickup_grasp_configs:
            pickup_value, pickup_path, steps = get_path_value(
                prm, current_vid, pickup_vid,
                action_reward=1,  # pick_up_cup_reward
                discount_factor=discount_factor,
                current_steps=current_steps
            )
            if pickup_value == float('-inf'):
                continue

            putdown_grasp_configs = prm.get_grasp_configs_for_position(target_pos)

            # Try all putdown configurations
            for putdown_vid, _, _ in putdown_grasp_configs:
                putdown_value, putdown_path, new_steps = get_path_value(
                    prm, pickup_vid, putdown_vid,
                    action_reward=5,  # put_in_place_reward
                    discount_factor=discount_factor,
                    current_steps=steps + 1  # +1 for pickup action
                )
                if putdown_value == float('-inf'):
                    continue

                stage_value = pickup_value + putdown_value
                if stage_value > best_cup_stage_value:
                    best_cup_stage_value = stage_value
                    best_steps = new_steps + 1  # +1 for putdown action
                    best_putdown_vid = putdown_vid

        if best_cup_stage_value == float('-inf'):
            remaining_steps = 70 - current_steps
            return remaining_steps * -0.5

        cups_value += best_cup_stage_value
        current_steps = best_steps
        current_vid = best_putdown_vid

    # Handle can after cups are placed
    can_grasp_configs = prm.get_grasp_configs_for_position(can_position)
    best_full_sequence_value = float('-inf')

    if not can_grasp_configs:  # If no grasp configs for can
        remaining_steps = 70 - current_steps
        return cups_value + (remaining_steps * -0.5)

    # Try all can grasp configurations
    for can_vid, _, _ in can_grasp_configs:
        can_value, can_path, new_steps = get_path_value(
            prm, current_vid, can_vid,
            action_reward=0.1,  # pick_up_can_reward
            discount_factor=discount_factor,
            current_steps=current_steps
        )
        if can_value == float('-inf'):
            continue

        # Need to get corresponding vid in y_up prm
        config = prm.vertices[can_vid]
        can_vid_y_up = prm_y_up.vertex_to_id[tuple(config)]

        # Check pour and trash sequence from this grasp using y_up prm
        pour_value, final_steps = evaluate_pour_sequence(
            prm_y_up, can_vid_y_up,  # Use the y_up vid
            new_steps + 1,  # +1 for pickup action
            discount_factor
        )

        full_sequence_value = cups_value + can_value + pour_value
        best_full_sequence_value = max(best_full_sequence_value, full_sequence_value)

    if best_full_sequence_value == float('-inf'):
        remaining_steps = 70 - current_steps
        return cups_value + (remaining_steps * -0.5)

    return best_full_sequence_value


def plan_full_info_manipulation(prm: ManipulationPRM, prm_y_up: ManipulationPRM,
                               init_state: State, discount_factor: float = 0.98) -> float:
    """
    Find optimal sequence of actions in fully observable case.
    Returns the total discounted reward for best sequence.
    """
    start_vid = init_state.robot_cfg_id
    best_value = float('-inf')

    # Get initial position of cups and can
    cup_positions = []
    can_position = None
    for pos, obj in init_state.positions_to_objects.items():
        if obj in ["red_cup", "blue_cup"]:
            cup_positions.append((obj, pos))
        elif obj == "soda_can":
            can_position = pos

    # Try both cup orders
    for cup_order in itertools.permutations(cup_positions):
        value = evaluate_cup_sequence(prm, prm_y_up, start_vid, cup_order, can_position, discount_factor)
        if value > best_value:
            best_value = value

    return best_value


