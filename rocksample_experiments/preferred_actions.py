"""
Implementing the Preferred actions as in POMCP papers for this problem: give higher initial value to actions that are
more likely to be useful and also using the preferred action set for the rollout policy.
"""

import random
import pomdp_py
import numpy as np
from rocksample_experiments.rocksample_problem import RSPolicyModel, CheckAction, MoveEast, MoveNorth, \
    MoveSouth, MoveWest, SampleAction, State, RockType


class RSActionPrior(pomdp_py.ActionPrior):
    def __init__(self, n, k, rock_locs, v_init=10, n_visits_init=10):
        self.n = n
        self.k = k
        self.rock_locs = rock_locs
        self.rock_pos = np.zeros((k, 2))
        for key, value in rock_locs.items():
            self.rock_pos[value] = key
        self.v_init = v_init
        self.n_visits_init = n_visits_init

    def get_preferred_actions(self, state, history):
        # Check if there's a rock at current position worth sampling
        is_at_rock = (state.position[0] == self.rock_pos[:, 0]) & (state.position[1] == self.rock_pos[:, 1])
        current_rock = None if not is_at_rock.any() else np.where(is_at_rock)[0][0]

        if current_rock is not None:
            # Check if rock should be sampled
            for action, observation in reversed(history):
                if isinstance(action, SampleAction) and action.position[0] == state.position[0] and \
                        action.position[1] == state.position[1]:
                    break
                if isinstance(action, CheckAction) and action.rock_id == current_rock:
                    if observation.quality == RockType.GOOD:
                        return [(SampleAction(state.position), self.n_visits_init, self.v_init)]
                    elif observation.quality == RockType.BAD:
                        break

        # Process rocks to determine interesting directions
        per_rock_totals = np.zeros(len(self.rock_locs))
        for action, observation in history:
            if isinstance(action, SampleAction):
                if action.position in self.rock_locs:
                    per_rock_totals[self.rock_locs[action.position]] = -100
            elif isinstance(action, CheckAction):
                if observation.quality == RockType.GOOD:
                    per_rock_totals[action.rock_id] += 1
                elif observation.quality == RockType.BAD:
                    per_rock_totals[action.rock_id] -= 1

        interesting_rock_ids = np.where(per_rock_totals >= 0)[0]
        if len(interesting_rock_ids) == 0:
            return [(MoveEast, self.n_visits_init, self.v_init)]

        interesting_positions = self.rock_pos[interesting_rock_ids]
        north_interesting = (interesting_positions[:, 1] < state.position[1]).any()
        south_interesting = (interesting_positions[:, 1] > state.position[1]).any()
        east_interesting = (interesting_positions[:, 0] > state.position[0]).any()
        west_interesting = (interesting_positions[:, 0] < state.position[0]).any()

        actions = []
        if north_interesting and state.position[1] > 0:
            actions.append(MoveNorth)
        if south_interesting and state.position[1] < self.n - 1:
            actions.append(MoveSouth)
        if east_interesting and state.position[0] < self.n - 1:
            actions.append(MoveEast)
        if west_interesting and state.position[0] > 0:
            actions.append(MoveWest)

        recent_checks = {action.rock_id for action, _ in history[-5:]
                         if isinstance(action, CheckAction)}

        for rock_id in interesting_rock_ids:
            if rock_id not in recent_checks:
                actions.append(CheckAction(rock_id))

        if not actions:
            return []

        return [(action, self.n_visits_init, self.v_init) for action in actions]


class CustomRSPolicyModel(RSPolicyModel):
    def __init__(self, n, k, actions_prior):
        self.action_prior = actions_prior
        super().__init__(n, k)

    def rollout(self, state: State, history=None):
        if history is None:
            return super().rollout(state, history)

        preferred_actions = self.action_prior.get_preferred_actions(state, history)
        if not preferred_actions:
            return super().rollout(state, history)

        return random.choice([action for action, _, _ in preferred_actions])

