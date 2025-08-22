import random
import time
from typing import Any, Optional
import pomdp_py
from lab_po_manipulation.env_configurations.objects_and_positions import \
    sensing_configs_dict as default_sensing_configs, positions_dict
from lab_po_manipulation.manipulation_prm import ManipulationPRM


# # # State
class State(pomdp_py.State):
    def __init__(self, robot_cfg_id: int, positions_to_objects: dict, held_object: Optional[str] = None,
                 red_cup_full: bool = False, blue_cup_full: bool = False, can_in_trash: bool = False, ):
        self.robot_cfg_id = robot_cfg_id  # vertex id in the roadmap
        self.held_object = held_object
        self.positions_to_objects = positions_to_objects  # map from position name to object name, None if empty
        # (if object is held, it's not in this dict)

        # predicates for task completion:
        self.red_cup_full = red_cup_full
        self.blue_cup_full = blue_cup_full
        self.can_in_trash = can_in_trash

        self.terminal = can_in_trash

    def __hash__(self):
        # Convert dictionary to a tuple of sorted items
        frozen_positions = tuple(sorted(self.positions_to_objects.items()))
        return hash((self.robot_cfg_id, self.held_object, frozen_positions, self.terminal, self.red_cup_full,
                     self.blue_cup_full, self.can_in_trash))

    def __eq__(self, other):
        return (self.robot_cfg_id == other.robot_cfg_id and
                self.held_object == other.held_object and
                self.positions_to_objects == self.positions_to_objects and
                self.terminal == other.terminal and
                self.red_cup_full == other.red_cup_full and
                self.blue_cup_full == other.blue_cup_full and
                self.can_in_trash == other.can_in_trash)

    def __repr__(self):
        return f"State(robot_cfg_id={self.robot_cfg_id}, held_object={self.held_object}," \
               f" object_positions={self.positions_to_objects}, terminal={self.terminal}), " \
               f"red_cup_full={self.red_cup_full}, blue_cup_full={self.blue_cup_full}," \
               f" can_in_trash={self.can_in_trash})"


# # # Actions
class Action(pomdp_py.Action):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Action({self.name})"


class MoveAction(Action):
    def __init__(self, cfg_id):
        super().__init__(name=f"move_to_{cfg_id}")
        self.cfg_id = cfg_id


class PickUpAction(Action):
    def __init__(self, ):
        super().__init__(name="pick_up")


class PutDownAction(Action):
    def __init__(self, ):
        super().__init__(name="put_down")


class PourAction(Action):
    def __init__(self, ):
        super().__init__(name="pour")


class SenseAction(Action):
    def __init__(self, ):
        super().__init__(name="sense")


class TrashAction(Action):
    ''' can only happen when in trash configuration and finishes the task'''

    def __init__(self, ):
        super().__init__(name="trash")


# # # Observations
class BaseObservation(pomdp_py.Observation):
    def __init__(self, held_object: Optional[str],
                 red_cup_full: bool,
                 blue_cup_full: bool):
        self.held_object = held_object
        self.red_cup_full = red_cup_full
        self.blue_cup_full = blue_cup_full

    def __hash__(self):
        return hash((self.held_object, self.red_cup_full, self.blue_cup_full))

    def __eq__(self, other):
        return (isinstance(other, BaseObservation) and
                self.held_object == other.held_object and
                self.red_cup_full == other.red_cup_full and
                self.blue_cup_full == other.blue_cup_full)

    def __repr__(self):
        return f"BaseObservation(held_object={self.held_object}, red_cup_full={self.red_cup_full}, blue_cup_full={self.blue_cup_full})"


class SensingObservation(BaseObservation):
    def __init__(self, positions_to_objects: dict,
                 robot_cfg_id: int,
                 held_object: Optional[str],
                 red_cup_full: bool,
                 blue_cup_full: bool):
        super().__init__(held_object, red_cup_full, blue_cup_full)
        self.robot_cfg_id = robot_cfg_id
        self.positions_to_objects = positions_to_objects

    def __hash__(self):
        frozen_positions = tuple(sorted(self.positions_to_objects.items()))
        return hash((frozen_positions, self.robot_cfg_id, self.held_object,
                     self.red_cup_full, self.blue_cup_full))

    def __eq__(self, other):
        if not isinstance(other, SensingObservation):
            return False
        return (self.positions_to_objects == other.positions_to_objects and
                self.robot_cfg_id == other.robot_cfg_id and
                self.held_object == other.held_object and
                self.red_cup_full == other.red_cup_full and
                self.blue_cup_full == other.blue_cup_full)

    def __repr__(self):
        return f"SensingObservation(object_positions={self.positions_to_objects}, " \
               f"robot_cfg_id={self.robot_cfg_id}, held_object={self.held_object}, " \
               f"red_cup_full={self.red_cup_full}, blue_cup_full={self.blue_cup_full})"


# # # Transition Model
class POMTransitionModel(pomdp_py.TransitionModel):
    def __init__(self, prm: ManipulationPRM, prm_y_up: ManipulationPRM):
        """

        @param prm: roadmap of configurations that is used to infer which actions are possible
        @param prm_y_up: roadmap of configurations when holding can, so it wont spill y of gripper frame is up
            in all configs and edges (up to some tolerance)
        """
        self.prm = prm
        self.prm_y_up = prm_y_up

        self.prm_y_up_interesting_configs = self.prm_y_up.get_all_interesting_configs_as_dict()
        self.prm_interesting_configs = self.prm.get_all_interesting_configs_as_dict()

    def sample(self, state: State, action: Action):
        # defaults:
        cfg_id = state.robot_cfg_id
        positions_to_objects = state.positions_to_objects.copy()
        held_object = state.held_object
        red_cup_full = state.red_cup_full
        blue_cup_full = state.blue_cup_full
        can_in_trash = state.can_in_trash

        curr_prm = self.prm if state.held_object != "soda_can" else self.prm_y_up
        if isinstance(action, MoveAction):
            cfg_id = action.cfg_id

        elif isinstance(action, PickUpAction):
            assert held_object is None, "Applied pickup action while holding something"
            grasp_config = curr_prm.grasp_configs.get(state.robot_cfg_id, None)
            assert grasp_config is not None, f"Applied pickup action at a wrong configuration {state.robot_cfg_id}"
            position_name = grasp_config["position_name"]
            object_name = positions_to_objects[position_name]
            assert object_name != 'empty', "attempt to pick up from a marked empty position"
            if object_name is not None:
                held_object = object_name
                positions_to_objects[position_name] = None
            # Mark position as empty after pickup attempt regardless of success
            positions_to_objects[position_name] = 'empty'

            # if holding can, the agent moves to a different roadmap. Retrieve the corresponding configuration id
            # in the constrained roadmap
            if held_object == "soda_can":
                config = self.prm.vertices[cfg_id]
                cfg_id = self.prm_y_up.vertex_to_id[tuple(config)]

        elif isinstance(action, PutDownAction):
            assert held_object is not None, "Applied putdown action while holding nothing"
            grasp_config = curr_prm.grasp_configs.get(state.robot_cfg_id, None)
            assert grasp_config is not None, f"Applied putdown action at a wrong configuration {state.robot_cfg_id}"
            position_name = grasp_config["position_name"]
            assert positions_to_objects[position_name] is None, f"Applied putdown action to a non-empty position" \
                                                                f"state: {state}, action: {action}, history: {history}"
            positions_to_objects[position_name] = held_object
            held_object = None

            # if were holding can, move back to the unconstrained roadmap
            if positions_to_objects[position_name] == "soda_can":
                config = self.prm_y_up.vertices[cfg_id]
                cfg_id = self.prm.vertex_to_id[tuple(config)]

        elif isinstance(action, PourAction):
            assert held_object == "soda_can", "Applied pour action while holding nothing or wrong object"
            # holding soda can we're on the constrained roadmap
            pour_red_config = self.prm_y_up_interesting_configs["pre_pour_red"][0]
            pour_blue_config = self.prm_y_up_interesting_configs["pre_pour_blue"][0]
            assert cfg_id in [pour_red_config, pour_blue_config], "Applied pour action at a wrong configuration"
            if cfg_id == pour_red_config:
                red_cup_full = True
            elif cfg_id == pour_blue_config:
                blue_cup_full = True

        elif isinstance(action, SenseAction):
            # sense action doesn't change the state
            pass

        elif isinstance(action, TrashAction):
            assert state.held_object == "soda_can", "Applied trash action while holding nothing or wrong object"
            assert self.prm_y_up_interesting_configs["trash"][0] == cfg_id, \
                f"Applied trash action at a wrong configuration curent config {cfg_id}" \
                f"trash config {self.prm_y_up_interesting_configs['trash'][0]}"

            can_in_trash = True

        else:
            raise ValueError(f"Unknown action type {action}")

        return State(robot_cfg_id=cfg_id, positions_to_objects=positions_to_objects, held_object=held_object,
                     red_cup_full=red_cup_full, blue_cup_full=blue_cup_full, can_in_trash=can_in_trash)


# # # Observation Model
class POMObservationModel(pomdp_py.ObservationModel):
    def __init__(self, cfg_id_to_sensing_config_name: dict, sensing_configs=default_sensing_configs):
        self.sensing_configs = sensing_configs  # map name to config and data about visibility
        self.cfg_id_to_sensing_config_name = cfg_id_to_sensing_config_name  # map cfg_id in unconstrained prm to
        # sensing config name

    def sample(self, next_state: State, action: Action):
        if not isinstance(action, SenseAction):
            return BaseObservation(
                held_object=next_state.held_object,
                red_cup_full=next_state.red_cup_full,
                blue_cup_full=next_state.blue_cup_full
            )

        cfg_id = next_state.robot_cfg_id
        sense_config_name = self.cfg_id_to_sensing_config_name.get(cfg_id, None)
        if sense_config_name is None:
            print(f"Warning: no sensing config for cfg_id {cfg_id}. Sensing action shouldn't be available")
            return BaseObservation(
                held_object=next_state.held_object,
                red_cup_full=next_state.red_cup_full,
                blue_cup_full=next_state.blue_cup_full
            )

        detect_prob = self.sensing_configs[sense_config_name].detection_probability
        visible_positions = self.sensing_configs[sense_config_name].visible_positions

        positions_to_detected_objects = {}
        for position in visible_positions:
            obj = next_state.positions_to_objects[position]
            # Only include actual objects, not 'empty' or None
            if obj is not None and obj != 'empty':
                if random.random() < detect_prob:
                    positions_to_detected_objects[position] = obj

        return SensingObservation(
            positions_to_detected_objects,
            robot_cfg_id=next_state.robot_cfg_id,
            held_object=next_state.held_object,
            red_cup_full=next_state.red_cup_full,
            blue_cup_full=next_state.blue_cup_full
        )


# # # Reward Model
class POMRewardModel(pomdp_py.RewardModel):
    def __init__(self, ):
        self.all_action_cost = -0.5 # cost for all actions
        # additional rewards to cost:
        self.pick_up_cup_reward = 1
        self.pick_up_can_reward = 0.1
        self.put_in_place_reward = 5
        self.trash_reward = 1
        self.fill_cup_reward = 10

    def sample(self, state: State, action: Action, next_state: State):
        reward = self.all_action_cost

        if isinstance(action, PickUpAction):
            reward += self.get_pick_up_reward(state, next_state)
        elif isinstance(action, PutDownAction):
            reward += self.get_put_down_reward(state, next_state)
        elif isinstance(action, PourAction):
            reward += self.get_fill_cup_reward(state, next_state)
        elif isinstance(action, TrashAction):
            reward += self.trash_reward

        return reward

    def get_pick_up_reward(self, state, next_state):
        if state.held_object is None and next_state.held_object in ["red_cup", "blue_cup"]:
            return self.pick_up_cup_reward
        if state.held_object is None and next_state.held_object == "soda_can":
            return self.pick_up_can_reward
        return 0

    def get_put_down_reward(self, state, next_state):
        """ only get reward if put down one of the cups in it's place"""
        if state.held_object == "red_cup" and next_state.held_object is None:
            if next_state.positions_to_objects["workspace_red_cup"] == "red_cup":
                return self.put_in_place_reward
        elif state.held_object == "blue_cup" and next_state.held_object is None:
            if next_state.positions_to_objects["workspace_blue_cup"] == "blue_cup":
                return self.put_in_place_reward
        return 0

    def get_fill_cup_reward(self, state, next_state):
        if not state.red_cup_full and next_state.red_cup_full:
            return self.fill_cup_reward
        if not state.blue_cup_full and next_state.blue_cup_full:
            return self.fill_cup_reward
        return 0


# # # policy model
class POMPolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, prm: ManipulationPRM, prm_y_up: ManipulationPRM, vid_to_sensing_config_name: dict):
        self.prm = prm
        self.prm_y_up = prm_y_up
        self.vid_to_sensing_config_name = vid_to_sensing_config_name

        self.prm_interesting_configs = self.prm.get_all_interesting_configs_as_dict()
        self.prm_y_up_interesting_configs = self.prm_y_up.get_all_interesting_configs_as_dict()

    def get_all_actions(self, **kwargs):
        state = kwargs["state"]
        # history = kwargs["history"]

        cfg_id = state.robot_cfg_id

        # if in pour config and glass is not full, that's the only action
        if state.held_object == "soda_can" and state.positions_to_objects["workspace_red_cup"] == "red_cup" and \
                self.prm_y_up_interesting_configs["pre_pour_red"][0] == cfg_id and not state.red_cup_full:
            return [PourAction()]
        if state.held_object == "soda_can" and state.positions_to_objects["workspace_blue_cup"] == "blue_cup" and \
                self.prm_y_up_interesting_configs["pre_pour_blue"][0] == cfg_id and not state.blue_cup_full:
            return [PourAction()]

        actions = []

        # add movement actions:
        current_prm = self.prm if state.held_object != "soda_can" else self.prm_y_up
        for neighbor in current_prm.vertex_id_to_edges[cfg_id]:
            actions.append(MoveAction(neighbor))

        # add pick up actions at positions:
        if state.held_object is None:
            grasp_config = current_prm.grasp_configs.get(cfg_id, None)
            if grasp_config is not None:
                position_name = grasp_config["position_name"]
                # Only add pickup if position has an actual object (not None and not 'empty')
                if (position_name not in ["workspace_red_cup", "workspace_blue_cup"] and
                        state.positions_to_objects[position_name] is not None and
                        state.positions_to_objects[position_name] != 'empty'):
                    actions.append(PickUpAction())

        # add put down actions at relevant positions only (if holding something):
        if state.held_object is not None:
            grasp_config = current_prm.grasp_configs.get(cfg_id, None)
            if grasp_config is not None:
                if grasp_config["position_name"] == "workspace_red_cup" and state.held_object == "red_cup":
                    return [PutDownAction()]
                if grasp_config["position_name"] == "workspace_blue_cup" and state.held_object == "blue_cup":
                    return [PutDownAction()]

        # add trash action:
        if state.held_object == 'soda_can' and state.robot_cfg_id == self.prm_y_up_interesting_configs["trash"][0]:
            actions.append(TrashAction())

        # add sense action:
        if state.held_object is None and cfg_id in self.vid_to_sensing_config_name.keys():
            actions.append(SenseAction())

        return actions

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state), 1)[0]


# # # problem
class POMProblem(pomdp_py.POMDP):
    def __init__(self, prm: ManipulationPRM,
                 prm_y_up: ManipulationPRM,
                 init_state=None,
                 init_belief=None,
                 n_belief_particles=3000):

        agent = self.build_agent(prm, prm_y_up, init_belief, n_belief_particles)

        if init_state is None:
            init_state = self.sample_init_state(prm)

        env = pomdp_py.Environment(init_state=init_state, transition_model=POMTransitionModel(prm, prm_y_up),
                                   reward_model=POMRewardModel())

        super().__init__(agent, env, name="POMProblem")

    @staticmethod
    def build_agent(prm: ManipulationPRM, prm_y_up: ManipulationPRM, init_belief=None, n_belief_particles=3000):
        """
        useful when you don't need to simulate environment but just to plan with an agent
        and execute in real world
        """
        if init_belief is None:
            init_belief = POMProblem.init_particle_belief(n_belief_particles, prm)

        sensing_configs = default_sensing_configs
        vid_to_sensing_config_name = {}
        for config_name, cfg_id in prm.get_all_interesting_configs():
            if config_name in sensing_configs.keys():
                vid_to_sensing_config_name[cfg_id] = config_name

        agent = pomdp_py.Agent(init_belief=init_belief,
                               policy_model=POMPolicyModel(prm, prm_y_up, vid_to_sensing_config_name),
                               transition_model=POMTransitionModel(prm, prm_y_up),
                               observation_model=POMObservationModel(vid_to_sensing_config_name, sensing_configs),
                               reward_model=POMRewardModel())
        return agent

    @staticmethod
    def sample_init_state(prm: ManipulationPRM = None):
        cups_possible_positions = ["top_shelf_left", "top_shelf_right", "middle_shelf", "middle_shelf_left",
                                   "bottom_left", "bottom_right"]
        can_possible_positions = cups_possible_positions.copy()

        all_possible_positions = list(positions_dict.keys())

        positions_to_objects = {pos: None for pos in all_possible_positions}

        # sample can position:
        can_position = random.choice(can_possible_positions)
        positions_to_objects[can_position] = "soda_can"
        cups_possible_positions.remove(can_position)

        # sample cup positions:
        red_cup_position = random.choice(cups_possible_positions)
        cups_possible_positions.remove(red_cup_position)
        blue_cup_position = random.choice(cups_possible_positions)

        positions_to_objects[red_cup_position] = "red_cup"
        positions_to_objects[blue_cup_position] = "blue_cup"

        return State(robot_cfg_id=prm.home_cfg_idx, positions_to_objects=positions_to_objects,
                     held_object=None, red_cup_full=False, blue_cup_full=False, can_in_trash=False)

    @staticmethod
    def init_particle_belief(num_particles, prm: ManipulationPRM = None):
        particles = [POMProblem.sample_init_state(prm) for _ in range(num_particles)]
        return pomdp_py.Particles(particles)

class POMActionPrior(pomdp_py.ActionPrior):
    def __init__(self, prm: ManipulationPRM, prm_y_up: ManipulationPRM,
                 vid_to_sensing_config_name: dict, v_init=2, n_visits_init=2):
        self.prm = prm
        self.prm_y_up = prm_y_up
        self.vid_to_sensing_config_name = vid_to_sensing_config_name
        self.v_init = v_init
        self.n_visits_init = n_visits_init

        self.prm_distances = self.prm.compute_distances_to_interesting_configs()
        self.prm_y_up_distances = self.prm_y_up.compute_distances_to_interesting_configs()

        self.prm_interesting_configs = self.prm.get_all_interesting_configs_as_dict()
        self.prm_y_up_interesting_configs = self.prm_y_up.get_all_interesting_configs_as_dict()

        # map tuple of (vid, sense_config_name) to best neighbour for sensing
        self.prm_cfg_to_best_neighbour_for_sensing = self.generate_cache_for_sense_best_neighbour()

    def get_preferred_actions(self, state, history):

        if state.held_object is None:
            return self.get_preferred_actions_gripper_empty(state, history)

        return self.get_preferred_actions_gripper_full(state, history)

    def get_preferred_actions_gripper_empty(self, state, history):
        preffered_actions = []

        sense_history = [(a, o) for a, o in history if isinstance(a, SenseAction)]

        # if this is one of the first 50 steps. sensing is a preffered action if available and didn't sense
        # at that position yet
        if len(history) < 50 and state.robot_cfg_id in self.vid_to_sensing_config_name.keys():
            sensed_here = False
            for action, observation in sense_history:
                if isinstance(action, SenseAction) and observation.robot_cfg_id == state.robot_cfg_id:
                    sensed_here = True
                    break
            if not sensed_here:
                preffered_actions.append((SenseAction(), self.n_visits_init, self.v_init))

        # process to get all positions where objects detected:
        detected_positions = {}
        for action, observation in sense_history:
            for pos, obj in observation.positions_to_objects.items():
                if obj is not None:
                    detected_positions[pos] = obj  # That uses the last detection of this position...

        # if not all objects were detected yet, moving toward sense configs that were not performed yet is also a
        # preferred action
        if len(detected_positions) < 3  and state.robot_cfg_id not in self.vid_to_sensing_config_name.keys():
            sensed_configs = [o.robot_cfg_id for a, o in sense_history]
            for vid, config_name in self.vid_to_sensing_config_name.items():
                if vid not in sensed_configs:
                    best_neighbour = self.prm_cfg_to_best_neighbour_for_sensing.get((state.robot_cfg_id, config_name), None)
                    if best_neighbour is not None:
                        preffered_actions.append((MoveAction(best_neighbour), self.n_visits_init, self.v_init))

        # if at position that was sensed for a cup pick it up is the only preffered action
        # same if in position that was sensed for a can and two cups in position (or lower value for one cup)
        grasp_config = self.prm.grasp_configs.get(state.robot_cfg_id, None)
        if grasp_config is not None:
            position_name = grasp_config["position_name"]
            object_in_position = detected_positions.get(position_name, None)
            if state.positions_to_objects[position_name] == 'empty':
                # already picked up (marked)
                object_in_position = None

            if object_in_position in ["red_cup", "blue_cup"]:
                return [(PickUpAction(), self.n_visits_init, self.v_init)]
            if object_in_position == "soda_can" and state.positions_to_objects["workspace_red_cup"] == "red_cup" and \
                    state.positions_to_objects["workspace_blue_cup"] == "blue_cup":
                return [(PickUpAction(), self.n_visits_init, self.v_init)]
            if object_in_position == "soda_can" and (state.positions_to_objects["workspace_red_cup"] == "red_cup" or
                                                     state.positions_to_objects["workspace_blue_cup"] == "blue_cup"):
                preffered_actions.append((PickUpAction(), self.n_visits_init, self.v_init / 2))

        neighbors = self.prm.get_neighbors(state.robot_cfg_id)

        # if red cup not in place and detected, moving toward it is a preferred action
        if state.positions_to_objects["workspace_red_cup"] != "red_cup" and "red_cup" in detected_positions.values():
            red_cup_position = [pos for pos, obj in detected_positions.items() if obj == "red_cup"][0]
            best_neighbour = self.find_best_neighbour(neighbors, red_cup_position)
            if best_neighbour is not None:
                preffered_actions.append((MoveAction(best_neighbour), self.n_visits_init, self.v_init))

        # same for blue cup
        if state.positions_to_objects["workspace_blue_cup"] != "blue_cup" and "blue_cup" in detected_positions.values():
            blue_cup_position = [pos for pos, obj in detected_positions.items() if obj == "blue_cup"][0]
            best_neighbour = self.find_best_neighbour(neighbors, blue_cup_position)
            if best_neighbour is not None:
                preffered_actions.append((MoveAction(best_neighbour), self.n_visits_init, self.v_init))

        # same for can but with lowe value as if there are other object in the workspace
        # it's better to collect them first:
        if state.positions_to_objects["workspace_can"] != "soda_can" and "soda_can" in detected_positions.values():
            can_position = [pos for pos, obj in detected_positions.items() if obj == "soda_can"][0]
            best_neighbour = self.find_best_neighbour(neighbors, can_position)
            if best_neighbour is not None:
                preffered_actions.append((MoveAction(best_neighbour), self.n_visits_init, self.v_init / 2))

        return preffered_actions

    def get_preferred_actions_gripper_full(self, state, history):

        # if holding red cup move toward it's position, similarely for blue:
        if state.held_object == "red_cup" and \
                state.robot_cfg_id != self.prm_interesting_configs["workspace_red_cup"][0]:
            best_neighbour = self.find_best_neighbour(self.prm.get_neighbors(state.robot_cfg_id), "workspace_red_cup")
            if best_neighbour is not None:
                return [(MoveAction(best_neighbour), self.n_visits_init, self.v_init)]
        if state.held_object == "blue_cup" and \
                state.robot_cfg_id != self.prm_interesting_configs["workspace_red_cup"][0]:
            best_neighbour = self.find_best_neighbour(self.prm.get_neighbors(state.robot_cfg_id), "workspace_blue_cup")
            if best_neighbour is not None:
                return [(MoveAction(best_neighbour), self.n_visits_init, self.v_init)]

        preferred_actions = []
        # if holding can and not in pour config, add preferred actions toward blue and red if cups are there
        if state.held_object == "soda_can" and \
                state.robot_cfg_id not in self.prm_y_up_interesting_configs["pre_pour_red"] and \
                state.robot_cfg_id not in self.prm_y_up_interesting_configs["pre_pour_blue"]:
            neighbors = self.prm_y_up.get_neighbors(state.robot_cfg_id)
            if state.positions_to_objects["workspace_red_cup"] == "red_cup" and not state.red_cup_full:
                best_neighbour = self.find_best_neighbour(neighbors, "pre_pour_red")
                if best_neighbour is not None:
                    preferred_actions.append((MoveAction(best_neighbour), self.n_visits_init, self.v_init))
            if state.positions_to_objects["workspace_blue_cup"] == "blue_cup" and not state.blue_cup_full:
                best_neighbour = self.find_best_neighbour(neighbors, "pre_pour_blue")
                if best_neighbour is not None:
                    preferred_actions.append((MoveAction(best_neighbour), self.n_visits_init, self.v_init))

        # otherwise just move toward trash and finish ...
        elif state.held_object == "soda_can":
            if state.robot_cfg_id == self.prm_y_up_interesting_configs["trash"][0]:
                preferred_actions.append((TrashAction(), self.n_visits_init, self.v_init))
            best_neighbour = self.find_best_neighbour(self.prm_y_up.get_neighbors(state.robot_cfg_id), "trash")
            if best_neighbour is not None:
                preferred_actions.append((MoveAction(best_neighbour), self.n_visits_init, self.v_init))

        return preferred_actions

    def find_best_neighbour(self, neighbors, position):
        best_neighbour = None
        best_distance = float("inf")
        for neighbour in neighbors:
            distance = self.prm_distances.get((neighbour, position), float("inf"))
            if distance < best_distance:
                best_distance = distance
                best_neighbour = neighbour
        return best_neighbour

    def generate_cache_for_sense_best_neighbour(self):
        res = {}
        t = time.time()
        print("Generating cache for sense best neighbour")
        for sense_vid, config_name in self.vid_to_sensing_config_name.items():
            for vid in range(len(self.prm.vertices)):
                if vid == sense_vid:
                    res[(vid, config_name)] = 0
                best_neighbour = self.find_best_neighbour(self.prm.get_neighbors(vid), config_name)
                res[(vid, config_name)] = best_neighbour
        print(f"Cache generated in {time.time() - t} seconds")
        return res

class POMRolloutPolicy(POMPolicyModel):
    def __init__(self, prm: ManipulationPRM, prm_y_up: ManipulationPRM, vid_to_sensing_config_name: dict,
                 preferred_actions: POMActionPrior):
        super().__init__(prm, prm_y_up, vid_to_sensing_config_name)
        self.preferred_actions = preferred_actions

    def rollout(self, state, history=None):
        if history is None:
            history = []

        preferred_actions = self.preferred_actions.get_preferred_actions(state, history)
        if preferred_actions:
            return random.choice([action for action, _, _ in preferred_actions])

        return super().rollout(state, history)
