import json
import os

import pandas as pd
import pomdp_py

from rocksample_experiments.rocksample_problem import RockSampleProblem, State, RockType
import ast


def get_problem_from_voa_row_and_state(row : pd.Series, n, rocktypes) -> RockSampleProblem:
    rock_locs = ast.literal_eval(row['rock_locations'])
    rock_locs = {eval(k): v for k, v in rock_locs.items()}

    rover_pos = ast.literal_eval(row['rover_position'])
    rover_pos = tuple(rover_pos)

    k = len(rock_locs)

    init_state = State(position=rover_pos, rocktypes=rocktypes, terminal=False)

    # generate init belief:
    belief_path = os.path.join(os.getcwd(), 'configurations', f'particles_k{k}.json')
    with open(belief_path, 'r') as f:
        rocktypes = json.load(f)
    particles = [State(position=init_state.position, rocktypes=rocktypes[i], terminal=False)
                 for i in range(1000)]
    init_belief = pomdp_py.Particles(particles)

    return RockSampleProblem(n=n, k=k, rock_locs=rock_locs, init_state=init_state, init_belief=init_belief)


def sample_problem_from_voa_row(row : pd.Series, n, deterministic_rock_types=None) -> RockSampleProblem:
    k = len(eval(row['rock_locations']))
    if deterministic_rock_types:
        rocktypes = deterministic_rock_types
    else:
        rocktypes = [RockType.random() for _ in range(k)]

    return get_problem_from_voa_row_and_state(row, n, rocktypes)


def get_help_action_from_row(row : pd.Series) -> dict:
    help_action = ast.literal_eval(row['help_actions'])
    help_action = {int(k): v for k, v in help_action.items()}
    return help_action
