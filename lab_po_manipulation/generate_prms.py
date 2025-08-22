import typer
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional
import multiprocessing as mp
from lab_po_manipulation.poman_motion_planner import POManMotionPlanner
from lab_po_manipulation.manipulation_prm import ManipulationPRM
from lab_po_manipulation.env_configurations.objects_and_positions import (
    positions, pre_pour_config_red, pre_pour_config_blue, trash_config, sensing_configs
)
from lab_po_manipulation.prm import y_up_constraint

app = typer.Typer()

PRM_DIR = Path(__file__).parent / "env_configurations" / "prms"
PRM_DIR.mkdir(exist_ok=True)


def get_prm_filename(obstacles_to_exclude: List[int], constrained: bool) -> str:
    assert len(obstacles_to_exclude) <= 1, "not supported"
    if not obstacles_to_exclude:
        base = "prm_all_obs"
    else:
        base = f"prm_no_obs{obstacles_to_exclude[0]}"
    if constrained:
        base += "_y_up"
    return str(PRM_DIR / f"{base}.npy")

@app.command()
def build_single_prm(
    obstacles_to_exclude: List[int] = typer.Option([]), # Changed this line
    constrained: bool = False,
    n_samples: int = 5000,
    k_neighbors: int = 500,
    max_edge_distance: float = 5.0,
    eps: float = 0.1,
    n_extended_workspace: int = 80,
    n_small_workspace: int = 40
) -> None:
    """Build and save a single PRM"""
    assert len(obstacles_to_exclude) <= 1, "not supported"

    planner = POManMotionPlanner()

    # Add obstacles except excluded ones
    for i in range(1, 5):
        if i not in obstacles_to_exclude:
            planner.add_mobile_obstacle_by_number(i)

    prm_positions = positions

    prm = ManipulationPRM(
        planner, "ur5e_1", prm_positions,
        n_samples=n_samples,
        k_neighbors=k_neighbors,
        max_edge_distance=max_edge_distance,
        eps=eps,
        constraint_func=y_up_constraint if constrained else None
    )

    prm.build_roadmap()

    print("Adding manipulation special configs...")
    prm.add_interesting_config("pre_pour_red", pre_pour_config_red)
    prm.add_interesting_config("pre_pour_blue", pre_pour_config_blue)
    prm.add_interesting_config("trash", trash_config)
    if not constrained:
        print("Adding sensing configs...")
        for sense_config in sensing_configs:
            prm.add_interesting_config(sense_config.name, sense_config.robot_config)

    # Add workspace vertices - for constrained only add extended workspace
    prm.add_workspace_vertices(n_extended_workspace, (-1.5, -0.15), (-0.5, 0.5), (0, 0.8))

    prm.add_workspace_vertices(n_small_workspace, (-1.5, -0.35), (-0.4, 0.4), (0, 0.5))

    # For constrained PRM, copy grasp configs from unconstrained
    if constrained:
        # Load corresponding unconstrained PRM
        unconstrained_filename = get_prm_filename(obstacles_to_exclude, False)
        try:
            unconstrained_prm = ManipulationPRM.load_roadmap(planner, unconstrained_filename)
            prm.copy_grasp_configs_from(unconstrained_prm)
        except FileNotFoundError:
            print(f"Warning: Could not find unconstrained PRM {unconstrained_filename}")
            exit(1)
    else:
        prm.add_grasp_configs_to_roadmap()

    prm = prm.optimize_prm()

    filename = get_prm_filename(obstacles_to_exclude, constrained)
    print(f"Saving PRM to {filename}")
    prm.save_roadmap(filename)


@app.command()
def build_prms(
        constrained: bool = typer.Option(..., help="Whether to use y_up constraint"),
        n_samples: int = typer.Option(4000, help="Number of PRM samples"),
        k_neighbors: int = typer.Option(400, help="Number of neighbors for connections"),
        max_edge_distance: float = typer.Option(5.0, help="Max distance for edge connections"),
        eps: float = typer.Option(0.1, help="Resolution for collision checking"),
        n_extended_workspace: int = typer.Option(40, help="Number of vertices in extended workspace"),
        n_small_workspace: int = typer.Option(20, help="Number of vertices in smaller workspace")
):
    """Build PRMs (must build unconstrained before constrained)"""
    if constrained:
        # Verify unconstrained PRMs exist
        for obs in [[], [1], [2], [3], [4]]:
            filename = get_prm_filename(obs, False)
            if not Path(filename).exists():
                print(f"Error: Unconstrained PRM {filename} not found!")
                print("Please build unconstrained PRMs first")
                return

    configs = [
        [],  # all obstacles
        [1],  # without obstacle 1
        [2],  # without obstacle 2
        [3],  # without obstacle 3
        [4]  # without obstacle 4
    ]

    print(f"Building {'constrained' if constrained else 'unconstrained'} PRMs with parameters:")
    print(f"n_samples: {n_samples}")
    print(f"k_neighbors: {k_neighbors}")
    print(f"max_edge_distance: {max_edge_distance}")
    print(f"eps: {eps}")

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = []
        for obstacles_to_exclude in configs:
            future = executor.submit(
                build_single_prm,
                obstacles_to_exclude,
                constrained,
                n_samples,
                k_neighbors,
                max_edge_distance,
                eps,
                n_extended_workspace,
                n_small_workspace
            )
            futures.append(future)

        for future in futures:
            future.result()

    if constrained:
        clean_unreachable_grasps()

@app.command()
def load_prm(
        obstacles_to_exclude: List[int] = typer.Option([], help="List of obstacles to exclude"),
        constrained: bool = typer.Option(..., help="Whether to use y_up constraint"),
) -> ManipulationPRM:
    """Load PRM for given configuration"""
    planner = POManMotionPlanner()

    # Add obstacles except excluded ones
    for i in range(1, 5):
        if i not in obstacles_to_exclude:
            planner.add_mobile_obstacle_by_number(i)

    # Load PRM
    filename = get_prm_filename(obstacles_to_exclude, constrained)
    return ManipulationPRM.load_roadmap(planner, filename)

@app.command()
def test_path_to_pour_constrained():
    """
    load all roadmap with y up constraint, and check whether there is a path between two top shelf and to middle shelfs
    to two pour configurations. Print the results for each roadmap
    """
    for i in range(1, 5):
        prm = load_prm([i], True)
        interest_configs = prm.get_all_interesting_configs_as_dict()
        pour_red_vid = interest_configs["pre_pour_red"][0]
        pour_blue_vid = interest_configs["pre_pour_blue"][0]

        print()
        print("----PRM without obstacle", i)
        for pname in ["top_shelf_left", "top_shelf_right", "middle_shelf_left", "middle_shelf", "workspace_can",
                      "bottom_left", "bottom_right"]:
            path, _, _ = prm.get_shortest_path_to_position(pour_red_vid, pname)
            if path:
                print(f"Path to {pname} from pour red exists. Length: {len(path)}")
            else:
                print(f"\033[93m Path to {pname} from pour red does not exist \033[0m")
            path, _, _ = prm.get_shortest_path_to_position(pour_blue_vid, pname)
            if path:
                print(f"Path to {pname} from pour blue exists. Length: {len(path)}")
            else:
                print(f"\033[93m Path to {pname} from pour blue does not exist \033[0m")

        # also validate path to trash config
        trash_vid = interest_configs["trash"][0]
        path = prm.find_path_by_vertices_id(pour_red_vid, trash_vid)
        if path:
            print(f"Path to trash from pour red exists. Length: {len(path)}")
        else:
            print(f"\033[93m Path to trash from pour red does not exist \033[0m")
        # that's enough... if there's path to red there's path to blue

        # same for all obstacles prm:
        prm = load_prm([], True)
        interest_configs = prm.get_all_interesting_configs_as_dict()
        pour_red_vid = interest_configs["pre_pour_red"][0]
        pour_blue_vid = interest_configs["pre_pour_blue"][0]

    print()
    print("----PRM with all obstacles")
    for pname in ["top_shelf_left", "top_shelf_right", "middle_shelf_left", "middle_shelf", "workspace_can",
                  "bottom_left", "bottom_right"]:
        path, _, _ = prm.get_shortest_path_to_position(pour_red_vid, pname)
        if path:
            print(f"Path to {pname} from pour red exists. Length: {len(path)}")
        else:
            print(f"\033[93m Path to {pname} from pour red does not exist \033[0m")
        path, _, _ = prm.get_shortest_path_to_position(pour_blue_vid, pname)
        if path:
            print(f"Path to {pname} from pour blue exists. Length: {len(path)}")
        else:
            print(f"\033[93m Path to {pname} from pour red does not exist \033[0m")
    # also validate path to trash config
    trash_vid = interest_configs["trash"][0]
    path = prm.find_path_by_vertices_id(pour_red_vid, trash_vid)
    if path:
        print(f"Path to trash from pour red exists. Length: {len(path)}")
    else:
        print(f"\033[93m Path to trash from pour red does not exist \033[0m")
    # that's enough... if there's path to red there's path to blue


@app.command()
def clean_unreachable_grasps() -> None:
    """
    Clean unreachable grasp configurations from both constrained and unconstrained PRMs
    by checking reachability from pre_pour_red in constrained PRMs.
    Only processes shelf and bottom positions.
    Creates backups before modification.
    """
    # Positions to check
    positions_to_clean = [
        "top_shelf_left", "top_shelf_right",
        "middle_shelf_left", "middle_shelf",
        "bottom_left", "bottom_right"
    ]

    configs = [
        [],  # all obstacles
        [1],  # without obstacle 1
        [2],  # without obstacle 2
        [3],  # without obstacle 3
        [4]  # without obstacle 4
    ]

    for obstacles_to_exclude in configs:
        print(f"\nProcessing PRMs without obstacles {obstacles_to_exclude}")

        base_filename = get_prm_filename(obstacles_to_exclude, False)
        constrained_filename = get_prm_filename(obstacles_to_exclude, True)

        try:
            # Create backups
            import shutil
            backup_base = base_filename.replace('.npy', '_backup.npy')
            backup_constrained = constrained_filename.replace('.npy', '_backup.npy')
            shutil.copy2(base_filename, backup_base)
            shutil.copy2(constrained_filename, backup_constrained)
            print(f"Created backups at {backup_base} and {backup_constrained}")

            # Load PRMs
            planner = POManMotionPlanner()
            for i in range(1, 5):
                if i not in obstacles_to_exclude:
                    planner.add_mobile_obstacle_by_number(i)

            base_prm = ManipulationPRM.load_roadmap(planner, base_filename)
            constrained_prm = ManipulationPRM.load_roadmap(planner, constrained_filename)

            # Get pre_pour_red vertex id
            pour_red_vid = constrained_prm.get_all_interesting_configs_as_dict()["pre_pour_red"][0]

            # Find unreachable vertices in constrained PRM (only for specified positions)
            unreachable_vertices = []
            for vertex_id in constrained_prm.grasp_configs:
                position = constrained_prm.grasp_configs[vertex_id]['position_name']
                if position not in positions_to_clean:
                    continue

                path = constrained_prm.find_path_by_vertices_id(pour_red_vid, vertex_id)
                if not path:
                    unreachable_vertices.append(vertex_id)

            # Remove unreachable configs from both PRMs
            removed_configs = []
            for vertex_id in unreachable_vertices:
                config = tuple(constrained_prm.vertices[vertex_id])
                position = constrained_prm.grasp_configs[vertex_id]['position_name']

                # Remove from constrained PRM
                del constrained_prm.grasp_configs[vertex_id]

                # Find and remove matching config from base PRM
                for base_vid in list(base_prm.grasp_configs.keys()):
                    if tuple(base_prm.vertices[base_vid]) == config:
                        del base_prm.grasp_configs[base_vid]
                        removed_configs.append((position, vertex_id, base_vid))
                        break

            # Print results
            print("\nRemoved grasp configs:")
            for position, const_vid, base_vid in removed_configs:
                print(f"Position: {position}, constrained ID: {const_vid}, base ID: {base_vid}")

            # Save modified PRMs
            base_prm.save_roadmap(base_filename)
            constrained_prm.save_roadmap(constrained_filename)
            print(f"\nTotal: Removed {len(removed_configs)} unreachable grasp configs from both PRMs")

        except FileNotFoundError as e:
            print(f"Error: Could not find PRM file - {e}")
        except Exception as e:
            print(f"Error processing PRMs: {e}")


if __name__ == "__main__":
    app()
