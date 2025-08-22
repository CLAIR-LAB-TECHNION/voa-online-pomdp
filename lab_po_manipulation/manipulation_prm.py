import time
from typing import List, Dict, Tuple, Optional, Sequence
import numpy as np
from klampt.math import se3, so3
from scipy.spatial.transform import Rotation as R
from lab_po_manipulation.poman_motion_planner import POManMotionPlanner
from lab_po_manipulation.env_configurations.objects_and_positions import ItemPosition, pre_pour_config_red, \
    pre_pour_config_blue, trash_config
from lab_po_manipulation.prm import default_joint_limits_high, default_joint_limits_low, PRM, y_up_constraint


class ManipulationPRM(PRM):
    def __init__(self, motion_planner, robot_name: str, positions_list: List[ItemPosition], n_samples=1000,
                 k_neighbors=10, max_edge_distance=2.0, eps=1e-2,
                 joint_limits_high=default_joint_limits_high,
                 joint_limits_low=default_joint_limits_low,
                 constraint_func=None):

        super().__init__(motion_planner, robot_name, n_samples, k_neighbors,
                         max_edge_distance, eps, joint_limits_high, joint_limits_low,
                         constraint_func)

        self.positions = positions_list
        self.grasp_configs = {}  # Maps vertex_id to grasp metadata
        self.interest_configs = []  # list of additional vertices that are important for manipulation (name, v_id)

    def add_interesting_config(self, name: str, configs):
        vid = self.add_vertex(configs)
        self.interest_configs.append((name, vid))

    def get_all_interesting_configs(self):
        """ including grasp and interest configs"""
        # get (name, vid) format for both grasp configs and interest configs
        grasp_configs = [(metadata['position_name'], vid) for vid, metadata in self.grasp_configs.items()]
        return grasp_configs + self.interest_configs

    def get_all_interesting_configs_as_dict(self):
        """
        including grasp and interest configs. returns dict with name as key and vid list as value
        (there may be multiple configs for a single name)
        """
        interest_configs_list = self.get_all_interesting_configs()
        interest_configs_dict = {}
        for name, vid in interest_configs_list:
            if name not in interest_configs_dict:
                interest_configs_dict[name] = []
            interest_configs_dict[name].append(vid)
        return interest_configs_dict

    def _compute_grasp_transform(self, position: Sequence[float],
                                 grasp_offset: Sequence[float],
                                 grasp_rotation: float) -> tuple:
        """
        Compute se3 transform for grasp configuration
        Returns tuple of (R, t) where R is 3x3 rotation matrix and t is translation
        """
        # Compute approach vector from offset
        approach_vector = -np.array(grasp_offset)
        approach_vector = approach_vector / np.linalg.norm(approach_vector)

        # Create rotation matrix
        mat_z_col = approach_vector
        world_z = np.array([0, 0, 1])

        # Project world Z onto plane perpendicular to approach vector
        proj = world_z - (np.dot(world_z, mat_z_col) * mat_z_col)
        if np.linalg.norm(proj) < 1e-5:
            mat_y_col = np.array([0, -1, 0])
        else:
            mat_y_col = -proj / np.linalg.norm(proj)

        mat_x_col = np.cross(mat_y_col, mat_z_col)

        rotation_matrix = np.array([mat_x_col, mat_y_col, mat_z_col])
        rotation_matrix = rotation_matrix @ R.from_euler("z", grasp_rotation).as_matrix()

        grasp_position = np.array(position) + np.array(grasp_offset)

        # Convert to se3 format
        return (rotation_matrix.flatten().tolist(), grasp_position)

    def _find_ik_solutions(self, transform: tuple, max_solutions: int = 2) -> List[List[float]]:
        """
        Find multiple IK solutions for a given transform
        Returns list of valid configurations
        """
        solutions = []
        attempts = 0
        max_attempts = max_solutions * 500  # Allow more attempts than desired solutions

        joint_limits_l = self.mp.klampt_to_config6d(self.robot.getJointLimits()[0])
        joint_limits_h = self.mp.klampt_to_config6d(self.robot.getJointLimits()[1])

        while len(solutions) < max_solutions and attempts < max_attempts:
            # Try from random start config
            q_near = np.random.uniform(joint_limits_l, joint_limits_h)

            config = self.mp.ik_solve(self.robot_name, transform, start_config=q_near)

            if config is not None and self.mp.is_config_feasible(self.robot_name, config) and \
                    np.all(config >= joint_limits_l) and np.all(config <= joint_limits_h):
                # Check if solution is sufficiently different from existing ones
                is_new = True
                for existing_sol in solutions:
                    if np.allclose(config, existing_sol, atol=1e-2):
                        is_new = False
                        break

                if is_new:
                    solutions.append(config)

            attempts += 1

        if not solutions:
            print(f"Warning: No IK solutions found for transform")
        elif len(solutions) < max_solutions:
            print(f"Warning: Only found {len(solutions)} IK solutions (wanted {max_solutions})")

        return solutions

    def _add_grasp_configs(self):
        """
        Add grasp configurations to the roadmap for all positions
        """
        # check if already built?
        if self.grasp_configs:
            print("Grasp configurations already added to roadmap. If you wanna add additional need to implement this")
            return
        print("Adding grasp configurations to roadmap...")

        # remove graps object attachment as it collides with the objects protection obstacles so no solution are found.
        # wiil return it in the end of this method
        self.mp.remove_grasped_object_attachment()

        for position in self.positions:
            position_name = position.name
            position_coords = position.position

            for grasp_idx, (grasp_offset, grasp_rotation) in enumerate(position.grasps_offset_rz):
                # Compute grasp transform
                transform = self._compute_grasp_transform(position_coords, grasp_offset, grasp_rotation)

                # Find IK solutions
                ik_solutions = self._find_ik_solutions(transform)

                if not ik_solutions:
                    print(f"Warning: No valid IK solutions found for {position_name}, grasp {grasp_idx}")
                    continue

                # Add each IK solution as a vertex
                for config in ik_solutions:
                    vertex_id = self.add_vertex(config)

                    # Store grasp metadata
                    self.grasp_configs[vertex_id] = {
                        'position_name': position_name,
                        'position': position_coords,
                        'grasp_offset': grasp_offset,
                        'grasp_rotation': grasp_rotation
                    }

        print(f"Added {len(self.grasp_configs)} grasp configurations")

        self.mp.add_grasped_object_attachment()

    def copy_grasp_configs_from(self, other_prm: 'ManipulationPRM'):
        """
        Instead of generating new grasp configs, copy them from another PRM
        """
        print("Copying grasp configurations from other PRM...")

        # First verify these configs satisfy our constraint
        for other_vid, metadata in other_prm.grasp_configs.items():
            config = other_prm.vertices[other_vid]

            # Add vertex and metadata
            vertex_id = self.add_vertex(config, remove_grasped_object_attachment_for_feasibility=True)
            self.grasp_configs[vertex_id] = metadata.copy()

        print(f"Copied {len(self.grasp_configs)} grasp configurations")

        self._check_grasp_connectivity()

    def add_grasp_configs_to_roadmap(self):
        grasp_start = time.time()
        self._add_grasp_configs()
        grasp_time = time.time() - grasp_start

        # Add grasp time to build times
        self.build_times['grasp'] = grasp_time
        self.build_times['total'] += grasp_time

        # Check for disconnected grasp configurations
        self._check_grasp_connectivity()

    def _check_grasp_connectivity(self):
        """
        Check and report connectivity of grasp configurations
        """
        disconnected_grasps = []

        # Check each grasp config
        for vertex_id in self.grasp_configs:
            neighbors = self.vertex_id_to_edges.get(vertex_id, [])
            if not neighbors:
                metadata = self.grasp_configs[vertex_id]
                disconnected_grasps.append((vertex_id, metadata['position_name']))

        if disconnected_grasps:
            print("\nWarning: Found disconnected grasp configurations:")
            for vertex_id, pos_name in disconnected_grasps:
                print(f"  Position: {pos_name}, Vertex ID: {vertex_id}")
            # just remove it from the grasp_configs
            for vertex_id, pos_name in disconnected_grasps:
                del self.grasp_configs[vertex_id]

    def is_grasp_config(self, config: List[float]) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a configuration is a grasp configuration
        Returns (is_grasp, metadata)
        """
        config_tuple = tuple(config)
        vertex_id = self.vertex_to_id.get(config_tuple)

        if vertex_id is None or vertex_id not in self.grasp_configs:
            return False, None

        return True, self.grasp_configs[vertex_id]

    def get_grasp_configs_for_position(self, position_name: str) -> List[Tuple[int, List[float], Dict]]:
        """
        Get all grasp configurations for a specific position
        Returns list of (vertex_id,config, metadata) tuples
        """
        grasp_configs = []

        for vertex_id, metadata in self.grasp_configs.items():
            if metadata['position_name'] == position_name:
                grasp_configs.append((vertex_id, self.vertices[vertex_id], metadata))

        return grasp_configs

    def get_shortest_path_to_position(self, start_vertex_idx, position_name) -> Tuple[List, int, Dict]:
        """
        find paths to all grasp configs for a position and return the shortest one
        returns the path, goal_vertex_id, goal_metadata
        """
        grasp_configs = self.get_grasp_configs_for_position(position_name)
        if not grasp_configs:
            return [], -1, {}

        shortest_path = []
        goal_vertex_id = None
        goal_metadata = None
        shortest_path_length = float('inf')
        for vertex_id, _, metadata in grasp_configs:
            path = self.find_path_by_vertices_id(start_vertex_idx, vertex_id)
            if path and len(path) < shortest_path_length:
                shortest_path = path
                shortest_path_length = len(path)
                goal_vertex_id = vertex_id
                goal_metadata = metadata

        return shortest_path, goal_vertex_id, goal_metadata

    def analyze_grasp_coverage(self):
        """
        Analyze which positions have valid grasp configurations.
        Returns:
            - List of position names with no valid grasp configurations
            - Summary statistics about grasp configurations
        """
        ungraspable_positions = []
        grasp_stats = {}

        for position in self.positions:
            grasp_configs = self.get_grasp_configs_for_position(position.name)

            if not grasp_configs:
                ungraspable_positions.append(position.name)

            grasp_stats[position.name] = len(grasp_configs)

        # Print summary
        print("\nGrasp Coverage Analysis:")
        print(f"Total positions: {len(self.positions)}")
        print(f"Positions with valid grasps: {len(self.positions) - len(ungraspable_positions)}")
        if ungraspable_positions:
            print("\nUngraspable positions:")
            for pos_name in ungraspable_positions:
                print(f"  - {pos_name}")

        print("\nGrasps per position:")
        for pos_name, count in grasp_stats.items():
            print(f"  {pos_name}: {count} grasp configurations")

        return ungraspable_positions, grasp_stats

    def analyze_paths_to_position(self, target_position: str):
        """
        Compute mean path length from all grasp configurations to grasp configurations
        of a specific position.
        """
        # Get target grasp configurations
        target_configs = self.get_grasp_configs_for_position(target_position)
        if not target_configs:
            print(f"No grasp configurations found for position {target_position}")
            return

        # Store results for each position
        results = {}

        # For each position, compute paths from its grasps to target grasps
        for position in self.positions:
            if position.name == target_position:
                continue

            source_configs = self.get_grasp_configs_for_position(position.name)
            if not source_configs:
                continue

            path_lengths = []
            for _, source_config, _ in source_configs:
                for _, target_config, _ in target_configs:
                    path = self.find_path(source_config, target_config)
                    path_length = len(path) - 1 if path else float('inf')  # -1 because path includes start node
                    path_lengths.append(path_length)

            if path_lengths:
                # Filter out unreachable paths (inf)
                reachable_paths = [l for l in path_lengths if l != float('inf')]
                if reachable_paths:
                    mean_length = sum(reachable_paths) / len(reachable_paths)
                    unreachable = len(path_lengths) - len(reachable_paths)
                else:
                    mean_length = float('inf')
                    unreachable = len(path_lengths)

                results[position.name] = {
                    'mean_length': mean_length,
                    'unreachable': unreachable,
                    'total_paths': len(path_lengths)
                }

        # Print results
        print(f"\nPath Analysis to {target_position}:")
        print("-" * 50)
        for pos_name, data in results.items():
            if data['mean_length'] == float('inf'):
                print(f"{pos_name:20} : No reachable paths")
            else:
                print(f"{pos_name:20} : Mean path length = {data['mean_length']:.2f}")
                if data['unreachable'] > 0:
                    print(f"{'':22}({data['unreachable']}/{data['total_paths']} paths unreachable)")

        return results

    def optimize_prm(self):
        """
        Optimize PRM by:
        1. Finding all relevant shortest paths between interesting points
        2. Optimizing these paths with shortcuts
        3. Creating a new PRM with only the essential vertices
        Returns new optimized ManipulationPRM
        """
        # make sure home config is interesting
        if not any(vid == self.home_cfg_idx for _, vid in self.interest_configs):
            self.interest_configs.append(("home", self.home_cfg_idx))

        # Get all interesting name-vertex pairs
        interesting_configs = self.get_all_interesting_configs()

        # Store all shortest paths
        print("Finding shortest paths between interesting configurations...")
        paths = []
        total_pairs = len(interesting_configs) * (len(interesting_configs) - 1) // 2
        count = 0

        for i, (name1, start_vid) in enumerate(interesting_configs):
            for name2, end_vid in interesting_configs[i + 1:]:
                count += 1
                if count % 100 == 0:
                    print(f"Processing pair {count}/{total_pairs}")

                # Skip if they are different grasp configs of same position
                if start_vid in self.grasp_configs and end_vid in self.grasp_configs:
                    if self.grasp_configs[start_vid]['position_name'] == self.grasp_configs[end_vid]['position_name']:
                        continue

                path = self.find_path_by_vertices_id(start_vid, end_vid)
                if path:  # Only store if path exists
                    paths.append(path)

        if not paths:
            print("Warning: No valid paths found between interesting configurations!")
            return None

        # Optimize paths with shortcuts
        print("\nOptimizing paths with shortcuts...")
        optimized_paths = self._optimize_paths(paths)

        # Create new PRM with only vertices from optimized paths
        return self._create_minimal_prm(optimized_paths)

    def _optimize_paths(self, paths):
        """
        Try to optimize paths by finding shortcuts
        Returns list of optimized paths
        """
        optimized_paths = []
        for path in paths:
            optimized_path = self._optimize_single_path(path)
            optimized_paths.append(optimized_path)

        return optimized_paths

    def _optimize_single_path(self, path):
        """
        Try to find shortcuts in a single path
        Returns optimized path
        """
        if len(path) <= 2:
            return path

        # Keep trying to optimize until no more shortcuts found
        while True:
            found_shortcut = False
            new_path = path.copy()

            # Try shortcuts between all pairs of vertices in path
            for i in range(len(path)):
                for j in range(i + 2, len(path)):  # j > i+1 to skip consecutive vertices
                    if self.check_edge_validity(path[i], path[j]):
                        # Found valid shortcut, update path
                        new_path = path[:i + 1] + path[j:]
                        found_shortcut = True
                        break
                if found_shortcut:
                    break

            if not found_shortcut:
                break
            path = new_path

        return path

    def _create_minimal_prm(self, optimized_paths):
        """
        Create new PRM containing only vertices from optimized paths
        Returns new ManipulationPRM instance
        """
        print("Creating minimal PRM...")

        # Get unique vertices from all paths
        unique_vertices = set()
        for path in optimized_paths:
            for config in path:
                unique_vertices.add(tuple(config))

        # Create new PRM
        minimal_prm = ManipulationPRM(self.mp, self.robot_name, self.positions,
                                      n_samples=-1,  # No sampling
                                      k_neighbors=self.k_neighbors,
                                      max_edge_distance=self.max_edge_distance,
                                      eps=self.eps,
                                      constraint_func=self.constraint_func)

        # Add vertices directly to list
        print("Adding vertices...")
        vertex_to_new_id = {}
        minimal_prm.vertices = []
        for vertex in unique_vertices:
            new_id = len(minimal_prm.vertices)
            minimal_prm.vertices.append(list(vertex))
            vertex_to_new_id[vertex] = new_id

        # Copy metadata
        print("Copying metadata...")
        for old_vid, metadata in self.grasp_configs.items():
            old_vertex = tuple(self.vertices[old_vid])
            if old_vertex in vertex_to_new_id:
                new_vid = vertex_to_new_id[old_vertex]
                minimal_prm.grasp_configs[new_vid] = metadata.copy()

        print("Copying interest configs...")
        original_interest_count = len(self.interest_configs)
        copied_interest_count = 0
        for name, old_vid in self.interest_configs:
            old_vertex = tuple(self.vertices[old_vid])
            if old_vertex in vertex_to_new_id:
                new_vid = vertex_to_new_id[old_vertex]
                minimal_prm.interest_configs.append((name, new_vid))
                copied_interest_count += 1
            else:
                print(f"Warning: Interest config {old_vid} ({name}) not found in optimized paths!")

        if copied_interest_count != original_interest_count:
            print(f"Warning: Only copied {copied_interest_count}/{original_interest_count} interest configs")

        # Add edges from optimized paths
        print("Adding edges...")
        minimal_prm.edges = []
        edges_added = set()
        for path in optimized_paths:
            for i in range(len(path) - 1):
                v1 = tuple(path[i])
                v2 = tuple(path[i + 1])
                edge = (vertex_to_new_id[v1], vertex_to_new_id[v2])
                rev_edge = (vertex_to_new_id[v2], vertex_to_new_id[v1])
                if edge not in edges_added:
                    minimal_prm.edges.append(edge)
                    minimal_prm.edges.append(rev_edge)
                    edges_added.add(edge)
                    edges_added.add(rev_edge)

        # Now build neighbor maps once at the end
        minimal_prm.build_neighbor_maps()

        # set home config index
        minimal_prm.home_cfg_idx = vertex_to_new_id[tuple(self.vertices[self.home_cfg_idx])]

        print(f"\nMinimal PRM created with {len(minimal_prm.vertices)} vertices and {len(minimal_prm.edges)} edges")
        print(f"Original PRM had {len(self.vertices)} vertices and {len(self.edges)} edges")

        # Final verification
        print("\nVerifying minimal PRM...")
        ungraspable_positions, _ = minimal_prm.analyze_grasp_coverage()
        if ungraspable_positions:
            print("Warning: Some positions became ungraspable in minimal PRM!")

        for name, vid in minimal_prm.interest_configs:
            if not minimal_prm.vertex_id_to_edges[vid]:
                print(f"Warning: Interest config {vid} ({name}) is disconnected in minimal PRM!")

        for path in optimized_paths:
            start = path[0]
            end = path[-1]
            new_path = minimal_prm.find_path(start, end)
            if not new_path:
                print(f"Warning: Lost connectivity between vertices in minimal PRM!")
                break

        return minimal_prm

    def compute_distances_to_interesting_configs(self) -> Dict[Tuple[int, str], float]:
        """
        Compute shortest path lengths from all vertices to all interesting configs.
        Returns dict mapping (vertex_id, interesting_name) -> shortest path length
        """
        print("Computing distances to interesting configurations...")
        distances = {}

        # Get mapping of names to vertex ids for interesting configs
        interesting_dict = self.get_all_interesting_configs_as_dict()
        total_vertices = len(self.vertices)

        # For each vertex
        for vid in range(total_vertices):
            if vid % 100 == 0:
                print(f"Processing vertex {vid}/{total_vertices}")

            # For each interesting position
            for name, target_vids in interesting_dict.items():
                shortest_length = float('inf')

                # Try path to each config of this position
                for target_vid in target_vids:
                    if vid == target_vid:
                        shortest_length = 0
                        break

                    path = self.find_path_by_vertices_id(vid, target_vid)
                    if path:
                        path_length = len(path) - 1  # -1 because path includes start
                        if path_length < shortest_length:
                            shortest_length = path_length

                distances[(vid, name)] = shortest_length

        print(f"Computed distances from {total_vertices} vertices to {len(interesting_dict)} interesting positions")
        return distances

    def save_roadmap(self, filename: str):
        """Save roadmap to file using pickle"""
        import pickle

        data = {
            'vertices': self.vertices,
            'edges': self.edges,

            'robot_name': self.robot_name,
            'positions': self.positions,
            'grasp_configs': self.grasp_configs,
            'home_cfg_idx': self.home_cfg_idx,
            'interest_configs': self.interest_configs,
            'params': {
                'n_samples': self.n_samples,
                'k_neighbors': self.k_neighbors,
                'max_edge_distance': self.max_edge_distance
            }
        }

        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            print(f"Successfully saved roadmap to {filename}")
        except Exception as e:
            print(f"Failed to save roadmap: {str(e)}")

    @classmethod
    def load_roadmap(cls, motion_planner, filename: str):
        """Load roadmap from file using pickle"""
        import pickle

        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Create new PRM instance with saved parameters
        prm = cls(motion_planner,
                  robot_name=data['robot_name'],
                  positions_list=data['positions'],
                  n_samples=data['params']['n_samples'],
                  k_neighbors=data['params']['k_neighbors'],
                  max_edge_distance=data['params']['max_edge_distance'])

        prm.vertices = data['vertices']
        prm.edges = data['edges']
        prm.grasp_configs = data['grasp_configs']
        prm.home_cfg_idx = data['home_cfg_idx']
        prm.interest_configs = data['interest_configs']

        print(f"Loaded roadmap with {len(prm.vertices)} vertices and {len(prm.edges)} edges")

        prm.build_neighbor_maps()
        return prm




if __name__ == "__main__":
    from lab_po_manipulation.env_configurations.objects_and_positions import positions

    planner = POManMotionPlanner()
    for i in range(1, 5):
        if i in [1]:
            continue
        planner.add_mobile_obstacle_by_number(i)

    prm = ManipulationPRM(planner, "ur5e_1", positions, n_samples=7000,
                          k_neighbors=400, max_edge_distance=5., eps=0.3,
                          constraint_func=y_up_constraint
                          )
    prm.build_roadmap()

    print("Adding interesting configurations...")
    prm.add_interesting_config("red_cup_pour", pre_pour_config_red)
    prm.add_interesting_config("blue_cup_pour", pre_pour_config_blue)
    if prm.constraint_func is None:
        prm.add_interesting_config("trash", trash_config)

    print("Adding grasp configurations...")
    xlims = (-1.5, -0.15)
    ylims = (-0.5, 0.5)
    zlims = (0, 0.8)
    prm.add_workspace_vertices(80, xlims, ylims, zlims)
    xlims = (-1.5, -0.35)
    ylims = (-0.4, 0.4)
    zlims = (0, 0.5)
    prm.add_workspace_vertices(40, xlims, ylims, zlims)

    prm.add_grasp_configs_to_roadmap()

    prm.save_roadmap("roadmap_ur5e_1_with_grasps_7000_400_cons.npy")

    prm = prm.optimize_prm()

    prm.print_statistics()

    ungraspable_positions, grasp_stats = prm.analyze_grasp_coverage()

    # planner.visualize()
    # prm.visualize_roadmap_ee_poses()

    prm.save_roadmap("roadmap_ur5e_1_with_grasps_7000_400_cons_opt.npy")

    # # Test loading
    # loaded_prm = ManipulationPRM.load_roadmap(planner, "roadmap_ur5e_1_with_grasps_cons.npy")
    # loaded_prm.print_statistics()
    # loaded_prm.analyze_grasp_coverage()