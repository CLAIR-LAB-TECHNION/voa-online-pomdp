from klampt import *
from klampt.plan import robotcspace
from klampt.plan import cspace, robotplanning
import numpy as np
import time
from typing import List, Tuple
from klampt.plan.robotplanning import make_space

from lab_po_manipulation.poman_motion_planner import POManMotionPlanner
from lab_ur_stack.motion_planning.motion_planner import MotionPlanner
from collections import deque


# all joints are 2 pi to each direction, but we limit shoulder because it has to go through the table to
# reach the other side, and it creates sets of un connected components. Similarly, we limit the elbow because
# it can't realy do more than pi to each direction. joint one before last is also limited to -pi pi because
# it's also likely to create unconnected components.
default_joint_limits_low = (-2 * np.pi, -3.5, -np.pi, - 2 * np.pi, -np.pi, -2 * np.pi)
default_joint_limits_high = (2 * np.pi, 0.5, np.pi, 2 * np.pi, np.pi, 2 * np.pi)

def y_up_constraint(config, motion_planner, tol=0.5):
    """
    ee constraints y axis direction so when holding can for example, nothing will pour
    """
    ee_pose = motion_planner.get_forward_kinematics("ur5e_1", config)
    ee_orientation = np.array(ee_pose[0]).reshape(3, 3).T
    y_axis = ee_orientation[:, 1]

    # y needs to be facing down
    goal_y = np.array([0, 0, -1])
    return bool(np.dot(y_axis, goal_y) > 1-tol)



class PRM:
    def __init__(self, motion_planner, robot_name, n_samples=1000, k_neighbors=10, max_edge_distance=2.0, eps=1e-2,
                 joint_limits_high=default_joint_limits_high, joint_limits_low=default_joint_limits_low,
                 constraint_func=None):
        """
        Probabilistic Roadmap for robot motion planning
        Args:
            motion_planner: Instance of MotionPlanner class
            robot_name: Name of robot to build roadmap for
            n_samples: Number of random configurations to sample
            k_neighbors: Number of nearest neighbors to try connecting
            max_edge_distance: Maximum allowed distance between configs to attempt connection
            eps: Resolution for collision checking
            constraint_func: Function that takes a configuration and motion planner and returns True if the config satisfies constraints
        """
        self.n_samples = n_samples
        self.k_neighbors = k_neighbors
        self.max_edge_distance = max_edge_distance
        self.constraint_func = constraint_func
        self.mp = motion_planner
        self.robot_name = robot_name
        self.robot = self.mp.robot_name_mapping[robot_name]
        self.eps = eps

        self.robot.setJointLimits(self.mp.config6d_to_klampt(joint_limits_low),
                                  self.mp.config6d_to_klampt(joint_limits_high))

        self.space = make_space(self.mp.world, self.robot)

        # workaround to set space where visibility check works:
        def visibility_space_constraint_func(q):
            return self.constraint_func(self.mp.klampt_to_config6d(q), self.mp)
        xtra_constraints = [visibility_space_constraint_func] if constraint_func is not None else []
        c1 = np.array(self.mp.config6d_to_klampt([0, -np.pi / 2, 0, -np.pi / 2, 0, 0]))
        self.robot.setConfig(c1)
        c2 = c1.copy()
        c2[1:7] += 0.01
        planner = robotplanning.plan_to_config(self.mp.world, self.robot, c2, **self.mp.settings,
                                               edgeCheckResolution=eps,
                                               extraConstraints=xtra_constraints)
        self.space_for_visibility = planner.space

        # Roadmap storage
        self.vertices = []  # List of configurations
        self.edges = []  # List of (vertex_idx1, vertex_idx2)
        self.vertex_id_to_edges = {}  # Maps vertex index to list of neighbor indices
        self.vertex_to_id = {}  # Maps vertex tuple to its index
        self.home_cfg_idx = None

        self.build_times = None

    def sample_random_config(self):
        """Generate random 6D configuration using CSpace sampler"""
        while True:
            q = self.space.sample()
            q = self.mp.klampt_to_config6d(q)
            if self.mp.is_config_feasible(self.robot_name, q) and (
                    self.constraint_func is None or self.constraint_func(q, self.mp)):
                return q

    def build_roadmap(self, add_shortcuts=False):
        """Build the PRM roadmap by sampling and connecting vertices"""
        if self.vertices or self.edges:
            raise ValueError("Roadmap already exists. Clear it before building a new one")

        build_start_time = time.time()

        print(f"Building roadmap with {self.n_samples} samples...")

        # Sample valid configurations
        sampling_start = time.time()
        for i in range(self.n_samples):
            if i % 100 == 0:
                print(f"Sampling vertex {i}/{self.n_samples}".ljust(50), end="\r")
            config = self.sample_random_config()
            self.vertices.append(config)
        sampling_time = time.time() - sampling_start
        print()

        print("Connecting vertices...")
        connection_start = time.time()

        # Convert vertices list to numpy array for faster operations
        vertices_array = np.array(self.vertices)

        num_attempted = 0
        num_successful = 0

        n_vertices = len(self.vertices)
        diff = vertices_array[:, np.newaxis, :] - vertices_array[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)

        for i in range(n_vertices):
            if i % 10 == 0:
                print(f"Processing vertex {i}/{n_vertices}".ljust(75), end="\r")

            dist_to_others = distances[i]
            dist_to_others[i] = float('inf')  # Exclude self
            neighbor_indices = np.argpartition(dist_to_others, self.k_neighbors)[:self.k_neighbors]

            for j in neighbor_indices:
                dist = distances[i, j]
                if dist > self.max_edge_distance:
                    continue

                num_attempted += 1
                if self.check_edge_validity(self.vertices[i], self.vertices[j]):
                    self.edges.append((i, j))
                    num_successful += 1

        self.home_cfg_idx = self.add_vertex([0, -np.pi / 2, 0, -np.pi / 2, 0, 0])

        print()
        print(f"Roadmap built with {len(self.vertices)} vertices and {len(self.edges)} edges")
        print(f"Attempted {num_attempted} connections, {num_successful} successful")

        connection_time = time.time() - connection_start
        total_time = time.time() - build_start_time

        self.build_times = {
            'sampling': sampling_time,
            'connection': connection_time,
            'total': total_time
        }

        self.post_process()

    def add_vertex(self, config: List[float], remove_grasped_object_attachment_for_feasibility=False) -> int:
        """
        Add a new vertex to the roadmap and connect it to k nearest neighbors
        if remove_grasped_object_attachment_for_feasibility is True, the feasibility check
        for adding the config will not consider collisions beteween grasp object protection on the ee
        Returns the index of the new vertex
        """
        if remove_grasped_object_attachment_for_feasibility:
            self.mp.remove_grasped_object_attachment()

        if not self.mp.is_config_feasible(self.robot_name, config) or (
                self.constraint_func is not None and not self.constraint_func(config, self.mp)):
            raise ValueError("Invalid configuration provided. config: ", config)

        # Add the vertex
        new_idx = len(self.vertices)
        self.vertices.append(config)

        # Compute distances to all existing vertices
        vertices_array = np.array(self.vertices)
        diff = np.array(config) - vertices_array
        distances = np.linalg.norm(diff, axis=1)
        distances[new_idx] = float('inf')  # Exclude self

        # Find k nearest neighbors
        neighbor_indices = np.argpartition(distances, self.k_neighbors)[:self.k_neighbors]

        for j in neighbor_indices:
            if distances[j] > self.max_edge_distance:
                continue

            if self.check_edge_validity(config, self.vertices[j]):
                self.edges.append((new_idx, j))
                self.edges.append((j, new_idx))

        self.build_neighbor_maps()  # Update the neighbor maps

        if remove_grasped_object_attachment_for_feasibility:
            self.mp.add_grasped_object_attachment()

        return new_idx

    def add_workspace_vertices(self, n_samples: int,
                               xlims: Tuple[float, float],
                               ylims: Tuple[float, float],
                               zlims: Tuple[float, float]) -> List[int]:
        """
        Add vertices by sampling configurations that place the end effector
        within specified workspace bounds.
        """
        added_vertices = []
        attempts = 0
        max_attempts = n_samples * 100  # Limit total attempts to avoid infinite loop

        while len(added_vertices) < n_samples and attempts < max_attempts:
            attempts += 1

            # Sample random configuration
            config = self.sample_random_config()

            # Check if end effector position is within bounds
            ee_pose = self.mp.get_forward_kinematics(self.robot_name, config)
            ee_pos = ee_pose[1]

            if (xlims[0] <= ee_pos[0] <= xlims[1] and
                    ylims[0] <= ee_pos[1] <= ylims[1] and
                    zlims[0] <= ee_pos[2] <= zlims[1]):

                # Add vertex to roadmap
                vertex_id = self.add_vertex(config)
                added_vertices.append(vertex_id)

                if len(added_vertices) % 10 == 0:
                    print(f"Added {len(added_vertices)}/{n_samples} vertices".ljust(50), end="\r")

        print(f"\nAdded {len(added_vertices)} vertices in {attempts} attempts")
        if len(added_vertices) < n_samples:
            print("Warning: Could not find enough valid configurations in the specified workspace region")

        return added_vertices

    def check_edge_validity(self, config1, config2):
        """Check if direct path between two configs is valid using CSpace"""
        config1_klampt = self.mp.config6d_to_klampt(config1)
        self.robot.setConfig(config1_klampt)
        result = self.space_for_visibility.isVisible(config1, config2)
        return result

    def _dfs_components(self):
        """Helper method for DFS to find connected components
        Returns:
            List[List[int]]: List of components, where each component is a list of vertex indices
        """
        visited = set()
        components = []

        def dfs(v):
            component = []
            stack = [v]
            while stack:
                vertex = stack.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    component.append(vertex)
                    stack.extend(neighbor for neighbor in self.vertex_id_to_edges[vertex]
                                 if neighbor not in visited)
            return component

        for v in range(len(self.vertices)):
            if v not in visited:
                components.append(dfs(v))

        return components

    def post_process(self, add_shortcuts=False, reduce_neighbors=False):
        """
        Run all post-processing steps
        """
        self.add_reverse_edges()
        self.build_neighbor_maps()
        # self.remove_duplicate_edges()  already checked in add_reverse, so no duplicates.
        if add_shortcuts:
            self.add_shortcuts()
        if reduce_neighbors:
            self.reduce_to_k_neighbors()
        self.check_connectivity()

    def add_reverse_edges(self):
        """Add reverse edges if they don't already exist"""
        edge_set = set(self.edges)
        new_edges = []
        for i, j in self.edges:
            rev_edge = (j, i)
            if rev_edge not in edge_set:
                new_edges.append(rev_edge)
                edge_set.add(rev_edge)
        self.edges.extend(new_edges)

    def build_neighbor_maps(self):
        """Build O(1) lookup structures"""
        # Map vertices to their indices
        self.vertex_to_id = {tuple(v): i for i, v in enumerate(self.vertices)}

        # Build adjacency lists
        self.vertex_id_to_edges = {i: [] for i in range(len(self.vertices))}
        for i, j in self.edges:
            self.vertex_id_to_edges[i].append(j)

    def remove_duplicate_edges(self):
        """Remove any duplicate edges"""
        print("Removing duplicate edges...")
        self.edges = list(set(self.edges))

    def reduce_to_k_neighbors(self):
        """
        Post-processing step to reduce number of neighbors back to k_neighbors
        for vertices that have more connections than k_neighbors.
        Keeps the k closest neighbors based on configuration space distance.
        """
        vertices_array = np.array(self.vertices)
        modified_edges = []
        vertices_processed = 0

        # Calculate pairwise distances once
        diff = vertices_array[:, np.newaxis, :] - vertices_array[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)

        for vertex_idx in range(len(self.vertices)):
            neighbors = self.vertex_id_to_edges[vertex_idx]
            if len(neighbors) <= self.k_neighbors:
                # Keep all edges for this vertex
                modified_edges.extend((vertex_idx, n) for n in neighbors)
                continue

            # Get distances to all neighbors
            neighbor_distances = [(n, distances[vertex_idx][n]) for n in neighbors]
            # Sort by distance and keep k closest
            sorted_neighbors = sorted(neighbor_distances, key=lambda x: x[1])
            kept_neighbors = sorted_neighbors[:self.k_neighbors]

            # Add the kept edges
            modified_edges.extend((vertex_idx, n) for n, _ in kept_neighbors)
            vertices_processed += 1

        # Update the roadmap
        self.edges = modified_edges
        self.build_neighbor_maps()

        print(f"Processed {vertices_processed} vertices with more than {self.k_neighbors} neighbors")

    def add_shortcuts(self, max_attempts=100):
        """
        Try to add shortcut edges to the roadmap.
        For paths of length 2 (i->j->k), try to connect i directly to k.
        """
        print("Adding shortcuts...")
        new_edges = []
        attempts = 0

        for i in range(len(self.vertices)):
            if attempts >= max_attempts:
                break

            # Get neighbors of neighbors (2-hop neighbors)
            for j in self.vertex_id_to_edges[i]:
                for k in self.vertex_id_to_edges[j]:
                    if k == i or (i, k) in self.edges:
                        continue

                    attempts += 1
                    if self.check_edge_validity(self.vertices[i], self.vertices[k]):
                        new_edges.append((i, k))
                        new_edges.append((k, i))

        self.edges.extend(new_edges)
        print(f"Added {len(new_edges)} shortcuts")

    def check_connectivity(self):
        """Check and report connectivity statistics of the roadmap"""
        components = self._dfs_components()
        print(f"Roadmap has {len(components)} connected components")
        print(f"Largest component size: {max(len(c) for c in components)}")

    def get_neighbors(self, vertex_idx: int) -> List[int]:
        """Get indices of all neighbors for a vertex"""
        return self.vertex_id_to_edges[vertex_idx]

    def get_vertex_id(self, config: List[float]) -> int:
        """Get the ID of a configuration if it exists in the roadmap"""
        return self.vertex_to_id.get(tuple(config))

    def is_edge(self, i: int, j: int) -> bool:
        """Check if edge exists between vertices i and j"""
        return j in self.vertex_id_to_edges[i]

    def save_roadmap(self, filename: str):
        """Save roadmap to file"""
        data = {
            'vertices': self.vertices,
            'edges': self.edges,
            'robot_name': self.robot_name,
            'home_cfg_idx': self.home_cfg_idx,
            'params': {
                'n_samples': self.n_samples,
                'k_neighbors': self.k_neighbors,
                'max_edge_distance': self.max_edge_distance
            }
        }
        np.save(filename, data)

    @classmethod
    def load_roadmap(cls, motion_planner, filename: str, **kwargs):
        """Load roadmap from file and return new PRM instance

        Args:
            motion_planner: Instance of MotionPlanner class
            filename: Path to saved roadmap file

        Returns:
            PRM: New instance with loaded roadmap
        """
        data = np.load(filename, allow_pickle=True).item()

        # Create new PRM instance with saved parameters
        prm = cls(motion_planner,
                  robot_name=data['robot_name'],
                  n_samples=data['params']['n_samples'],
                  k_neighbors=data['params']['k_neighbors'],
                  max_edge_distance=data['params']['max_edge_distance'])

        prm.vertices = data['vertices']
        prm.edges = data['edges']
        prm.home_cfg_idx = data['home_cfg_idx']
        print(f"Loaded roadmap with {len(prm.vertices)} vertices and {len(prm.edges)} edges")

        prm.post_process()  # Build the neighbor maps and add reverse edges
        return prm

    def visualize_roadmap(self):
        """Visualize the roadmap in Klampt viewer"""
        # Visualize vertices
        for i, config in enumerate(self.vertices):
            name = f"vertex_{i}"
            self.mp.vis_config(self.robot_name, config, vis_name=name, rgba=(0, 0, 1, 0.3))

    def visualize_roadmap_ee_poses(self):
        """Visualize the roadmap in Klampt viewer showing end effector positions"""
        # Visualize vertices
        for i, config in enumerate(self.vertices):
            name = f"vertex_{i}"
            ee_pose = self.mp.get_forward_kinematics(self.robot_name, config)
            self.mp.show_point_vis(ee_pose[1], name=f"{i}")

    def visualize_connected_components_ee_poses(self):
        """
        Visualize the roadmap's connected components in different colors, showing end effector positions.
        Each connected component will be displayed in a distinct color using a preset color palette.
        """
        # Preset colors (RGB values)
        colors = [
            (1, 0, 0),  # Red
            (0, 1, 0),  # Green
            (0, 0, 1),  # Blue
            (1, 1, 0),  # Yellow
            (1, 0, 1),  # Magenta
            (0, 1, 1),  # Cyan
            (1, 0.5, 0),  # Orange
            (0.5, 0, 1),  # Purple
            (0, 0.5, 0),  # Dark Green
            (0.5, 0.5, 1),  # Light Blue
        ]

        # Get connected components
        components = self._dfs_components()

        # Visualize each component
        for component_idx, component in enumerate(components):
            color = (*colors[component_idx % len(colors)], 0.7)  # Cycle through colors if more components than colors

            for vertex_idx in component:
                config = self.vertices[vertex_idx]
                ee_pose = self.mp.get_forward_kinematics(self.robot_name, config)
                self.mp.show_point_vis(ee_pose[1], name=f"cc{component_idx}", rgba=color)

        print(f"Visualized {len(components)} connected components in different colors")

    def find_path(self, start_config: List[float], goal_config: List[float]) -> List[List[float]]:
        """
        Find a path between start and goal configurations using the roadmap.
        If start or goal configs are not in the roadmap, they will be added.
        """
        # Convert configs to tuples for lookup
        start_tuple = tuple(start_config)
        goal_tuple = tuple(goal_config)

        # Check if configs are already in roadmap
        start_id = self.vertex_to_id.get(start_tuple)
        goal_id = self.vertex_to_id.get(goal_tuple)

        # Add start config if not in roadmap
        if start_id is None:
            if not self.mp.is_config_feasible(self.robot_name, start_config):
                raise ValueError("Start configuration is invalid")
            if self.constraint_func is not None and not self.constraint_func(start_config, self.mp):
                raise ValueError("Start configuration violates constraint")
            start_id = self.add_vertex(start_config)

        # Add goal config if not in roadmap
        if goal_id is None:
            if not self.mp.is_config_feasible(self.robot_name, goal_config):
                raise ValueError("Goal configuration is invalid")
            if self.constraint_func is not None and not self.constraint_func(goal_config, self.mp):
                raise ValueError("Goal configuration violates constraint")
            goal_id = self.add_vertex(goal_config)

        # Use BFS to find shortest path
        visited = {start_id}
        queue = deque([(start_id, [start_id])])
        path_found = None

        while queue and not path_found:
            current_id, path = queue.popleft()

            if current_id == goal_id:
                path_found = path
                break

            for neighbor_id in self.vertex_id_to_edges[current_id]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        # Convert vertex IDs to configurations
        if path_found:
            return [self.vertices[i] for i in path_found]

        return []

    def find_path_by_vertices_id(self, start_vertex_id: int, goal_vertex_id: int):
        # TODO this is a little bit stupid, it should be the other way, find path should call this.
        start_config = self.vertices[start_vertex_id]
        goal_config = self.vertices[goal_vertex_id]

        return self.find_path(start_config, goal_config)

    def get_statistics(self) -> dict:
        """Get comprehensive statistics about the roadmap
        Returns:
            dict containing various statistics about the roadmap structure
        """
        stats = {}

        # Basic counts
        stats['num_vertices'] = len(self.vertices)
        stats['num_edges'] = len(self.edges)
        stats['average_degree'] = len(self.edges) / len(self.vertices)

        # Edge distribution
        degrees = [len(neighbors) for neighbors in self.vertex_id_to_edges.values()]
        stats['max_vertex_degree'] = max(degrees)
        stats['min_vertex_degree'] = min(degrees)
        stats['vertices_exceeding_k'] = sum(1 for d in degrees if d > self.k_neighbors)

        # Joint space coverage
        vertices_array = np.array(self.vertices)
        joint_min = np.min(vertices_array, axis=0)
        joint_max = np.max(vertices_array, axis=0)
        stats['joint_space_bounds'] = {
            'min': joint_min.tolist(),
            'max': joint_max.tolist()
        }
        stats['joint_space_volume'] = np.prod(joint_max - joint_min)

        # Connected components and average shortest paths
        components = self._dfs_components()
        stats['num_components'] = len(components)
        component_sizes = [len(c) for c in components]
        stats['component_size_distribution'] = {size: component_sizes.count(size)
                                                for size in set(component_sizes)}

        stats['largest_component_size'] = max(component_sizes)
        stats['smallest_component_size'] = min(component_sizes)
        stats['average_component_size'] = sum(component_sizes) / len(component_sizes)
        stats['isolated_vertices'] = sum(1 for d in degrees if d == 0)

        def bfs_shortest_paths(start, component_vertices):
            distances = {}
            queue = deque([(start, 0)])
            while queue:
                vertex, dist = queue.popleft()
                if vertex not in distances:
                    distances[vertex] = dist
                    for neighbor in self.vertex_id_to_edges[vertex]:
                        if neighbor in component_vertices and neighbor not in distances:
                            queue.append((neighbor, dist + 1))
            return distances

        total_paths = 0
        total_length = 0
        all_distances = []
        for component in components:
            component_set = set(component)
            for v in component:
                distances = bfs_shortest_paths(v, component_set)
                all_distances.extend(distances.values())
                total_length += sum(distances.values())
                total_paths += len(distances) - 1  # exclude path to self

        stats['average_shortest_path'] = total_length / total_paths if total_paths > 0 else float('inf')
        stats['shortest_path_distribution'] = {dist: all_distances.count(dist) for dist in set(all_distances)}

        # Build timing statistics if available
        if hasattr(self, 'build_times'):
            stats['build_times'] = self.build_times

        # Path and distance analysis
        vertices_array = np.array(self.vertices)
        diff = vertices_array[:, np.newaxis, :] - vertices_array[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        stats['max_edge_length'] = max(distances[i][j] for i, j in self.edges)
        stats['min_edge_length'] = min(distances[i][j] for i, j in self.edges)
        stats['average_edge_length'] = np.mean([distances[i][j] for i, j in self.edges])

        # Coverage analysis (using end effector positions)
        ee_positions = [self.mp.get_forward_kinematics(self.robot_name, v)[1] for v in self.vertices]
        ee_positions = np.array(ee_positions)
        ee_min = np.min(ee_positions, axis=0)
        ee_max = np.max(ee_positions, axis=0)
        stats['workspace_volume'] = np.prod(ee_max - ee_min)
        stats['workspace_bounds'] = {
            'min': ee_min.tolist(),
            'max': ee_max.tolist()
        }

        # Bidirectional edge verification
        num_unidirectional = sum(1 for i, j in self.edges if (j, i) not in self.edges)
        stats['unidirectional_edges'] = num_unidirectional

        return stats

    def print_statistics(self):
        """Print formatted statistics about the roadmap"""
        stats = self.get_statistics()
        print("\nRoadmap Statistics:")
        print(f"Number of vertices: {stats['num_vertices']}")
        print(f"Number of edges: {stats['num_edges']}")
        print(f"Average degree: {stats['average_degree']:.2f}")

        if 'build_times' in stats and stats['build_times'] is not None:
            print(f"\nBuild Times:")
            print(f"  Sampling time: {stats['build_times']['sampling']:.2f}s")
            print(f"  Connection time: {stats['build_times']['connection']:.2f}s")
            print(f"  Total time: {stats['build_times']['total']:.2f}s")

        print(f"\nEdge Distribution:")
        print(f"  Maximum vertex degree: {stats['max_vertex_degree']}")
        print(f"  Minimum vertex degree: {stats['min_vertex_degree']}")
        print(f"  Vertices exceeding k({self.k_neighbors}): {stats['vertices_exceeding_k']}")

        print(f"\nConnected Components:")
        print(f"  Number of components: {stats['num_components']}")
        print(f"  Largest component size: {stats['largest_component_size']}")
        print(f"  Smallest component size: {stats['smallest_component_size']}")
        print(f"  Average component size: {stats['average_component_size']:.2f}")
        print(f"  Isolated vertices: {stats['isolated_vertices']}")

        print("\nComponent Size Distribution:")
        for size, count in sorted(stats['component_size_distribution'].items()):
            print(f"  Size {size}: {count} components")

        print(f"\nPath Analysis:")
        print(f"  Maximum edge length: {stats['max_edge_length']:.3f}")
        print(f"  Minimum edge length: {stats['min_edge_length']:.3f}")
        print(f"  Average edge length: {stats['average_edge_length']:.3f}")
        print(f"  Average shortest path length (uniform cost): {stats['average_shortest_path']:.2f}")

        print("\nShortest Path Distribution:")
        for dist, count in sorted(stats['shortest_path_distribution'].items()):
            print(f"  Length {dist}: {count} paths")

        print(f"\nJoint Space Coverage:")
        print(f"  Volume: {stats['joint_space_volume']:.3f}")
        for i, (min_val, max_val) in enumerate(zip(stats['joint_space_bounds']['min'],
                                                   stats['joint_space_bounds']['max'])):
            print(f"  Joint {i}: [{min_val:.3f}, {max_val:.3f}]")

        print(f"\nWorkspace Coverage:")
        print(f"  Volume: {stats['workspace_volume']:.3f}")
        print(f"  Bounds min: {[f'{x:.3f}' for x in stats['workspace_bounds']['min']]}")
        print(f"  Bounds max: {[f'{x:.3f}' for x in stats['workspace_bounds']['max']]}")

        print(f"\nEdge Properties:")
        print(f"  Unidirectional edges: {stats['unidirectional_edges']}")


if __name__ == "__main__":
    planner = POManMotionPlanner()
    for i in range(1, 5):
        planner.add_mobile_obstacle_by_number(i)

    #
    prm = PRM(planner, "ur5e_1", n_samples=150, k_neighbors=7, max_edge_distance=5., eps=1e-1,
              # constraint_func=y_up_constraint
              )
    prm.build_roadmap()

    # prm.print_statistics()
    # Add some test configs
    # test_config = [0, -np.pi / 2, 0, -np.pi / 2, 0, 0]
    # idx = prm.add_vertex(test_config)
    # print(f"Added test config at index {idx}")
    # prm.print_statistics()

    planner.visualize()
    # prm.visualize_roadmap()
    # prm.visualize_connected_components_ee_poses()

    # add more vertices near the obstacle area
    xlims = (-1.5, -0.15)
    ylims = (-0.5, 0.5)
    zlims = (0, 0.8)
    prm.add_workspace_vertices(75, xlims, ylims, zlims)

    xlims = (-1.5, -0.35)
    ylims = (-0.4, 0.4)
    zlims = (0, 0.5)
    prm.add_workspace_vertices(25, xlims, ylims, zlims)

    prm.visualize_roadmap_ee_poses()
    prm.print_statistics()
    prm.save_roadmap("roadmap_ur5e_1.npy")
    pass
    # prm = PRM.load_roadmap(planner, "roadmap_ur5e_1.npy")
    # prm.print_statistics()
    #

