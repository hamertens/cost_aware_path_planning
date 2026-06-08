import numpy as np
import math
import random
from planners.base import get_path_cost, BasePlanner
from planners.deployment_time import get_rig_deployment_time

class RIGNode:
    def __init__(self, state, cost_from_root=0.0, info_from_root=0.0, parent=None, path_indices=None):
        self.state = state
        self.cost_from_root = cost_from_root
        self.info_from_root = info_from_root
        self.parent = parent
        # Store a set of all indices visited on the path to this node
        self.path_indices = path_indices if path_indices is not None else set()

class RIGPlanner(BasePlanner):
    def __init__(self, size, df, gp, output_dir, inf_criteria, seed=42, normalize_reward=True, optimized_noise_var=0.03, **kwargs):
        super().__init__(size, df, gp=gp, output_dir=output_dir, inf_criteria=inf_criteria, seed=seed, normalize_reward=normalize_reward, optimized_noise_var=optimized_noise_var, **kwargs)
        # Optuna passes these via kwargs
        self.step_limit  = int(kwargs.get('horizon', 3))
        self.samples     = int(kwargs.get('samples', 4500))
        self.gamma       = float(kwargs.get('gamma', 0.81))
        self.rng = random.Random(self.seed)

    def find_closest_unvisited(self, start_idx, global_history):
            """BFS to find the nearest grid cell not in history."""
            queue = [start_idx]
            visited_in_search = {start_idx}
            
            while queue:
                curr = queue.pop(0)
                if curr not in global_history:
                    return curr
                
                cx, cy = curr % self.size, curr // self.size
                # Check 4-connectivity neighbors
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        n_idx = ny * self.size + nx
                        if n_idx not in visited_in_search:
                            visited_in_search.add(n_idx)
                            queue.append(n_idx)
            
            return start_idx

    def select_next(self, sigma_map, sampled_indices):
        self.deployment_time = get_rig_deployment_time(
            len(sampled_indices), self.inf_criteria, self.step_limit, self.samples, self.size_cat
        )
        self.step_budget = self.compute_step_budget(len(sampled_indices))
        curr_row = self.df.iloc[sampled_indices[-1]]
        root_state = (curr_row['x'], curr_row['y'])
        global_history = set(sampled_indices)
        
        root_idx = int(round(root_state[1])) * self.size + int(round(root_state[0]))
        nodes = [RIGNode(root_state, path_indices={root_idx})]
        node_depths = {nodes[0]: 0} 
        
        best_leaf = nodes[0]
        max_discounted_info = -1.0
        absolute_max_sigma = np.max(sigma_map) 

        for _ in range(self.samples):
            # 1. Sample and Initial Steer (to get a candidate new_state)
            rand_state = (self.rng.uniform(0, self.size-1), self.rng.uniform(0, self.size-1))
            nearest_node = min(nodes, key=lambda n: math.dist(n.state, rand_state))
            
            # We steer from the nearest node just to find the target coordinates
            new_state, _ = self.steer_by_cost(nearest_node.state, rand_state)

            nx, ny = int(round(new_state[0])), int(round(new_state[1]))
            target_idx = ny * self.size + nx

            # Basic validity checks
            if target_idx in global_history:
                continue

            # 2. Radius Search for the Best Parent
            search_radius = self.step_budget * 1.5
            neighbors = [n for n in nodes if math.dist(n.state, new_state) <= search_radius]
            
            current_cell_sigma = self.get_cell_sigma(new_state, sigma_map)
            best_parent = None
            best_potential_total_info = -1.0
            saved_actual_cost = 0.0 # To store the specific cost of the winning edge

            for neighbor in neighbors:
                if target_idx in neighbor.path_indices:
                    continue
                    
                depth = node_depths[neighbor] + 1
                if depth > self.step_limit:
                    continue
                
                # --- THE FIX: Calculate specific cost for this specific edge ---
                # Using the helper from your base class
                edge_cost = get_path_cost(
                    neighbor.state[0], neighbor.state[1],
                    new_state[0], new_state[1],
                    self.slope_grid, self.elevation_grid, self.size,
                    len(sampled_indices), self.deployment_time
                )

                # Prevent division by zero or infinitesimally small moves
                if edge_cost <= 0: continue

                # Calculate reward based on THIS neighbor's edge cost
                step_reward = current_cell_sigma / edge_cost if self.normalize_reward else current_cell_sigma
                discounted_reward = step_reward * (self.gamma ** (depth - 1))
                total_info_from_this_parent = neighbor.info_from_root + discounted_reward
                
                # --- Branch and Bound ---
                remaining_steps = self.step_limit - depth
                max_future = absolute_max_sigma * (self.gamma**depth) * (1 - self.gamma**remaining_steps) / (1 - self.gamma) if self.gamma < 1.0 else absolute_max_sigma * remaining_steps
                
                if (total_info_from_this_parent + max_future) < max_discounted_info:
                    continue
                    
                if total_info_from_this_parent > best_potential_total_info:
                    best_potential_total_info = total_info_from_this_parent
                    best_parent = neighbor
                    saved_actual_cost = edge_cost

            # 3. Add Node
            if best_parent:
                new_path = best_parent.path_indices.copy()
                new_path.add(target_idx)
                
                new_node = RIGNode(
                    new_state, 
                    best_parent.cost_from_root + saved_actual_cost, 
                    best_potential_total_info, 
                    best_parent,
                    path_indices=new_path
                )
                
                nodes.append(new_node)
                node_depths[new_node] = node_depths[best_parent] + 1

                if best_potential_total_info > max_discounted_info:
                    max_discounted_info = best_potential_total_info
                    best_leaf = new_node

        # 4. Traceback (same as before)
        if best_leaf == nodes[0]:
            return self.find_closest_unvisited(root_idx, global_history)

        curr = best_leaf
        while curr.parent and curr.parent.parent:
            curr = curr.parent
        
        return int(round(curr.state[1])) * self.size + int(round(curr.state[0]))