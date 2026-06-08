import numpy as np
import math
from scipy.linalg import solve_triangular
from planners.base import get_travel_cost, BasePlanner
import json
import os

class IPPMPEPlanner(BasePlanner):
    def __init__(self, size, df, gp, output_dir, inf_criteria, seed=42, normalize_reward=True, optimized_noise_var=0.03, **kwargs):
        super().__init__(size, df, gp=gp, output_dir=output_dir, inf_criteria=inf_criteria, seed=seed, normalize_reward=normalize_reward, optimized_noise_var=optimized_noise_var, **kwargs)
        self.deployment_time = 3.0
        self.gp = gp
        self.max_landmarks = int(kwargs.get('samples', 10))
        self.noise_var = 0.1
        self.path_file = os.path.join(output_dir, "planned_trajectory.json") # Persistent file

        if not self.is_dynamic:
            raise ValueError("IIGPlanner requires is_dynamic=True as it relies on GP updates and a dynamic information criteria.")


    def select_next(self, inf_map, sampled_indices):
        # 1. Load the path if it exists
        path = []
        if os.path.exists(self.path_file):
            with open(self.path_file, 'r') as f:
                try:
                    path = json.load(f)
                except json.JSONDecodeError:
                    path = []

        # 2. If empty, generate a fresh batch of self.max_landmarks
        if not path:
            # Plan a new trajectory using the fixed batch size
            path = self._calculate_new_plan(inf_map, sampled_indices)
            
            # Save it immediately
            with open(self.path_file, 'w') as f:
                json.dump(path, f)

        # 3. Pop the next step
        next_step = path.pop(0)
        
        # 4. Save the remaining path back to the file
        with open(self.path_file, 'w') as f:
            json.dump(path, f)
            
        return next_step

    def _calculate_new_plan(self, inf_map, sampled_indices):
        self.prepare_step()
        
        """
        Phase 1: MPE Sampler to find informative landmarks.
        Phase 2: Greedy TSP to connect them by action cost.
        """
        # 1. Initialize GP state based on all history
        curr_L = self.gp.L_
        curr_X = self.gp.X_train_
        curr_y = self.gp.y_train_
        curr_alpha = self.gp.alpha_
        # Use Cohn's reference matrix if it was prepared in prepare_step
        curr_w = getattr(self, 'w_ref_root', None)
        
        # Ensure we don't pick points the robot already visited
        global_history = set(sampled_indices)
        landmarks = []
        
        # --- PHASE 1: LANDMARK SELECTION (MPE Sampler) ---
        for _ in range(self.max_landmarks):
            max_score = -1.0
            best_idx = -1
            
            # Find point with highest uncertainty (the "blind spot")
            # Note: We iterate over indices in the dataframe
            for idx in range(len(self.df)):
                if idx in global_history or idx in landmarks:
                    continue
                candidate_row = self.df.iloc[idx]
                
                # Get predictive uncertainty at this candidate index
                score = self.get_inf_score((candidate_row['x'], candidate_row['y']), curr_L, curr_X, curr_w, inf_map=inf_map)
                
                if score > max_score:
                    max_score = score
                    best_idx = idx
            
            # If we found a good landmark, 'observe' it to update knowledge
            if best_idx != -1:
                landmarks.append(best_idx)
                row = self.df.iloc[best_idx]
                pos = (row['x'], row['y'])
                k_star = self.gp.kernel_(curr_X, np.atleast_2d(pos))
                pred_mean = (k_star.T @ curr_alpha)[0]
                curr_L, curr_X, curr_y, curr_alpha, curr_w = self.update_gp_and_w(
                    curr_L, curr_X, curr_y, curr_alpha, curr_w, pos, pred_mean
                )
            else:
                break # No more informative points found

        # --- PHASE 2: GREEDY TSP (Action Cost Based) ---
        trajectory = []
        remaining = landmarks.copy()
        current_idx = sampled_indices[-1] # Start from where we are now
        
        while remaining:
            best_next = -1
            min_cost = float('inf')
            
            curr_row = self.df.iloc[current_idx]
            
            for l_idx in remaining:
                l_row = self.df.iloc[l_idx]
                
                cost = get_travel_cost(
                    curr_row['x'], curr_row['y'],
                    l_row['x'], l_row['y'],
                    self.slope_grid, self.elevation_grid, self.size
                )
                
                if cost < min_cost:
                    min_cost = cost
                    best_next = l_idx
            
            if best_next != -1:
                trajectory.append(best_next)
                remaining.remove(best_next)
                current_idx = best_next
            else:
                # If we get stuck and can't reach any more, stop planning
                break
        
        return trajectory

    def find_closest_unvisited(self, start_idx, global_history):
        """Simple BFS fallback."""
        queue = [start_idx]
        visited = {start_idx}
        while queue:
            curr = queue.pop(0)
            if curr not in global_history:
                return curr
            cx, cy = curr % self.size, curr // self.size
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    n_idx = ny * self.size + nx
                    if n_idx not in visited:
                        visited.add(n_idx)
                        queue.append(n_idx)
        return start_idx