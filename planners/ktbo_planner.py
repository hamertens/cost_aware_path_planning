import numpy as np
from skopt import Optimizer
from skopt.space import Real
from planners.base import get_travel_cost, BasePlanner
from planners.deployment_time import get_ktbo_deployment_time

class KTBOPlanner(BasePlanner):
    def __init__(self, size, df, gp, output_dir, inf_criteria, seed=42, normalize_reward=True, optimized_noise_var=0.03, **kwargs):
        # Using named arguments to ensure values go to the right place
        super().__init__(size, df, gp=gp, output_dir=output_dir, inf_criteria=inf_criteria, seed=seed, normalize_reward=normalize_reward, optimized_noise_var=optimized_noise_var, **kwargs)

        # --- System Hyperparameters (Optimized by Optuna) ---
        # We use .get() with your original defaults
        self.simulations   = int(kwargs.get('simulations', 42))
        self.ls_y          = float(kwargs.get('ls_y', 0.77547))
        self.n_samples     = int(kwargs.get('n_samples', 3))
        self.num_anchors   = int(kwargs.get('num_anchors', 4))
        self.horizon_scale = float(kwargs.get('horizon_scale', 1.0179))
        self.gamma         = float(kwargs.get('gamma', 0.778))
        

        np.random.seed(self.seed)
        
        # Static Regularizer
        self.delta = 1e-6
    def get_state_by_budget(self, path_clipped, budget):
        """
        Traverses the generated path and returns the coordinate (x, y) 
        where the budget is exhausted.
        """
        total_cost = 0.0
        # Start position
        prev_p = path_clipped[0]
        
        for i in range(1, len(path_clipped)):
            curr_p = path_clipped[i]
            
            # Calculate cost for this specific segment
            step_cost = get_travel_cost(
                prev_p[0], prev_p[1], curr_p[0], curr_p[1],
                self.slope_grid, self.elevation_grid, self.size
            )
            
            if total_cost + step_cost > budget:
                # The budget ends somewhere on this segment (prev_p -> curr_p)
                remaining_budget = budget - total_cost
                return self._find_point_on_segment(prev_p, curr_p, remaining_budget)
            
            total_cost += step_cost
            prev_p = curr_p
            
        # If we finish the path and still have budget, return the last point
        return path_clipped[-1]

    def _find_point_on_segment(self, start, end, budget_limit):
        """
        Binary search to find the point between start and end 
        that matches the budget_limit cost.
        """
        low = 0.0
        high = 1.0
        best_p = start
        
        # 5 iterations is usually enough for sub-pixel precision
        for _ in range(5):
            mid = (low + high) / 2
            test_p = start + mid * (end - start)
            
            cost = get_travel_cost(
                start[0], start[1], test_p[0], test_p[1],
                self.slope_grid, self.elevation_grid, self.size
            )
            
            if cost <= budget_limit:
                best_p = test_p
                low = mid
            else:
                high = mid
                
        return best_p

    def _get_rbf_kernel(self, a, b, lengthscale):
        """Matrix-based RBF kernel calculation."""
        sq_dist = np.sum(a**2, axis=1).reshape(-1, 1) + np.sum(b**2, axis=1) - 2 * np.dot(a, b.T)
        return np.exp(-0.5 * sq_dist / lengthscale**2)

    def generate_trajectory(self, angles, seg_dist, start_pos):
        """
        Implements Kernel Bayes' Rule to generate N waypoints 
        along a smooth RKHS curve.
        """
        # 1. Create Anchor Points in space (X) based on relative angles
        anchors_x = np.zeros((self.num_anchors + 1, 2))
        anchors_x[0] = start_pos
        curr_angle = 0
        for i in range(self.num_anchors):
            curr_angle += angles[i]
            anchors_x[i+1, 0] = anchors_x[i, 0] + seg_dist * np.cos(curr_angle)
            anchors_x[i+1, 1] = anchors_x[i, 1] + seg_dist * np.sin(curr_angle)

        # 2. Define Anchor Times (Y) and Query Times (y_q)
        Y = np.linspace(0, 1, self.num_anchors + 1).reshape(-1, 1)
        y_q = np.linspace(0, 1, self.n_samples + 1).reshape(-1, 1)

        # 3. Kernel Interpolation Math (The "Paper" way)
        KyY = self._get_rbf_kernel(Y, Y, self.ls_y)
        Kqy = self._get_rbf_kernel(y_q, Y, self.ls_y)
        
        # Solve for weights: w = (KyY + delta*I)^-1 * Kqy
        weights = np.linalg.solve(KyY + self.delta * np.eye(len(Y)), Kqy.T).T
        
        # Path is the weighted sum of anchor positions
        path_raw = np.dot(weights, anchors_x)
        
        # Clipping for map boundaries
        path_clipped = np.clip(path_raw, 0, self.size - 1)
        
        return path_clipped, path_raw

    def calculate_reward(self, path_clipped, path_raw, sigma_map, sampled_indices):
        total_reward = 0.0

        curr_L, curr_X, curr_y, curr_alpha, curr_w = None, None, None, None, None
        if self.is_dynamic:
            curr_L = self.gp.L_
            curr_X = self.gp.X_train_
            curr_y = self.gp.y_train_
            curr_alpha = self.gp.alpha_
            curr_w = getattr(self, 'w_ref_root', None)
        
        # --- 1. Boundary Penalties (Unchanged) ---
        out_of_bounds_mask = np.any((path_raw < 0) | (path_raw > self.size - 1), axis=1)
        oob_indices = np.where(out_of_bounds_mask)[0]
        total_points = len(path_raw)
        
        boundary_penalty = 0.0
        if (len(oob_indices) / total_points) > 0.10:
            for idx in oob_indices:
                temporal_weight = (total_points - idx) / total_points
                boundary_penalty -= (100 / total_points) * temporal_weight

        # --- 2. Information Gain with Redundancy Check ---
        global_history = set(sampled_indices)
        local_trajectory_history = set() # Track pixels used in THIS specific curve
        
        # Start the local history with the current robot position so we don't reward staying still
        start_ix = int(round(path_clipped[0, 0]))
        start_iy = int(round(path_clipped[0, 1]))
        local_trajectory_history.add(start_iy * self.size + start_ix)

        for i in range(1, len(path_clipped)):
            p_prev, p_curr = path_clipped[i-1], path_clipped[i]

            ix, iy = int(round(p_curr[0])), int(round(p_curr[1]))
            idx = iy * self.size + ix
            
            # Straight-line cost
            step_cost = get_travel_cost(p_prev[0], p_prev[1], p_curr[0], p_curr[1],
                                      self.slope_grid, self.elevation_grid, self.size)

            # --- INFORMATION GAIN ---
            if self.is_dynamic:
                # Dynamic: Calculate score based on CURRENT hallucinated belief
                score = float(self.get_inf_score((p_curr[0], p_curr[1]), curr_L, curr_X, curr_w, inf_map=sigma_map))
                
                # Hallucinate observation to update belief for the NEXT point in the curve
                k_star = self.gp.kernel_(curr_X, np.atleast_2d(p_curr))
                pred_mean = (k_star.T @ curr_alpha)[0]
                curr_L, curr_X, curr_y, curr_alpha, curr_w = self.update_gp_and_w(
                    curr_L, curr_X, curr_y, curr_alpha, curr_w, p_curr, pred_mean
                )
            else:
                # Static: Use the fixed map and simple redundancy check
                score = 0.0 if idx in global_history else sigma_map[idx]
                        
            # Penalize staying still
            if i == 1 and idx == sampled_indices[-1]:
                total_reward -= 50
                continue

            segment_reward = score / max(0.1, step_cost) if self.normalize_reward else score
            total_reward += (self.gamma ** (i-1)) * segment_reward
                
        return float(total_reward + boundary_penalty)

    def select_next(self, sigma_map, sampled_indices):
        self.deployment_time = get_ktbo_deployment_time(
            len(sampled_indices), self.inf_criteria, self.simulations, self.size_cat
        )
        self.step_budget = self.compute_step_budget(len(sampled_indices))
        self.prepare_step()
        curr_row = self.df.iloc[sampled_indices[-1]]
        start_pos = np.array([curr_row['x'], curr_row['y']])

        # 1. Define Search Space
        dist_max = (self.size * self.horizon_scale) / self.num_anchors
        # Angles [-pi, pi] for each anchor, plus Segment Distance [1.0, max]
        space = [Real(-np.pi, np.pi) for _ in range(self.num_anchors)] + \
                [Real(1.0, max(1.1, dist_max))]

        # 2. Initialize Skopt Optimizer
        # acq_func can be 'EI', 'LCB', etc. 
        opt = Optimizer(
            dimensions=space,
            base_estimator="GP",
            n_initial_points=10,
            acq_func="EI", 
            random_state=self.seed
        )

        best_reward = -np.inf
        best_candidate = None

        # 3. Optimization Loop
        for _ in range(self.simulations):
            # "Ask" for a new point to evaluate
            candidate = opt.ask() 
            
            angles = np.array(candidate[:self.num_anchors])
            seg_dist = candidate[-1]

            # Generate trajectory and calculate reward
            p_clipped, p_raw = self.generate_trajectory(angles, seg_dist, start_pos)
            reward = self.calculate_reward(p_clipped, p_raw, sigma_map, sampled_indices)

            # "Tell" the optimizer the result (Negative because skopt minimizes)
            opt.tell(candidate, -reward)

            if reward > best_reward:
                best_reward = reward
                best_candidate = candidate

        # 4. Reconstruct the best path
        best_angles = np.array(best_candidate[:self.num_anchors])
        best_seg_dist = best_candidate[-1]
        
        path_f, _ = self.generate_trajectory(best_angles, best_seg_dist, start_pos)
        
        # Return first waypoint index
        target_pos = self.get_state_by_budget(path_f, self.step_budget)
        return int(round(target_pos[1])) * self.size + int(round(target_pos[0]))