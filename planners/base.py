import numpy as np
from scipy.linalg import solve_triangular
from planners.deployment_time import get_myopic_deployment_time

def get_training_time(n):
    if n <= 100:
        return 0.0007 * n + 0.0627
    else:
        return 3.674824e-09 * n**3 + 1.139843e-05 * n**2 + -1.519382e-03 * n + 0.191615

def get_sampling_cost(n_train, deployment_time):
    return 0.005 * (deployment_time + get_training_time(n_train))

def get_travel_cost(x0, y0, x1, y1, slope_grid, elevation_grid, size):
    """
    Amanatides-Woo Fast Voxel Traversal for 2D.
    Returns pure traversal cost — no sampling cost component.
    Going uphill: cell_cost_per_dist = 0.0052184063 * slope + 0.0545374711
    Going downhill: cell_cost_per_dist = 0.0004212098 * slope + 0.0545374711
    """
    if x0 == x1 and y0 == y1:
        return 0.0

    dx, dy = x1 - x0, y1 - y0
    total_dist = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / total_dist, dy / total_dist

    step_x = 1 if dx > 0 else -1 if dx < 0 else 0
    step_y = 1 if dy > 0 else -1 if dy < 0 else 0

    t_delta_x = abs(1.0 / ux) if ux != 0 else float('inf')
    t_delta_y = abs(1.0 / uy) if uy != 0 else float('inf')

    ix, iy = int(round(x0)), int(round(y0))

    if ux > 0: t_max_x = (ix + 0.5 - x0) / ux
    elif ux < 0: t_max_x = (ix - 0.5 - x0) / ux
    else: t_max_x = float('inf')

    if uy > 0: t_max_y = (iy + 0.5 - y0) / uy
    elif uy < 0: t_max_y = (iy - 0.5 - y0) / uy
    else: t_max_y = float('inf')

    t_curr = 0.0
    total_cost = 0.0

    while t_curr < 1.0:
        t_next = min(t_max_x, t_max_y, 1.0)
        segment_len = (t_next - t_curr) * total_dist

        c_ix, c_iy = max(0, min(size-1, ix)), max(0, min(size-1, iy))
        slope = slope_grid[c_iy, c_ix]

        z_start = elevation_grid[c_iy, c_ix]
        next_ix, next_iy = ix, iy
        if t_max_x < t_max_y: next_ix += step_x
        else: next_iy += step_y
        n_ix, n_iy = max(0, min(size-1, next_ix)), max(0, min(size-1, next_iy))
        z_end = elevation_grid[n_iy, n_ix]

        if z_end >= z_start:
            cell_cost_per_dist = 0.0052184063 * slope + 0.0545374711  # uphill
        else:
            cell_cost_per_dist = 0.0004212098 * slope + 0.0545374711  # downhill

        total_cost += segment_len * cell_cost_per_dist

        if t_next >= 1.0: break

        t_curr = t_next
        if t_max_x < t_max_y:
            t_max_x += t_delta_x
            ix += step_x
        else:
            t_max_y += t_delta_y
            iy += step_y

    return total_cost

def get_path_cost(x0, y0, x1, y1, slope_grid, elevation_grid, size, n_train, deployment_time):
    """Total cost = travel cost + sampling cost."""
    return get_travel_cost(x0, y0, x1, y1, slope_grid, elevation_grid, size) + get_sampling_cost(n_train, deployment_time)

def get_action_cost(current_idx, all_indices, df, size, n_train, deployment_time, training_time=None):
    """
    Compute path cost(s) from current_idx to each index in all_indices.

    When training_time is provided (measured wall-clock value), it is used directly
    in the sampling cost formula.  When omitted, the estimated get_training_time(n_train)
    is used instead (for planner-internal calls where no measurement is available).
    """
    slope_grid = np.zeros((size, size))
    slope_grid[df['y'].values.astype(int), df['x'].values.astype(int)] = df['slope_deg'].values
    elevation_grid = df['z'].values.reshape((size, size))

    curr_row = df.iloc[current_idx]
    x0, y0 = curr_row['x'], curr_row['y']

    targets_x = df['x'].values[all_indices]
    targets_y = df['y'].values[all_indices]

    if training_time is None:
        training_time = get_training_time(n_train)
    sampling_cost = 0.005 * (deployment_time + training_time)

    costs = np.array([get_travel_cost(x0, y0, tx, ty, slope_grid, elevation_grid, size) + sampling_cost
                      for tx, ty in zip(targets_x, targets_y)])
    return costs

class BasePlanner:
    def __init__(self, size, df, gp, output_dir, inf_criteria, seed=42, normalize_reward=True, optimized_noise_var=0.03, deployment_time=1.0, **kwargs):
        self.size, self.df = size, df
        self.gp = gp
        self.inf_criteria = inf_criteria
        self.is_dynamic = self.inf_criteria.startswith('dyn_')
        self.slope_grid = df['slope_deg'].values.reshape((size, size))
        self.elevation_grid = df['z'].values.reshape((size, size))
        self.deployment_time = deployment_time
        self.normalize_reward = normalize_reward
        self.seed = seed
        self.optimized_noise_var = optimized_noise_var
        self.size_cat = kwargs.get('size_cat', 'small')
        self.X_ref = df[['x', 'y']].values # For Cohn's ALC reference points
        self.w_ref = None
        self.budget_a = float(kwargs.get('budget_a', 1.0))
        self.budget_b = float(kwargs.get('budget_b', 5.0))

    def select_next(self, sigma, sampled_indices): raise NotImplementedError()

    def compute_step_budget(self, n_samples):
        """Dynamic step budget: budget_a + budget_b * get_sampling_cost(n_samples, deployment_time)."""
        return self.budget_a + self.budget_b * get_sampling_cost(n_samples, self.deployment_time)

    def get_cell_sigma(self, state, sigma_map):
        ix = int(round(state[0]))
        iy = int(round(state[1]))
        ix = max(0, min(self.size - 1, ix))
        iy = max(0, min(self.size - 1, iy))
        return sigma_map[iy * self.size + ix]
    
    def update_gp_and_w(self, L_old, X_old, y_old, alpha_old, w_old, x_new, y_new):
        """
        Updates the GP state AND the Cohn's reference matrix w.
        """
        # 1. Standard GP Rank-1 update (as before)
        x_new = np.atleast_2d(x_new)
        k_new = self.gp.kernel_(X_old, x_new)
        k_self = self.gp.kernel_(x_new, x_new) + self.optimized_noise_var
        
        v = solve_triangular(L_old, k_new, lower=True)
        gamma = np.sqrt(np.maximum(1e-10, k_self - np.dot(v.T, v)))
        
        new_L = np.block([[L_old, np.zeros((L_old.shape[0], 1))], [v.T, gamma]])
        new_X = np.vstack([X_old, x_new])
        new_y = np.append(y_old, y_new)
        
        temp = solve_triangular(new_L, new_y, lower=True)
        new_alpha = solve_triangular(new_L.T, temp, lower=False)

        # 2. THE RECURSIVE W UPDATE (The magic part)
        # New row for w: (K(x_new, X_ref) - v.T @ w_old) / gamma
        if "cohns" in self.inf_criteria:
            K_new_ref = self.gp.kernel_(x_new, self.X_ref)
            w_new_row = (K_new_ref - np.dot(v.T, w_old)) / gamma
            new_w = np.vstack([w_old, w_new_row])
        else:
            new_w = None

        return new_L, new_X, new_y, new_alpha, new_w
    
    def prepare_step(self):
        """Initial w calculation for the ROOT of the MCTS tree."""
        if "cohns" in self.inf_criteria and self.is_dynamic:
            # O(N_train * N_grid) - Only done ONCE per real-world step
            K_train_ref = self.gp.kernel_(self.gp.X_train_, self.X_ref)
            self.w_ref_root = solve_triangular(self.gp.L_, K_train_ref, lower=True)
        else:
            self.w_ref_root = None
    
    def get_inf_score(self, x_cand, L, X, w_current=None, inf_map=None):
        """
        Unified entry point for rewards.
        x_cand: the hypothetical point (continuous)
        L, X: Current node's GP state
        w_current: only needed for cohns
        """

        if not self.is_dynamic:
            ix = int(round(x_cand[0]))
            iy = int(round(x_cand[1]))
            # Bounds safety
            ix = max(0, min(self.size - 1, ix))
            iy = max(0, min(self.size - 1, iy))
            return inf_map[[iy * self.size + ix]]
    
        x_cand = np.atleast_2d(x_cand)
        k_cand = self.gp.kernel_(X, x_cand)
        
        # 1. Standard Predictive Variance math (v = L^-1 * k)
        v = solve_triangular(L, k_cand, lower=True)
        var_cand = self.gp.kernel_(x_cand, x_cand) - np.sum(v**2, axis=0)
        std_cand = np.sqrt(np.maximum(1e-10, var_cand))

        # 2. Return based on criteria
        if "cohns" in self.inf_criteria:
            # Standard Cohn's ALC logic
            K_cand_ref = self.gp.kernel_(x_cand, self.X_ref)
            cov_post = K_cand_ref - np.dot(v.T, w_current)
            alc_score = np.sum(np.square(cov_post)) / (var_cand + 1e-9)
            return alc_score # or alc_score[0]
        
        # Default for 'variance' and 'dyn_variance'
        return std_cand[0]
    
    def get_alc_score(self, x_cand, L, X, w_current):
        """Calculates ALC using the node's specific w_current."""
        x_cand = np.atleast_2d(x_cand)
        k_cand = self.gp.kernel_(X, x_cand)
        
        # Predictive Variance
        v = solve_triangular(L, k_cand, lower=True)
        var_cand = self.gp.kernel_(x_cand, x_cand) - np.sum(v**2, axis=0)
        
        # Cohn's numerator: Sum of squared posterior covariances
        # cov_post = K(x_cand, X_ref) - v.T @ w_current
        K_cand_ref = self.gp.kernel_(x_cand, self.X_ref)
        cov_post = K_cand_ref - np.dot(v.T, w_current)
        
        score = np.sum(np.square(cov_post)) / (var_cand + 1e-9)
        return score[0]

    def steer_by_cost(self, s_from, s_to):
        """
        Traverses from s_from towards s_to. 
        Stops if:
        1. The target (s_to) is reached.
        2. The accumulated_travel_cost hits self.step_budget.
        3. The ray leaves the map boundaries.
        """
        x0, y0 = s_from
        x1, y1 = s_to
        travel_budget = self.step_budget 
        
        n_train = len(self.gp.X_train_)
        sampling_cost = get_sampling_cost(n_train, self.deployment_time)

        if travel_budget <= 0:
            return s_from, sampling_cost

        dx, dy = x1 - x0, y1 - y0
        total_dist = np.sqrt(dx**2 + dy**2)
        if total_dist == 0:
            return s_from, sampling_cost
            
        ux, uy = dx / total_dist, dy / total_dist
        
        step_x = 1 if dx > 0 else -1 if dx < 0 else 0
        step_y = 1 if dy > 0 else -1 if dy < 0 else 0
        
        t_delta_x = abs(1.0 / ux) if ux != 0 else float('inf')
        t_delta_y = abs(1.0 / uy) if uy != 0 else float('inf')
        
        ix, iy = int(round(x0)), int(round(y0))

        # Initialize t_max based on direction
        if ux > 0: t_max_x = (ix + 0.5 - x0) / ux
        elif ux < 0: t_max_x = (ix - 0.5 - x0) / ux
        else: t_max_x = float('inf')
            
        if uy > 0: t_max_y = (iy + 0.5 - y0) / uy
        elif uy < 0: t_max_y = (iy - 0.5 - y0) / uy
        else: t_max_y = float('inf')

        t_curr = 0.0
        accumulated_travel_cost = 0.0
        curr_x, curr_y = x0, y0

        while t_curr < 1.0:
            # 1. Boundary Check: Stop if we leave the grid
            if ix < 0 or ix >= self.size or iy < 0 or iy >= self.size:
                break

            t_next = min(t_max_x, t_max_y, 1.0)
            segment_len = (t_next - t_curr) * total_dist
            
            # Determine current cell slope
            slope = self.slope_grid[iy, ix]

            # Determine elevation change to select appropriate slope cost function
            z_start = self.elevation_grid[iy, ix]
            next_ix, next_iy = (ix + step_x, iy) if t_max_x < t_max_y else (ix, iy + step_y)
            n_ix, n_iy = max(0, min(self.size-1, next_ix)), max(0, min(self.size-1, next_iy))
            z_end = self.elevation_grid[n_iy, n_ix]

            if z_end >= z_start:
                cell_unit_cost = 0.0052184063 * slope + 0.0545374711  # uphill
            else:
                cell_unit_cost = 0.0004212098 * slope + 0.0545374711  # downhill

            segment_cost = segment_len * cell_unit_cost
            
            # 2. Budget Check: Stop if this segment exceeds remaining energy
            if accumulated_travel_cost + segment_cost > travel_budget:
                remaining_travel = travel_budget - accumulated_travel_cost
                affordable_dist = remaining_travel / cell_unit_cost
                
                final_x = curr_x + ux * affordable_dist
                final_y = curr_y + uy * affordable_dist
                
                # Return capped position and full budget cost (+ sampling)
                return (final_x, final_y), self.step_budget + sampling_cost
            
            accumulated_travel_cost += segment_cost
            
            # Update current coordinates to the exit point of the cell
            curr_x = x0 + ux * t_next * total_dist
            curr_y = y0 + uy * t_next * total_dist
            
            if t_next >= 1.0: 
                break
                
            t_curr = t_next
            if t_max_x < t_max_y:
                t_max_x += t_delta_x
                ix += step_x
            else:
                t_max_y += t_delta_y
                iy += step_y
        
        # Reached the target or boundary: Return current pos and actual cost (+ sampling)
        return (curr_x, curr_y), accumulated_travel_cost + sampling_cost

class MyopicPlanner(BasePlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_next(self, sigma, sampled_indices):
        sigma_m = np.copy(sigma)
        sigma_m[sampled_indices] = -1
        if self.normalize_reward:
            self.deployment_time = get_myopic_deployment_time(len(sampled_indices), self.inf_criteria, self.size_cat)
            costs = get_action_cost(sampled_indices[-1], np.arange(len(sigma)), self.df, self.size, len(sampled_indices), self.deployment_time)
            return np.argmax(sigma_m / (costs + 1e-6))
        return np.argmax(sigma_m)

class ConstrainedPlanner(BasePlanner):
    def __init__(self, size, df, gp, inf_criteria, output_dir, seed=42, normalize_reward=False, optimized_noise_var=None, **kwargs):
        super().__init__(size, df, gp, output_dir, inf_criteria, seed, normalize_reward, optimized_noise_var, **kwargs)

    def select_next(self, sigma, sampled_indices):
        self.deployment_time = get_myopic_deployment_time(len(sampled_indices), self.inf_criteria, self.size_cat)
        budget = self.compute_step_budget(len(sampled_indices))
        costs = get_action_cost(sampled_indices[-1], np.arange(len(sigma)), self.df, self.size, len(sampled_indices), self.deployment_time)
        sigma_m = np.copy(sigma)
        sigma_m[sampled_indices] = -1
        valid = (costs <= budget) & (sigma_m != -1)
        if not np.any(valid):
            return np.argmin(np.where(sigma_m == -1, np.inf, costs))
        if self.normalize_reward:
            return np.argmax(np.where(valid, sigma_m / (costs + 1e-6), -np.inf))
        return np.argmax(np.where(valid, sigma_m, -1))


STRATEGIES = {"myopic": MyopicPlanner, "constrained": ConstrainedPlanner}