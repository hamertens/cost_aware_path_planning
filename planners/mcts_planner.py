import numpy as np
import math
import random
from planners.base import get_path_cost, get_sampling_cost, BasePlanner
from planners.deployment_time import get_mcts_deployment_time
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.linalg import solve_triangular, cholesky

# Ignore convergence warnings globally
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class MCTSNode:
    def __init__(self, state, parent=None, action_taken=None, reward=0.0, 
                 gp_L=None, gp_X=None, gp_y=None, gp_alpha=None, gp_w = None):
        self.state = state  # (x, y) continuous
        self.parent = parent
        self.action_taken = action_taken
        self.reward = reward # immediate reward
        self.children = {}  
        self.visits = 0
        self.total_reward = 0.0
        
        # Rank-1 GP State
        self.gp_L = gp_L      # Local Cholesky factor
        self.gp_X = gp_X      # Local training points
        self.gp_y = gp_y
        self.gp_alpha = gp_alpha # Precomputed (K^-1 * y) for mean prediction
        self.gp_w = gp_w # Precomputed Cohn's reference matrix


class MCTSPlanner(BasePlanner):
    def __init__(self, size, df, gp, output_dir, inf_criteria, seed=42, normalize_reward=True, optimized_noise_var=0.03, **kwargs):
        # 1. Pass core metadata up to BasePlanner
        super().__init__(size, df, gp=gp, output_dir=output_dir, inf_criteria=inf_criteria, seed=seed, normalize_reward=normalize_reward, optimized_noise_var=optimized_noise_var, **kwargs)

        # 2. Extract hyperparameters from kwargs with your original defaults
        self.simulations = int(kwargs.get('simulations', 1200))
        self.horizon     = int(kwargs.get('horizon', 4))
        self.gamma       = float(kwargs.get('gamma', 0.9))
        self.c_param     = float(kwargs.get('c_param', 0.8))
        self.k_a         = float(kwargs.get('k_a', 4.0))
        self.alpha_a     = float(kwargs.get('alpha_a', 0.2))
        self.rng = random.Random(self.seed)
        self.gp = gp
        if self.gp is not None:
            # L_ is the lower triangular Cholesky factor
            # alpha_ is the solved weight vector
            # X_train_ is the data
            self.L = self.gp.L_
            self.X = self.gp.X_train_
            self.alpha = self.gp.alpha_
        self.kernel_func = self.gp.kernel_
        self.max_steer_angle = 7*math.pi / 6

    
    def sample_by_budget(self, current_state, parent_state, history_set):
        # Try different directions until we find one that actually lets us move

        base_heading = None
        if parent_state is not None:
            base_heading = math.atan2(current_state[1] - parent_state[1], 
                                      current_state[0] - parent_state[0])
            
        for _ in range(50): 
            if base_heading is not None:
                # Random angle between -pi/2 and +pi/2 relative to base_heading
                lower_bound = base_heading - self.max_steer_angle
                upper_bound = base_heading + self.max_steer_angle
                theta = self.rng.uniform(lower_bound, upper_bound)
            else:
                # First step: any direction is fine
                theta = self.rng.random() * 2 * math.pi
            
            # Project far enough to definitely hit a boundary or exhaust budget
            large_dist = self.size * 1.5 
            far_target = (
                current_state[0] + large_dist * math.cos(theta),
                current_state[1] + large_dist * math.sin(theta)
            )
            
            # Find the furthest reachable point in this direction
            limit_state, _ = self.steer_by_cost(current_state, far_target)
            
            # Calculate how far that limit is from where we are
            dx = limit_state[0] - current_state[0]
            dy = limit_state[1] - current_state[1]
            max_reachable_dist = math.sqrt(dx**2 + dy**2)
            
            # If we can move at least 1.0 unit, this direction is valid!
            if max_reachable_dist >= 1.0:
                actual_dist = self.rng.uniform(1.0, max_reachable_dist)
                
                new_x = current_state[0] + actual_dist * math.cos(theta)
                new_y = current_state[1] + actual_dist * math.sin(theta)
                
                # Ensure the new point is strictly within map bounds
                new_x = max(0, min(self.size - 1, new_x))
                new_y = max(0, min(self.size - 1, new_y))
                
                new_state = (new_x, new_y)
                
                # Check if rounding to the nearest grid cell results in the SAME cell
                if int(round(new_x)) == int(round(current_state[0])) and \
                int(round(new_y)) == int(round(current_state[1])):
                    continue # Resample if we haven't actually moved to a new cell

                # DUPLICATE CHECK ---
                target_idx = int(round(new_y)) * self.size + int(round(new_x))
                if target_idx in history_set:
                    continue # Skip if already sampled in real life or earlier in this tree path
                    
                actual_step_cost = get_path_cost(
                    current_state[0], current_state[1],
                    new_x, new_y,
                    self.slope_grid, self.elevation_grid, self.size,
                    len(self.gp.X_train_), self.deployment_time
                )
                return new_state, actual_step_cost

        # EMERGENCY FALLBACK:
        # If we've tried 50 times and can't find a move > 1.0 (e.g. stuck in a corner),
        # just return the current state with a nominal cost to prevent infinite loops.
        return current_state, get_sampling_cost(len(self.gp.X_train_), self.deployment_time)
    
    
    def select_next(self, inf_map, sampled_indices):
        self.deployment_time = get_mcts_deployment_time(
            len(sampled_indices), self.inf_criteria, self.horizon, self.simulations, self.size_cat
        )
        self.step_budget = self.compute_step_budget(len(sampled_indices))
        # 1. Initialize the Root
        self.prepare_step()
        curr_row = self.df.iloc[sampled_indices[-1]]
        root_state = (curr_row['x'], curr_row['y'])
        root = MCTSNode(
            root_state, 
            gp_L=self.gp.L_, 
            gp_X=self.gp.X_train_, 
            gp_alpha=self.gp.alpha_,
            gp_y=self.gp.y_train_,
            gp_w=getattr(self, 'w_ref_root', None) # Cohn's w vector if applicable
        )

        root_idx = int(round(root_state[1])) * self.size + int(round(root_state[0]))

        # Real-world history
        global_history = set(sampled_indices)

        for _ in range(self.simulations):
            node = root
            depth = 0

            # Track what we've "sampled" in this specific simulation path
            current_path_set = global_history.copy()
            
            # --- SELECTION & EXPANSION ---
            while depth < self.horizon:
                # Progressive Widening Condition: 
                # Use max(1, ...) to ensure we don't have math errors on visit 0
                if len(node.children) <= self.k_a * (node.visits ** self.alpha_a):
                    # EXPAND: Create a new branch
                    parent_state = node.parent.state if node.parent is not None else None
                    next_state, cost = self.sample_by_budget(node.state, parent_state, current_path_set)

                    # If expansion failed to find a new point, stop
                    if next_state == node.state:
                        break

                    inf_score = self.get_inf_score(next_state, node.gp_L, node.gp_X, 
                                               node.gp_w, inf_map=inf_map)

                    

                    immediate_reward = inf_score / cost if self.normalize_reward else inf_score

                    if self.is_dynamic:
                        # Update GP for the child node
                        k_star = self.gp.kernel_(node.gp_X, np.atleast_2d(next_state))
                        pred_mean = (k_star.T @ node.gp_alpha)[0]
                    
                        new_L, new_X, new_y, new_alpha, new_w = self.update_gp_and_w(
                            node.gp_L, node.gp_X, node.gp_y, node.gp_alpha, node.gp_w, 
                            next_state, pred_mean
                        )
                    else:
                        # Static: Child inherits parent's GP state directly
                        new_L, new_X, new_y, new_alpha, new_w = (
                            node.gp_L, node.gp_X, node.gp_y, node.gp_alpha, node.gp_w
                        )

                    new_node = MCTSNode(
                        next_state, parent=node, action_taken=cost, reward=immediate_reward,
                        gp_L=new_L, gp_X=new_X, gp_y=new_y, gp_alpha=new_alpha, gp_w=new_w
                    )

                                                  
                    # Use the unique state/cost as the key for children
                    node.children[next_state] = new_node
                    node = new_node
                    depth += 1
                    break # After expanding, move to Rollout
                else:
                    # SELECT: Move deeper into the existing tree using UCT
                    best = self.best_child(node)
                    if best is None: # Safety check
                        break
                    node = best
                    # Update the local path memory as we descend the tree
                    target_idx = int(round(node.state[1])) * self.size + int(round(node.state[0]))
                    current_path_set.add(target_idx)
                    depth += 1

            # --- ROLLOUT ---
            total_rollout_reward = self.rollout(node, depth, current_path_set, inf_map)
            
            # --- BACKPROPAGATION ---
            self.backpropagate(node, total_rollout_reward)

        # --- FINAL SELECTION ---
        # After all simulations, pick the branch from the ROOT that was visited most
        if not root.children:
            # Fallback if no simulations succeeded (shouldn't happen with k_a > 0)
            return self.find_closest_unvisited(root_idx, global_history)

        best_child_node = max(root.children.values(), key=lambda x: x.visits)
        final_state = best_child_node.state
        
        # Convert continuous (x, y) back to grid index
        target_ix = int(round(final_state[0]))
        target_iy = int(round(final_state[1]))
        target_ix = max(0, min(self.size - 1, target_ix))
        target_iy = max(0, min(self.size - 1, target_iy))

        target_idx = target_iy * self.size + target_ix

        # 4. SAFETY CHECK: If the 'best' child is just the current cell, trigger BFS
        if target_idx == root_idx:
            return self.find_closest_unvisited(root_idx, global_history)
        
        return target_idx

    def best_child(self, node):
        best_val = -float('inf')
        best_node = None
        for child in node.children.values():
            # Standard UCT
            uct = (child.total_reward / child.visits) + \
                  self.c_param * math.sqrt(math.log(max(1, node.visits)) / child.visits)
            if uct > best_val:
                best_val = uct
                best_node = child
        return best_node

    def rollout(self, node, depth, current_path_set, inf_map):
        curr_state = node.state
        prev_state = node.parent.state if node.parent is not None else None
        local_path = current_path_set.copy()
        rollout_reward = 0
        
        L, X, y, alpha, w = node.gp_L, node.gp_X, node.gp_y, node.gp_alpha, node.gp_w

        # We loop from the current depth up to the horizon
        for d in range(self.horizon - depth):
            next_state, cost = self.sample_by_budget(curr_state, prev_state, local_path)

            # If rollout gets stuck, end the reward accumulation
            if next_state == curr_state:
                break

            inf_score = self.get_inf_score(next_state, L, X, w, inf_map=inf_map)
            rollout_reward += (self.gamma ** d) * (inf_score / cost if self.normalize_reward else inf_score)

            if self.is_dynamic:
                k_star = self.gp.kernel_(X, np.atleast_2d(next_state))
                pred_mean = (k_star.T @ alpha)[0]
                L, X, y, alpha, w = self.update_gp_and_w(L, X, y, alpha, w, next_state, pred_mean)        

            target_idx = int(round(next_state[1])) * self.size + int(round(next_state[0]))
            local_path.add(target_idx)
            
            prev_state, curr_state = curr_state, next_state
            
        return rollout_reward

    def backpropagate(self, node, rollout_reward):
        # 1. Start with the value of the 'future' relative to this node
        # Since rollout started at Depth + 1, it's 1 step away from 'node'.
        G = self.gamma * rollout_reward 

        curr = node
        while curr is not None:
            curr.visits += 1
            
            # 2. The 'Return' for this node is its own immediate reward 
            # plus the discounted future we just brought up.
            current_return = curr.reward + G
            curr.total_reward += current_return

            # 3. Update G for the parent:
            # The parent is one step closer to the root, so this node's 
            # entire return must be discounted by gamma.
            G = self.gamma * current_return
            
            curr = curr.parent

    def find_closest_unvisited(self, start_idx, global_history):
        """BFS to find the nearest grid cell not in history."""
        queue = [start_idx]
        visited_in_search = {start_idx}
        while queue:
            curr = queue.pop(0)
            if curr not in global_history:
                return curr
            cx, cy = curr % self.size, curr // self.size
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    n_idx = ny * self.size + nx
                    if n_idx not in visited_in_search:
                        visited_in_search.add(n_idx)
                        queue.append(n_idx)
        return start_idx