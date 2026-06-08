import torch
import numpy as np
import math
from types import SimpleNamespace
from planners.base import get_path_cost, get_sampling_cost, BasePlanner
from planners.deployment_time import get_ergodic_deployment_time
from ergodic_search import erg_planner
from ergodic_search.dynamics import DynModule


# 1. Continuous Dynamics for Point Mass
class SimplePoint(DynModule):
    def __init__(self, start_pose, traj_steps, dt=0.1):
        super().__init__(start_pose, traj_steps)
        self.state_dim = 2
        self.control_dim = 2
        # Initialize with small noise to help gradient flow
        self.controls = torch.nn.Parameter(torch.randn((traj_steps, self.control_dim)) * 0.1)
        self.dt = dt

    def forward(self, controls=None):
        if controls is None:
            controls = self.controls
        pos = torch.tensor(self.start_pose[:2], dtype=torch.float32).clone().detach()
        traj_xy = torch.cumsum(controls * self.dt, dim=0) + pos
        traj_xy = torch.clamp(traj_xy, 0.0, 1.0)
        # Append dummy theta for ErgPlanner compatibility
        theta = torch.zeros((traj_xy.shape[0], 1), device=traj_xy.device)
        return torch.cat([traj_xy, theta], dim=1)

# 2. Differentiable Cost Wrapper (Voxel Cost Approximation)
class ErgodicVoxelCostLoss(torch.nn.Module):
    def __init__(self, original_loss, cost_grid, cost_wt=0.1):
        super().__init__()
        object.__setattr__(self, 'original_loss', original_loss)
        object.__setattr__(self, 'dyn_model', original_loss.dyn_model)
        # Pre-calculated cost grid: (1 + k*slope)
        grid_tensor = torch.tensor(cost_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        object.__setattr__(self, 'cost_grid', grid_tensor)
        object.__setattr__(self, 'cost_wt', cost_wt)

    def forward(self, print_flag=False):
        base_loss = self.original_loss(print_flag=print_flag)
        traj = self.dyn_model.forward()
        # Map [0, size] -> [-1, 1] for grid_sample
        size = self.cost_grid.shape[-1]
        grid_coords = (traj[:, :2].unsqueeze(0).unsqueeze(2) / (size - 1) * 2) - 1
        sampled_costs = torch.nn.functional.grid_sample(
            self.cost_grid, grid_coords, align_corners=True, mode='bilinear'
        )
        return base_loss + self.cost_wt * torch.mean(sampled_costs)

    def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
    def __getattr__(self, name): return getattr(self.original_loss, name)

# 3. The Ergodic Planner Class
class ErgodicPlanner(BasePlanner):
    def __init__(self, size, df, gp, output_dir, inf_criteria, seed=42, normalize_reward=True, optimized_noise_var=None, **kwargs):
        super().__init__(size, df, gp, output_dir, inf_criteria, seed=seed, normalize_reward=normalize_reward, optimized_noise_var=optimized_noise_var, **kwargs)
        self.deployment_time = 8.0  # updated dynamically in select_next
        self._iters      = int(kwargs.get('iters', 700))
        self._traj_steps = int(kwargs.get('traj_steps', 100))

        # --- FIX: Bypass ErgArgs() argparse conflict ---
        # Instead of calling erg_planner.ErgArgs(), we manually build the namespace
        self.args = SimpleNamespace(
            learn_rate=[float(kwargs.get('learn_rate', 0.001))],
            num_pixels=size,
            gpu=False,
            traj_steps=int(kwargs.get('traj_steps', 100)),
            iters=int(kwargs.get('iters', 700)),
            epsilon=float(kwargs.get('epsilon', 0.005)),
            start_pose=[0.0, 0.0, 0.0],
            end_pose=[0.0, 0.0, 0.0],
            num_freqs=int(kwargs.get('num_freqs', 10)),
            erg_wt=1.0,
            transl_vel_wt=0.0,
            ang_vel_wt=0.0,
            bound_wt=1000.0,
            end_pose_wt=0.0,
            debug=False,
            outpath=None,
            replan_type='full',
            seed=seed
        )

        # Pre-calculate spatial cost grid (uphill formula used as conservative estimate;
        # direction of travel is not known at grid-build time)
        slope_data = df['slope_deg'].values.reshape((size, size))
        self.cost_grid = (0.0052184063-0.0004212098) * slope_data + 0.0545374711
        self.cost_wt   = float(kwargs.get('cost_wt', 0.05))
        self.time_diff = float(kwargs.get('dt', 0.5))

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
        curr_row = self.df.iloc[sampled_indices[-1]]

        root_state = (curr_row['x'], curr_row['y'])
        root_idx = int(round(root_state[1])) * self.size + int(round(root_state[0]))
        
        # 1. SCALE DOWN: Convert framework coords (e.g., 22) to [0, 1]
        start_x_norm = curr_row['x'] / (self.size - 1)
        start_y_norm = curr_row['y'] / (self.size - 1)
        root_state_norm = [start_x_norm, start_y_norm, 0.0]
        
        # Update args to use normalized start
        self.args.start_pose = root_state_norm
        
        # 2. Initialize Dynamics with Normalized Start
        # This ensures the optimizer is working in the [0, 1] range
        dyn = SimplePoint(root_state_norm, self.args.traj_steps, self.time_diff)
        
        planner = erg_planner.ErgPlanner(self.args, sigma_map, 
                                        init_controls=dyn.controls.data, 
                                        dyn_model=dyn)
        
        # Inject Cost (Ensure grid_coords mapping is also correct)
        planner.loss = ErgodicVoxelCostLoss(planner.loss, self.cost_grid, cost_wt=self.cost_wt)

        # 3. Optimize
        _, traj_tensor, _ = planner.compute_traj()
        traj = traj_tensor.detach().numpy()

        # 4. Dynamic deployment time + cost budget
        n = len(sampled_indices)
        self.deployment_time = get_ergodic_deployment_time(
            n, self.inf_criteria, self._iters, self._traj_steps, self.size_cat
        )
        budget = self.compute_step_budget(n)

        # Walk along the trajectory until cumulative traversal cost exhausts the budget
        chosen_idx = len(traj) - 1
        cumulative_cost = 0.0
        for step_i in range(1, len(traj)):
            prev = traj[step_i - 1]
            curr = traj[step_i]
            # Distance in normalised [0,1] space → convert to grid units
            dist_norm = float(np.hypot(curr[0] - prev[0], curr[1] - prev[1]))
            dist_grid = dist_norm * (self.size - 1)
            # Cost multiplier at this point (bilinear from cost_grid)
            gx = int(np.clip(round(curr[0] * (self.size - 1)), 0, self.size - 1))
            gy = int(np.clip(round(curr[1] * (self.size - 1)), 0, self.size - 1))
            cost_mult = float(self.cost_grid[gy, gx])
            cumulative_cost += cost_mult * dist_grid
            if cumulative_cost >= budget:
                chosen_idx = step_i
                break

        next_state_norm = traj[chosen_idx]

        # Map [0, 1] back to [0, size-1]
        nx = int(round(np.clip(next_state_norm[0] * (self.size - 1), 0, self.size - 1)))
        ny = int(round(np.clip(next_state_norm[1] * (self.size - 1), 0, self.size - 1)))

        if nx == curr_row['x'] and ny == curr_row['y']:
            # If the optimizer returns the same cell, find the closest unvisited one
            return self.find_closest_unvisited(root_idx, sampled_indices)
        
        return ny * self.size + nx