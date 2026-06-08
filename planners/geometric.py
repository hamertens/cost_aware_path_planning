import numpy as np
from scipy.stats import qmc
from planners.base import BasePlanner

def next_power_of_2(n):
    if n <= 0:
        return 1
    # Check if n is already a power of 2
    if (n & (n - 1) == 0):
        return n
    return 1 << n.bit_length()

class SnakePlanner(BasePlanner):
    def __init__(self, size, df, gp, output_dir, inf_criteria, seed=42, normalize_reward=True, optimized_noise_var=None, **kwargs):
        super().__init__(size, df, gp, output_dir, inf_criteria, seed=seed, normalize_reward=normalize_reward, optimized_noise_var=optimized_noise_var, **kwargs)
        self.deployment_time = 0.0
        self.path = []
        for y in range(size):
            # Even rows: Left -> Right | Odd rows: Right -> Left
            x_range = range(size) if y % 2 == 0 else range(size - 1, -1, -1)
            for x in x_range:
                self.path.append(int(y * size + x))

    def select_next(self, sigma, sampled_indices):
        # Return the index matching the current step count
        # if sampled_indices has 2 items, we want the 3rd item (index 2)
        return self.path[len(sampled_indices)]

class SpiralPlanner(BasePlanner):
    def __init__(self, size, df, gp, output_dir, inf_criteria, seed=42, normalize_reward=True, optimized_noise_var=None, **kwargs):
        super().__init__(size, df, gp, output_dir, inf_criteria, seed=seed, normalize_reward=normalize_reward, optimized_noise_var=optimized_noise_var, **kwargs)
        self.deployment_time = 0.0
        self.path = self._generate_spiral(size)

    def _generate_spiral(self, size):
        coords = []
        x, y = size // 2, size // 2
        coords.append(int(y * size + x))
        
        # Directions: Right, Up, Left, Down
        dx, dy = [1, 0, -1, 0], [0, 1, 0, -1]
        step_size, dir_idx = 1, 0
        
        while len(coords) < size * size:
            for _ in range(2): # Repeat step size twice before increasing
                for _ in range(step_size):
                    x, y = x + dx[dir_idx], y + dy[dir_idx]
                    if 0 <= x < size and 0 <= y < size:
                        idx = int(y * size + x)
                        if idx not in coords:
                            coords.append(idx)
                    if len(coords) == size * size: break
                dir_idx = (dir_idx + 1) % 4
                if len(coords) == size * size: break
            step_size += 1
        return coords

    def select_next(self, sigma, sampled_indices):
        return self.path[len(sampled_indices)]

class SobolPlanner(BasePlanner):
    def __init__(self, size, df, gp, output_dir, inf_criteria, seed=42, normalize_reward=True, optimized_noise_var=None, **kwargs):
        super().__init__(size, df, gp, output_dir, inf_criteria, seed=seed, normalize_reward=normalize_reward, optimized_noise_var=optimized_noise_var, **kwargs)
        self.deployment_time = 0.0
        sampler = qmc.Sobol(d=2, scramble=True, seed=self.seed)
        # Generate plenty of points
        raw = sampler.random(n=next_power_of_2(size*size))
        scaled = np.round(raw * (size - 1)).astype(int)
        
        self.path = []
        seen = set()
        for x, y in scaled:
            idx = int(y * size + x)
            if idx not in seen:
                self.path.append(idx)
                seen.add(idx)
        # Note: We don't break at len=2 anymore, we keep the whole sequence

    def select_next(self, sigma, sampled_indices):
        # Convert to set for O(1) lookups
        history = set(sampled_indices)
        
        # Look through our pre-generated Sobol sequence
        for candidate in self.path:
            # If this Sobol point hasn't been sampled yet, take it!
            if candidate not in history:
                return candidate
        
        # Emergency fallback: if for some reason the whole map is sampled
        return sampled_indices[-1]