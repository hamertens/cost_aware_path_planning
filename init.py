import os, sys, pandas as pd, numpy as np, time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error
from planners.base import get_action_cost
from scipy.stats import qmc

def get_initial_indices(planner_name, size, df, seed):
    """Returns the first two indices based on the planner strategy."""
    name = planner_name.lower()

    np.random.seed(int(seed))
    
    if name == "snake":
        # Snake starts at (0,0) then (1,0)
        idx1 = 0
        idx2 = 1
        return idx1, idx2
    
    elif name == "spiral":
        # Spiral starts at center, then moves right (middle, middle) -> (middle+1, middle)
        mid = size // 2
        idx1 = int(mid * size + mid)
        idx2 = int(mid * size + (mid + 1))
        return idx1, idx2

    elif name == "sobol":
        # Generate the first two unique points of the Sobol sequence
        sampler = qmc.Sobol(d=2, scramble=True, seed=int(seed))
        raw_samples = sampler.random(n=8) # Generate 8 to be safe
        scaled = np.round(raw_samples * (size - 1)).astype(int)
        
        unique_coords = []
        for x, y in scaled:
            if (x, y) not in unique_coords:
                unique_coords.append((x, y))
            if len(unique_coords) == 2:
                break
        
        idx1 = int(unique_coords[0][1] * size + unique_coords[0][0])
        idx2 = int(unique_coords[1][1] * size + unique_coords[1][0])
        return idx1, idx2

    else:
        # Default behavior for adaptive planners (Variance, KTBO, etc.)
        mid = size // 2
        idx1 = df[(df['x'] == mid) & (df['y'] == mid)].index[0]
        
        # Define the 8 neighbors (Moore neighborhood)
        neighbors = [
            (mid-1, mid-1), (mid, mid-1), (mid+1, mid-1),
            (mid-1, mid),                (mid+1, mid),
            (mid-1, mid+1), (mid, mid+1), (mid+1, mid+1)
        ]
        
        # Pick one neighbor randomly using the seed
        chosen_neighbor = neighbors[np.random.choice(len(neighbors))]
        nx, ny = chosen_neighbor
        
        # Map back to index
        idx2 = df[(df['x'] == nx) & (df['y'] == ny)].index[0]

        return idx1, idx2

def init(env_id, planner_name, size_cat, seed, norm_reward, inf_criteria):

    raw_inf_val = inf_criteria.lower() if isinstance(inf_criteria, str) else str(inf_criteria)
    # Robust boolean conversion
    if raw_inf_val in ['variance', 'var', 'v']:
        inf_criteria = 'variance'
    elif raw_inf_val in ['cohns', 'cohns', 'c']:
        inf_criteria = 'cohns'
    elif raw_inf_val in ['dyn_variance', 'dynamic_variance', 'dyn_var', 'dv']:
        inf_criteria = 'dyn_variance'
    elif raw_inf_val in ['dyn_cohns', 'dynamic_cohns', 'dyn_c', 'dc']:
        inf_criteria = 'dyn_cohns'
    else:
        raise ValueError(f"Invalid inf_criteria value: {inf_criteria}. Expected one of 'variance', 'cohns', 'dyn_variance', 'dyn_cohns' (or their variants).")
    env_str = f"{int(env_id):02d}"
    base_path = os.path.join(os.path.dirname(__file__), "environments", size_cat)
    
    # 1. Load the three sets: Exploration (Ground Truth), Test, and Validation
    df = pd.read_csv(f"{base_path}/exploration/env_expl_{env_str}.csv")
    df_test = pd.read_csv(f"{base_path}/test/env_test_{env_str}.csv")
    df_val = pd.read_csv(f"{base_path}/validation/env_val_{env_str}.csv")
    
    base_out_dir = f"output_data/output_data_env{env_id}_{planner_name}_{size_cat}"
    inf_folder = f"inf_{inf_criteria}"
    out_dir = os.path.join(base_out_dir, f"seed_{seed}", inf_folder)
    if str(norm_reward).lower() != "none":
        out_dir = os.path.join(out_dir, f"norm_{str(norm_reward).lower()}") # Subdirectory for normalization setting
    os.makedirs(out_dir, exist_ok=True)
    
    # Seed the global random state for initial noise
    np.random.seed(int(seed))

    # Grid setup
    size = int(df['x'].max() + 1)
    
    idx1, idx2 = get_initial_indices(planner_name, size, df, seed)

    X, y = df[['x', 'y']].values, df['water_ice'].values

    y_1 = y[[idx1]]
    sigma_1 = 0.03 * (y_1.max() - y_1.min())
    y_noisy_1 = y_1 + np.random.normal(0.0, sigma_1, size=y_1.shape)

    white_kernel1 = WhiteKernel(
                noise_level=sigma_1**2, 
                noise_level_bounds=(1e-6, max(1e-4, y_noisy_1.var()))
            )
    
    # Setup GP and Data
    kernel1 = C(1, (1e-3, 1e3)) * RBF(3, (0.01, 100.0)) + white_kernel1
    gp1 = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer=3, random_state=int(seed))

    y_2 = y[[idx1, idx2]]
    sigma_2 = 0.03 * (y_2.max() - y_2.min())
    y_noisy_2 = y_2 + np.random.normal(0.0, sigma_2, size=y_2.shape)

    white_kernel2 = WhiteKernel(
                noise_level=sigma_2**2, 
                noise_level_bounds=(1e-6, max(1e-4, y_noisy_2.var()))
            )   

    kernel2 = C(1, (1e-3, 1e3)) * RBF(3, (0.01, 100.0)) + white_kernel2
    gp2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=3, random_state=int(seed))

    
    X_test, y_test = df_test[['x', 'y']].values, df_test['water_ice'].values
    X_val, y_val = df_val[['x', 'y']].values, df_val['water_ice'].values
    
    # Calculate normalization ranges (denominator for NRMSE)
    range_test = np.max(y_test) - np.min(y_test)
    range_val = np.max(y_val) - np.min(y_val)
    
    # Ensure no division by zero if map is perfectly flat
    range_test = range_test if range_test > 0 else 1.0
    range_val = range_val if range_val > 0 else 1.0

    # Helper for random retroactive error
    def get_random_retro_error(trained_gp, exclude_indices):
        all_indices = np.arange(len(df))
        available = np.setdiff1d(all_indices, exclude_indices)
        random_idx = np.random.choice(available)
        y_pred = trained_gp.predict(X[[random_idx]])[0]
        return np.abs(y[random_idx] - y_pred)

    def calculate_nrmse(trained_gp, X_eval, y_eval, data_range):
        rmse = np.sqrt(mean_squared_error(y_eval, trained_gp.predict(X_eval)))
        return rmse / data_range

    # Log point 1: First sample at center
    start_t1 = time.time()
    gp1.fit(X[[idx1]], y_noisy_1)
    t_time1 = time.time() - start_t1
    retro1 = get_random_retro_error(gp1, [idx1])
    nrmse_test1 = calculate_nrmse(gp1, X_test, y_test, range_test)
    nrmse_val1 = calculate_nrmse(gp1, X_val, y_val, range_val)
    
    # Log point 2: Small step to immediate neighbor
    cost_to_2 = get_action_cost(idx1, [idx2], df, size, 1, deployment_time=3.0)[0]
    start_t2 = time.time()
    gp2.fit(X[[idx1, idx2]], y_noisy_2)
    t_time2 = time.time() - start_t2
    retro2 = get_random_retro_error(gp2, [idx1, idx2])
    nrmse_test2 = calculate_nrmse(gp2, X_test, y_test, range_test)
    nrmse_val2 = calculate_nrmse(gp2, X_val, y_val, range_val)
    
    # Create log with NRMSE and Validation headers
    log = pd.DataFrame([
        {
            'sample_index': idx1, 
            'nrmse_test': nrmse_test1, 
            'nrmse_val': nrmse_val1,
            'action_cost': 0.0, 
            'training_time': t_time1, 
            'deployment_time': 0.0, 
            'retroactive_error': retro1
        },
        {
            'sample_index': idx2, 
            'nrmse_test': nrmse_test2, 
            'nrmse_val': nrmse_val2,
            'action_cost': cost_to_2, 
            'training_time': t_time2, 
            'deployment_time': 0.0, 
            'retroactive_error': retro2
        }
    ])
    
    log.to_csv(os.path.join(out_dir, "log.csv"), index=False)

if __name__ == "__main__":
    # Expecting: python3 init.py <env_id> <planner_name> <size_cat> <seed> <normalize_reward> <inf_criteria>
    init(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])