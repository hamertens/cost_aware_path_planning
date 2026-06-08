import os, sys, pandas as pd, numpy as np, time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error
from planners.base import MyopicPlanner, ConstrainedPlanner, get_action_cost
from planners.mcts_planner import MCTSPlanner
from planners.rig_planner import RIGPlanner
from planners.iig_planner import IIGPlanner
from planners.ktbo_planner import KTBOPlanner
from planners.ergodic import ErgodicPlanner
from planners.ipp_mpe_planner import IPPMPEPlanner
from planners.geometric import SnakePlanner, SpiralPlanner, SobolPlanner

# Set your target NRMSE here (e.g., 0.05 for 5% error)
STOPPING_THRESHOLD = 0.03

STRATEGIES = {
    'myopic': MyopicPlanner,
    'constrained': ConstrainedPlanner,
    'mcts': MCTSPlanner,
    'rig' : RIGPlanner,
    'iig' : IIGPlanner,
    'ktbo' : KTBOPlanner,
    'ipp_mpe': IPPMPEPlanner,
    'ergodic': ErgodicPlanner,
    'snake': SnakePlanner,
    'spiral': SpiralPlanner,
    'sobol': SobolPlanner
}

from scipy.linalg import solve_triangular

def calculate_alc_scores(gp, X):
    """Calculates Cohn's ALC for all points in X using X as the reference set."""
    L = gp.L_
    # K(X_train, X_grid)
    K_train_X = gp.kernel_(gp.X_train_, X)
    # Whitening: L^-1 * K_train_X
    v = solve_triangular(L, K_train_X, lower=True)
    
    # Prior Covariance K(X, X)
    K_prior = gp.kernel_(X, X)
    # Posterior Covariance: K_prior - v.T @ v
    # This matrix contains cov(x_i, x_j | data) for all pairs in the grid
    cov_post = K_prior - np.dot(v.T, v)
    
    # Denominator: Predictive Variance
    _, std = gp.predict(X, return_std=True)
    var = std**2
    
    # ALC = Sum of squared posterior covariances / predictive variance
    alc_scores = np.sum(np.square(cov_post), axis=1) / (var + 1e-9)
    return alc_scores

def step(env_id, planner_name, size_cat, seed, norm_reward, inf_criteria, **extra_hps):

    raw_inf_val = inf_criteria.lower() if isinstance(inf_criteria, str) else str(inf_criteria)
    raw_norm_val = norm_reward.lower() if isinstance(norm_reward, str) else str(norm_reward)
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
    # Robust boolean conversion
    do_normalize = raw_norm_val in ['true', '1', 't', 'yes']
    env_str = f"{int(env_id):02d}"
    base_path = os.path.join(os.path.dirname(__file__), "environments", size_cat)
    
    # Load the sets
    df = pd.read_csv(f"{base_path}/exploration/env_expl_{env_str}.csv")
    df_test = pd.read_csv(f"{base_path}/test/env_test_{env_str}.csv")
    df_val = pd.read_csv(f"{base_path}/validation/env_val_{env_str}.csv")
    
    base_out_dir = f"output_data/output_data_env{env_id}_{planner_name}_{size_cat}"

    inf_folder = f"inf_{inf_criteria}"
    out_dir = os.path.join(base_out_dir, f"seed_{seed}", inf_folder) # Subdirectory for seed, directional, and information criteria settings
    if str(norm_reward).lower() != "none":
        out_dir = os.path.join(out_dir, f"norm_{str(norm_reward).lower()}")
    log_path = os.path.join(out_dir, "log.csv")
    log_df = pd.read_csv(log_path)
    
    X, y = df[['x', 'y']].values, df['water_ice'].values
    X_test, y_test = df_test[['x', 'y']].values, df_test['water_ice'].values
    X_val, y_val = df_val[['x', 'y']].values, df_val['water_ice'].values

    np.random.seed(int(seed))
    
    # Normalization ranges (max - min)
    range_test = np.max(y_test) - np.min(y_test)
    range_val = np.max(y_val) - np.min(y_val)
    range_test = range_test if range_test > 0 else 1.0
    range_val = range_val if range_val > 0 else 1.0
    
    sampled = log_df['sample_index'].tolist()
    size = int(df['x'].max() + 1)
    
    sigma = 0.03 * (y.max() - y.min())
    y_noisy = y[sampled] + np.random.normal(0.0, sigma, size=y[sampled].shape)

    white_kernel = WhiteKernel(
                noise_level=sigma**2, 
                noise_level_bounds=(1e-6, max(1e-4, y_noisy.var()))
            )
    
    kernel = C(1, (1e-3, 1e3)) * RBF(3, (0.01, 100.0)) + white_kernel

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=int(seed))
    
    # 1. Training Time Tracking
    start_train = time.time()
    gp.fit(X[sampled], y_noisy)
    training_time = time.time() - start_train
    
    # Validation NRMSE
    rmse_val = np.sqrt(mean_squared_error(y_val, gp.predict(X_val)))
    n_val = rmse_val / range_val

    # Test NRMSE
    rmse_test = np.sqrt(mean_squared_error(y_test, gp.predict(X_test)))
    n_test = rmse_test / range_test
    
    # 2. Deployment Time Tracking
    start_deploy = time.time()

    optimized_noise_var = gp.kernel_.get_params()['k2__noise_level']
    
    # Initialize the planner first
    planner_key = planner_name.lower()
    planner = STRATEGIES[planner_key](
        size,
        df,
        gp=gp,
        output_dir=out_dir,
        inf_criteria=inf_criteria,
        seed=int(seed),
        normalize_reward=do_normalize,
        noise_var=optimized_noise_var,
        size_cat=size_cat,
        **extra_hps
    )
    
    # Only calculate inf_map if the planner is NOT geometric
    # These 3 planners don't use the uncertainty values to choose the next point
    if planner_key in ['snake', 'spiral', 'sobol']:
        inf_map = None 
        next_idx = planner.select_next(inf_map, sampled)
    else:
        # Dynamic criteria recompute information inside the planner from the updated GP;
        # the static map is unused and skipped to save computation.
        if inf_criteria == 'cohns':
            inf_map = calculate_alc_scores(gp, X)
            inf_min, inf_max = inf_map.min(), inf_map.max()
            inf_map = (inf_map - inf_min) / (inf_max - inf_min) if inf_max > inf_min else np.zeros_like(inf_map)
        elif inf_criteria == 'variance':
            _, inf_map = gp.predict(X, return_std=True)
            inf_min, inf_max = inf_map.min(), inf_map.max()
            inf_map = (inf_map - inf_min) / (inf_max - inf_min) if inf_max > inf_min else np.zeros_like(inf_map)
        else:  # dyn_variance, dyn_cohns
            inf_map = None
            
        next_idx = planner.select_next(inf_map, sampled)
        
    deployment_time = time.time() - start_deploy
    
    # 3. Retroactive Error
    y_pred_before = gp.predict(X[[next_idx]])[0]
    retroactive_error = np.abs(y[next_idx] - y_pred_before)
    
    # Calculate travel cost using real measured times
    cost = get_action_cost(sampled[-1], [next_idx], df, size, len(sampled), deployment_time, training_time)[0]
    
    
    
    
    # Append to log
    pd.DataFrame([{
        'sample_index': next_idx, 
        'nrmse_test': n_test, 
        'nrmse_val': n_val,
        'action_cost': cost,
        'training_time': training_time,
        'deployment_time': deployment_time,
        'retroactive_error': retroactive_error
    }]).to_csv(log_path, mode='a', header=False, index=False)

    #Threshold-based Stopping
    # If the map quality is good enough, stop the mission.
    if n_val <= STOPPING_THRESHOLD:
        print(f"Target NRMSE reached: {n_val:.4f} <= {STOPPING_THRESHOLD}")
        sys.exit(100)

    

if __name__ == "__main__":
    # 1. Grab fixed arguments
    e_id, p_name, s_cat, s_seed, norm_reward, inf_criteria = sys.argv[1:7]

    # 2. Grab everything else (the --key value pairs)
    extra_args = sys.argv[7:]
    hp_dict = {}
    
    # Step through extra_args by 2 (key, then value)
    for i in range(0, len(extra_args), 2):
        key = extra_args[i].replace("--", "")
        val = extra_args[i+1]
        
        # Try to convert to float/int, otherwise keep as string
        try:
            hp_dict[key] = float(val) if "." in val else int(val)
        except ValueError:
            hp_dict[key] = val

    # 3. Call step with the dictionary unpacked
    step(e_id, p_name, s_cat, s_seed, norm_reward, inf_criteria, **hp_dict)