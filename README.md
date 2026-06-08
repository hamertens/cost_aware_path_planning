# cost_aware_path_planning

Active learning framework for cost-aware spatial exploration. A Gaussian Process is
iteratively refined by querying the environment at locations chosen by a planner; the
loop stops when validation NRMSE drops below 0.03.

## Directory layout

```
environments/
  small/          # 25×25 grids
    exploration/  # env_expl_NN.csv  — full ground-truth field used during sampling
    test/         # env_test_NN.csv  — held-out set for test NRMSE
    validation/   # env_val_NN.csv   — held-out set for early-stopping criterion
  large/          # 50×50 grids (same structure)
planners/         # planner implementations (see below)
init.py           # one-time setup script
step.py           # single active-learning step script
active_learning.sh# orchestration loop
```

## init.py

Run once before the loop. Loads the three CSV splits for the chosen environment, places
the agent at two seed locations (strategy-dependent), fits an initial GP to both points,
and writes `output_data/output_data_env<N>_<planner>_<size>/seed_<S>/inf_<criteria>/norm_<norm>/log.csv`
with the first two rows (sample index, nrmse_test, nrmse_val, action_cost, training_time,
deployment_time, retroactive_error).

## step.py

Run once per iteration. Reads the current log, re-fits the GP on all sampled points,
builds an information map (GP variance or Cohn's ALC), passes it to the planner's
`select_next`, appends the chosen point to the log, and exits with code **100** when
`nrmse_val ≤ 0.03` to signal convergence.

## Planners

| Name | Type | Key hyperparameters |
|---|---|---|
| `myopic` | adaptive | `budget_a`, `budget_b` |
| `constrained` | adaptive | `budget_a`, `budget_b` |
| `mcts` | adaptive | `simulations`, `horizon`, `gamma`, `c_param`, `k_a`, `alpha_a` |
| `rig` | adaptive | `horizon`, `samples`, `gamma` |
| `iig` | adaptive | `horizon`, `ric_threshold` |
| `ktbo` | adaptive | `simulations`, `ls_y`, `n_samples`, `num_anchors`, `horizon_scale`, `gamma` |
| `ipp_mpe` | adaptive | — |
| `ergodic` | adaptive | — |
| `snake` | geometric | — |
| `spiral` | geometric | — |
| `sobol` | geometric | — |

## Running

```bash
bash active_learning.sh <env_id> <planner> <size> <seed> <normalize_reward> <inf_criteria> [--key value ...]
```

| Argument | Values |
|---|---|
| `env_id` | integer (e.g. `1`–`10`) |
| `planner` | see table above |
| `size` | `small` or `large` |
| `seed` | integer |
| `normalize_reward` | `true` or `false` |
| `inf_criteria` | `variance` / `v`, `cohns` / `c`, `dyn_variance` / `dv`, `dyn_cohns` / `dc` |
| extra `--key value` | planner-specific hyperparameters (forwarded to `step.py` only) |

### Example — MCTS with custom hyperparameters

```bash
bash active_learning.sh 3 mcts small 42 true variance \
  --simulations 800 --horizon 5 --gamma 0.85 --c_param 1.0
```

This runs environment 03, small grid, seed 42, with reward normalization and GP-variance
as the information criterion. MCTS will use 800 simulations, a planning horizon of 5
steps, discount factor 0.85, and UCB exploration constant 1.0.

Output is written to:
```
output_data/output_data_env3_mcts_small/seed_42/inf_variance/norm_true/log.csv
```
