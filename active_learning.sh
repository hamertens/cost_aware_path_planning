#!/bin/bash
conda activate conda_env
# 1. Capture the fixed positional arguments
ENV_ID=${1:-1}
PLANNER=${2:-unconstrained}
SIZE=${3:-small}
SEED=${4:-1}
NORMALIZE_REWARD=${5:-true} # "true", "false"
INF_CRITERIA=${6:-variance} # "variance" / "v" or "cohns" / "c" or "dyn_variance" / "dv" or "dyn_cohns" / "dc"

# 2. Shift the argument list by 6.
shift 6

echo "METADATA: ENV_ID=$ENV_ID | PLANNER=$PLANNER | SIZE=$SIZE | SEED=$SEED | NORMALIZE_REWARD=$NORMALIZE_REWARD | INF_CRITERIA=$INF_CRITERIA"
echo "EXTRA HYPERPARAMS: $@"

python3 init.py $ENV_ID $PLANNER $SIZE $SEED $NORMALIZE_REWARD $INF_CRITERIA
for i in {1..1000}; do
   printf "Step %03d: " $i
   python3 step.py $ENV_ID $PLANNER $SIZE $SEED $NORMALIZE_REWARD $INF_CRITERIA "$@"

   if [ $? -eq 100 ]; then
      echo "Validation converged. Stopping."
      break
   fi
done
