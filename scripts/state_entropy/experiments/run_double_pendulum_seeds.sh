#!/bin/bash
# Runs the state-entropy double_pendulum baseline (Table 1, "APT (MPC)") sequentially over seeds.
# Uses the script's defaults (the CIP hyperparameters of Table 3).
#
#   scripts/state_entropy/experiments/run_double_pendulum_seeds.sh [START] [END]   # default seeds 0..9
#
# Activate the environment first (conda activate val). Set CUDA_VISIBLE_DEVICES to pick a GPU.

set -euo pipefail

START="${1:-0}"
END="${2:-9}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

for seed in $(seq "$START" "$END"); do
    echo "=== state_entropy double_pendulum seed=${seed} ==="
    python scripts/state_entropy/experiments/double_pendulum.py --seed "$seed"
done

echo "Done: double_pendulum seeds ${START}..${END}"
