#!/bin/bash
# Runs the state-entropy humulum baseline (Table 1, "APT (MPC)") sequentially over seeds.
# Uses the paper's state-entropy setting: control penalty 1.0.
#
#   scripts/state_entropy/experiments/run_humulum_seeds.sh [START] [END]   # default seeds 0..9
#
# Activate the environment first (conda activate val). Set CUDA_VISIBLE_DEVICES to pick a GPU.

set -euo pipefail

START="${1:-0}"
END="${2:-9}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

for seed in $(seq "$START" "$END"); do
    echo "=== state_entropy humulum seed=${seed} ==="
    python scripts/state_entropy/experiments/humulum.py --seed "$seed" --beta 1.0
done

echo "Done: humulum seeds ${START}..${END}"
