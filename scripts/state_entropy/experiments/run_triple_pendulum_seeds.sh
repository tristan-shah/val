#!/bin/bash
# Runs the state-entropy triple_pendulum baseline (Table 1, "APT (MPC)") sequentially over seeds.
# Uses the paper's state-entropy settings: control penalty 0.1 and a single CEM iteration.
#
#   scripts/state_entropy/experiments/run_triple_pendulum_seeds.sh [START] [END]   # default seeds 0..9
#
# Activate the environment first (conda activate val). Set CUDA_VISIBLE_DEVICES to pick a GPU.

set -euo pipefail

START="${1:-0}"
END="${2:-9}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

for seed in $(seq "$START" "$END"); do
    echo "=== state_entropy triple_pendulum seed=${seed} ==="
    python scripts/state_entropy/experiments/triple_pendulum.py --seed "$seed" --beta 0.1 --iterations 1
done

echo "Done: triple_pendulum seeds ${START}..${END}"
