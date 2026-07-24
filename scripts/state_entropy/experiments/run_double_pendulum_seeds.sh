#!/bin/bash
# Run state-entropy double_pendulum for a range of seeds SEQUENTIALLY (no SLURM; use tmux).
# Uses the experiment script's DEFAULT parameters (steps=2000, h=512, shots=1024, ...).
# Activate your env first, e.g.:  conda activate val
#
# Usage: scripts/state_entropy/experiments/run_double_pendulum_seeds.sh [START] [END]   # default 0 9

set -euo pipefail
START="${1:-0}"
END="${2:-9}"

for seed in $(seq "$START" "$END"); do
    echo "=== state_entropy double_pendulum seed=${seed} ==="
    python scripts/state_entropy/experiments/double_pendulum.py --seed "$seed"
done

echo "Done: double_pendulum seeds ${START}..${END}"
