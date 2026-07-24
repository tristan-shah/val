#!/bin/bash
# Run state-entropy humulum (gibbon) for a range of seeds SEQUENTIALLY (no SLURM; use tmux).
# Uses the experiment script's DEFAULT parameters (steps=1200, warmstart=10, h=512, shots=1024, ...).
# Activate your env first, e.g.:  conda activate val
#
# Note: humulum.py pins CUDA_VISIBLE_DEVICES=1 internally, so this runs on GPU 1.
#
# Usage: scripts/state_entropy/experiments/run_humulum_seeds.sh [START] [END]   # default 0 9

set -euo pipefail
START="${1:-0}"
END="${2:-9}"

for seed in $(seq "$START" "$END"); do
    echo "=== state_entropy humulum seed=${seed} ==="
    python scripts/state_entropy/experiments/humulum.py --seed "$seed"
done

echo "Done: humulum seeds ${START}..${END}"
