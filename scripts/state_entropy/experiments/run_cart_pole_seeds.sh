#!/bin/bash
# Run state-entropy cart_pole for a range of seeds SEQUENTIALLY (no SLURM; use tmux).
# Uses the experiment script's DEFAULT parameters (steps=1200, h=400, shots=512, ...).
# Activate your env first, e.g.:  conda activate val
#
# Usage: scripts/state_entropy/experiments/run_cart_pole_seeds.sh [START] [END]   # default 0 9

set -euo pipefail
START="${1:-0}"
END="${2:-9}"

for seed in $(seq "$START" "$END"); do
    echo "=== state_entropy cart_pole seed=${seed} ==="
    python scripts/state_entropy/experiments/cart_pole.py --seed "$seed"
done

echo "Done: cart_pole seeds ${START}..${END}"
