#!/bin/bash
# Run state-entropy triple_pendulum for a range of seeds SEQUENTIALLY (no SLURM; use tmux).
# Uses the experiment script's DEFAULT parameters (steps=2000, gear=25, h=128, shots=2048, ...).
# Activate your env first, e.g.:  conda activate val
#
# Usage: scripts/state_entropy/experiments/run_triple_pendulum_seeds.sh [START] [END]   # default 0 9

set -euo pipefail
START="${1:-0}"
END="${2:-9}"

for seed in $(seq "$START" "$END"); do
    echo "=== state_entropy triple_pendulum seed=${seed} ==="
    python scripts/state_entropy/experiments/triple_pendulum.py --seed "$seed"
done

echo "Done: triple_pendulum seeds ${START}..${END}"
