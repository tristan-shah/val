#!/bin/bash
# Run state-entropy triple_pendulum for a range of seeds SEQUENTIALLY (no SLURM; use tmux).
# Uses the experiment script's DEFAULT parameters (steps=2000, gear=25, h=128, shots=2048, ...).
# Activate your env first, e.g.:  conda activate val
#
# Usage: scripts/state_entropy/experiments/run_triple_pendulum_seeds.sh [START] [END]   # default 0 9

set -euo pipefail
START="${1:-0}"
END="${2:-9}"

## control penalty weight. Goes into the output dir name (beta=$BETA), so changing it
## writes to new dirs rather than overwriting previous runs.
BETA=0.1
ITER=1

for seed in $(seq "$START" "$END"); do
    echo "=== state_entropy triple_pendulum seed=${seed} beta=${BETA} ==="
    python scripts/state_entropy/experiments/triple_pendulum.py --seed "$seed" --beta "$BETA" --iterations "$ITER"
done

echo "Done: triple_pendulum seeds ${START}..${END}"
