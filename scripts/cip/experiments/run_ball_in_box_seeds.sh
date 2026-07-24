#!/bin/bash
# Run ball-in-box CIP experiments sequentially over a range of seeds.
#
# Usage:
#   scripts/cip/experiments/run_ball_in_box_seeds.sh [START] [END]
#
#   START  first seed to run (inclusive)   default: 10
#   END    last seed to run  (inclusive)   default: START + 9  (i.e. 10 seeds)
#
# Examples:
#   scripts/cip/experiments/run_ball_in_box_seeds.sh            # seeds 10..19
#   scripts/cip/experiments/run_ball_in_box_seeds.sh 20         # seeds 20..29
#   scripts/cip/experiments/run_ball_in_box_seeds.sh 10 14      # seeds 10..14

set -euo pipefail

START="${1:-10}"
END="${2:-$((START + 9))}"

# fixed experiment parameters (match the existing seed=0..9 sweep)
HORIZON=128
STEPS=1000
RESTITUTION=0.5

for seed in $(seq "$START" "$END"); do
    echo "=== ball_in_box seed=${seed} ==="
    python scripts/cip/experiments/ball_in_box.py \
        --horizon "$HORIZON" \
        --steps "$STEPS" \
        --restitution "$RESTITUTION" \
        --seed "$seed" \
        --random_spawn
done

echo "Done: seeds ${START}..${END}"
