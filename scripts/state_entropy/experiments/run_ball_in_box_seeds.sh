#!/bin/bash
# Run ball-in-box STATE-ENTROPY (APT) experiments sequentially over a range of seeds.
#
# Usage:
#   scripts/state_entropy/experiments/run_ball_in_box_seeds.sh [START] [END]
#
#   START  first seed to run (inclusive)   default: 10
#   END    last seed to run  (inclusive)   default: START + 9  (i.e. 10 seeds)
#
# Examples:
#   scripts/state_entropy/experiments/run_ball_in_box_seeds.sh            # seeds 10..19
#   scripts/state_entropy/experiments/run_ball_in_box_seeds.sh 20         # seeds 20..29
#   scripts/state_entropy/experiments/run_ball_in_box_seeds.sh 10 14      # seeds 10..14

set -euo pipefail

START="${1:-10}"
END="${2:-$((START + 9))}"

# fixed experiment parameters (match the CIP ball_in_box sweep for head-to-head)
HORIZON=128
STEPS=1000
RESTITUTION=0.5
K=12

for seed in $(seq "$START" "$END"); do
    echo "=== ball_in_box (state_entropy) seed=${seed} ==="
    python scripts/state_entropy/experiments/ball_in_box.py \
        --horizon "$HORIZON" \
        --steps "$STEPS" \
        --restitution "$RESTITUTION" \
        --k "$K" \
        --seed "$seed" \
        --random_spawn
done

echo "Done: seeds ${START}..${END}"
