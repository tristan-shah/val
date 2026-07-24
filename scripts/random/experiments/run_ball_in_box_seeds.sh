#!/bin/bash
# Run ball-in-box PURE-RANDOM-CONTROL baseline sequentially over a range of seeds.
#
# Usage:
#   scripts/random/experiments/run_ball_in_box_seeds.sh [START] [END]
#
#   START  first seed to run (inclusive)   default: 0
#   END    last seed to run  (inclusive)   default: START + 9  (i.e. 10 seeds)
#
# Examples:
#   scripts/random/experiments/run_ball_in_box_seeds.sh            # seeds 0..9
#   scripts/random/experiments/run_ball_in_box_seeds.sh 10 19      # seeds 10..19

set -euo pipefail

START="${1:-0}"
END="${2:-$((START + 9))}"

# fixed experiment parameters (match the CIP / state-entropy ball_in_box sweeps)
STEPS=1000
RESTITUTION=0.5

for seed in $(seq "$START" "$END"); do
    echo "=== ball_in_box (random) seed=${seed} ==="
    python scripts/random/experiments/ball_in_box.py \
        --steps "$STEPS" \
        --restitution "$RESTITUTION" \
        --seed "$seed" \
        --random_spawn
done

echo "Done: seeds ${START}..${END}"
