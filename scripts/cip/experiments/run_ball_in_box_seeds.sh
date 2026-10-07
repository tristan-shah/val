#!/bin/bash
# Runs the ball-in-box CIP experiment (Appendix A.1) sequentially over a range of seeds.
#
#   scripts/cip/experiments/run_ball_in_box_seeds.sh [START] [END]     # default seeds 0..9
#
# The paper's Figure 6 pools seeds 0..49 at restitution 0.5.

set -euo pipefail

START="${1:-0}"
END="${2:-$((START + 9))}"

for seed in $(seq "$START" "$END"); do
    echo "=== ball_in_box (CIP) seed=${seed} ==="
    python scripts/cip/experiments/ball_in_box.py --restitution 0.5 --seed "$seed"
done

echo "Done: seeds ${START}..${END}"
