#!/bin/bash
#SBATCH --job-name=cip
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0-9
#
# SLURM launcher for the CIP experiments of Section 4: one array task per seed.
# The experiment scripts' defaults are the paper's hyperparameters (Table 3), so no
# extra arguments are needed.
#
#   sbatch scripts/cip/experiments/run.sh cart_pole
#   sbatch scripts/cip/experiments/run.sh double_pendulum
#   sbatch scripts/cip/experiments/run.sh triple_pendulum
#   sbatch scripts/cip/experiments/run.sh humulum
#
# Without SLURM the same command runs a single seed:
#   bash scripts/cip/experiments/run.sh humulum 3

set -euo pipefail

ENV_NAME="${1:?usage: run.sh <cart_pole|double_pendulum|triple_pendulum|humulum> [seed]}"
SEED="${2:-${SLURM_ARRAY_TASK_ID:-0}}"

## activate the conda environment when a conda install is found; otherwise assume the env is already active
for conda_sh in ~/miniforge3/etc/profile.d/conda.sh ~/miniconda3/etc/profile.d/conda.sh /opt/miniconda3/etc/profile.d/conda.sh; do
    if [ -f "$conda_sh" ]; then source "$conda_sh" && conda activate val; break; fi
done

export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"                                  # offscreen rendering on a headless GPU box
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# GPU selection is left to SLURM (CUDA_VISIBLE_DEVICES); set it yourself when running by hand.

python "scripts/cip/experiments/${ENV_NAME}.py" --seed "$SEED"
