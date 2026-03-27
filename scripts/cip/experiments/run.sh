#!/bin/bash
#SBATCH --job-name=CIP          # Name of the job
#SBATCH --partition=h100          # Request the h100 partition
#SBATCH --gres=gpu:1              # Request GPU
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --time=24:00:00          # Max runtime
# SBATCH --output=slurm-%j.out    # Output file
# SBATCH --error=slurm-%j.err     # Error file

source ~/miniforge3/etc/profile.d/conda.sh 
conda activate val
export PYTHONUNBUFFERED=1  # Add this

# Run your Python script
# python scripts/cip/experiments/cart_pole.py --seed $SLURM_ARRAY_TASK_ID --component ol --horizon 400
python scripts/cip/experiments/single_pendulum.py --seed 5 --component cip --horizon 150