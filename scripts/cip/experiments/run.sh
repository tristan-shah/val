#!/bin/bash
#SBATCH --job-name=humanoid          # Name of the job
#SBATCH --partition=h100          # Request the h100 partition
#SBATCH --gres=gpu:1              # Request GPU
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --time=24:00:00          # Max runtime
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0-9

source ~/miniforge3/etc/profile.d/conda.sh 
conda activate val
export PYTHONUNBUFFERED=1  # Add this

## single pendulum
# python scripts/cip/experiments/single_pendulum.py --seed 0 --component ol --horizon 1024 --shots 1024 --elite_frac 0.05 --dt 0.01 --steps 1200

## cart pole
# python scripts/cip/experiments/cart_pole.py --seed $SLURM_ARRAY_TASK_ID --component ol --horizon 512 --shots 1024 --iterations 1
# python scripts/cip/experiments/cart_pole.py --seed $SLURM_ARRAY_TASK_ID --component ol --horizon 400 --shots 512 --iterations 1

## double pendulum
# python scripts/cip/experiments/double_pendulum.py --seed $SLURM_ARRAY_TASK_ID --component ol --shots 1024 --steps 2400

## triple pendulum
python scripts/cip/experiments/triple_pendulum.py --seed $SLURM_ARRAY_TASK_ID --component ol --horizon 128 --shots 2048 --gear 25 --iterations 10 --beta 2.5 --steps 2400

## humulum
# python scripts/cip/experiments/humulum.py --component ol --seed $SLURM_ARRAY_TASK_ID --beta 9.0 --horizon 512 --shots 1024 --iterations 1 --elite_frac 0.2 --warmstart 10
# python scripts/cip/experiments/humulum.py --component ol --seed $SLURM_ARRAY_TASK_ID --beta 10.0 --horizon 512 --shots 2048 --iterations 1 --elite_frac 0.2
# python scripts/cip/experiments/humulum.py --component ol --seed $SLURM_ARRAY_TASK_ID --beta 7.0 --horizon 700 --shots 1024 --iterations 1 --elite_frac 0.2