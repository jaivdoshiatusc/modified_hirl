#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:1080:1
#SBATCH --cpus-per-task=5
#SBATCH --time=1-00:00:00
#SBATCH --nodelist=ink-lucy
#SBATCH --output=/home/jaiv/drl-dqn-atari-pong/slurm_outputs/%j.out

source /opt/anaconda3/bin/activate /home/jaiv/miniconda3/envs/hirl

srun python3 /home/jaiv/modified_dqn_pong/main.py