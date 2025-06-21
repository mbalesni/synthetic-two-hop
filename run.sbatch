#!/bin/bash
#SBATCH --job-name=latent
#SBATCH --output=slurm_runs/job-%j.out
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slack:${SLACK_USER}
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --time=1-23:55:00

conda init
source ~/.bashrc
conda activate latent

"$@"