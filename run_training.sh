#!/bin/sh
#SBATCH --job-name=llm
#SBATCH -t 72:00:00
#SBATCH -o slurm_logs/slurm-%j.out  # %j = job ID
#SBATCH --partition=gypsum-rtx8000
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem 30GB

python full_fine_tuning.py