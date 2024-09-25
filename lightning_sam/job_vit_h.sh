#!/bin/bash
#SBATCH --time=15:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=96GB
#SBATCH --partition=gpu-a100 --gres=gpu:1
#SBATCH --output=log%j.txt
#SBATCH --error=error%j.txt
# ====================================
conda activate SAM_Project


# Submit the Python files as Slurm jobs
python /home/mdjaberal.nahian/lightning-sam/lightning_sam/Train_Updated_Vit_H.py