#!/bin/bash
#SBATCH --time=24:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48GB
#SBATCH --partition=gpu-a100 --gres=gpu:1
#SBATCH --output=log%j.txt
#SBATCH --error=error%j.txt
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=mdjaberal.nahian@ucalgary.ca
# ====================================
eval "$(conda shell.bash hook)"
# ====================================
conda activate SAM_Project


# Submit the Python files as Slurm jobs
python /home/mdjaberal.nahian/lightning-sam/lightning_sam/Train_Updated.py
python /home/mdjaberal.nahian/lightning-sam/lightning_sam/Train_H_100_error_1.py
