#!/bin/bash
#SBATCH --job-name=apply_crps_cnn
#SBATCH --output=logs/crossentropy_apply.out
#SBATCH --error=logs/crossentropy_apply.err
#SBATCH --ntasks=1
#SBATCH --partition=short-gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:59:59

eval "$(conda shell.bash hook)"
conda activate glp
python apply_model.py
