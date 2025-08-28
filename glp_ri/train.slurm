#!/bin/bash
#SBATCH --job-name=train_cnn
#SBATCH --output=logs/crps_train.out
#SBATCH --error=logs/crps_train.err
#SBATCH --ntasks=1
#SBATCH --partition=day-long-gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --time=23:59:59

eval "$(conda shell.bash hook)"
nvidia-smi
conda activate glp
python train_model.py
