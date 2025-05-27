#!/bin/bash
#SBATCH --job-name=train_glp_cnn
#SBATCH --output=logs/glp_train.out
#SBATCH --error=logs/glp_train.err
#SBATCH --ntasks=1
#SBATCH --partition=day-long-gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --time=23:59:59

eval "$(conda shell.bash hook)"
nvidia-smi
conda activate glp
python train_glp_model.py
