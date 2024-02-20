#!/bin/bash
#SBATCH --job-name=mae_trn
#SBATCH --output=logs/mae_train.out
#SBATCH --error=logs/mae_train.err
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --time=23:59:59

eval "$(conda shell.bash hook)"
conda activate glp
python simple_model_train.py