#!/bin/bash
#SBATCH --job-name=crps_trn
#SBATCH --output=logs/crps_train.out
#SBATCH --error=logs/crps_train.err
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --time=23:59:59

eval "$(conda shell.bash hook)"
conda activate glp
python train_model.py