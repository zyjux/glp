#!/bin/bash
#SBATCH --job-name=train_crossentropy_cnn
#SBATCH --output=logs/crossentropy_train.out
#SBATCH --error=logs/crossentropy_train.err
#SBATCH --ntasks=1
#SBATCH --partition=day-long-gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --time=23:59:59

eval "$(conda shell.bash hook)"
conda activate glp
python train_model_crossentropy.py
