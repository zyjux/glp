#!/bin/bash
#SBATCH --job-name=no_softmax
#SBATCH --output=logs/no_softmax_train.out
#SBATCH --error=logs/no_softmax_train.err
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --time=23:59:59

eval "$(conda shell.bash hook)"
conda activate glp
python train_model_no_softmax.py