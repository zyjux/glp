#!/bin/bash
#SBATCH --job-name=nn_training
#SBATCH --output=logs/synth_networks_train.out
#SBATCH --error=logs/synth_networks_train.err
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00

eval "$(conda shell.bash hook)"
conda activate glp
python ~/glp/synth_networks_train.py
