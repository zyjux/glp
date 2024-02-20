#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH --output=logs/gen_data.out
#SBATCH --error=logs/gen_data.err
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --time=23:59:59

eval "$(conda shell.bash hook)"
conda activate glp
python generate_data.py