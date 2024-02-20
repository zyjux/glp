#!/bin/bash
#SBATCH --job-name=storm_gen
#SBATCH --output=logs/storms_generation.out
#SBATCH --error=logs/storms_generation.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

eval "$(conda shell.bash hook)"
conda activate glp
python generate_synth_storms.py
