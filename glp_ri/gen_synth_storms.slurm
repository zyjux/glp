#!/bin/bash
#SBATCH --job-name=gen_storms
#SBATCH --output=logs/synth_storm_generation.out
#SBATCH --error=logs/synth_storm_generation.err
#SBATCH --time=24:00:00

eval "$(conda shell.bash hook)"
conda activate glp
python generate_synth_storms.py
