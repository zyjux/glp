#!/bin/bash
#SBATCH --job-name=gen_labels
#SBATCH --output=logs/label_generation.out
#SBATCH --error=logs/label_generation.err
#SBATCH --ntasks=2
#SBATCH --time=24:00:00

eval "$(conda shell.bash hook)"
conda activate glp
python ~/glp/glp_ri/generate_labels.py --filename train_labels.json --years 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2015 2016 2017 2018 2019 &
python ~/glp/glp_ri/generate_labels.py --filename valid_labels.json --years 2005 2006 2007 2008 2009 &
wait
