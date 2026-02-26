#!/usr/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate glp

TRAIN_SCRIPT="/home/lverhoef/glp/glp_ri/train_model.py"

# Train CNN while creating the appropriate logs
printf "\nTraining CNN"
CFG_FILE="/home/lverhoef/glp/glp_ri/experiments/starting_baseline/test_config_file.yml"
LOG_FILE="/home/lverhoef/glp/glp_ri/experiments/starting_baseline/cnn.log"
ERR_FILE="/home/lverhoef/glp/glp_ri/experiments/starting_baseline/cnn.err"
/home/lverhoef/.local/bin/safepython.sh $TRAIN_SCRIPT $CFG_FILE > $LOG_FILE 2> $ERR_FILE

# Then train GLP CNN
printf "\nTraining GLP CNN"
CFG_FILE="/home/lverhoef/glp/glp_ri/experiments/starting_baseline/test_glp_config_file.yml"
LOG_FILE="/home/lverhoef/glp/glp_ri/experiments/starting_baseline/glp.log"
ERR_FILE="/home/lverhoef/glp/glp_ri/experiments/starting_baseline/glp.err"
/home/lverhoef/.local/bin/safepython.sh $TRAIN_SCRIPT $CFG_FILE > $LOG_FILE 2> $ERR_FILE
