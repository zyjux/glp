#!/usr/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate glp

TRAIN_SCRIPT="/home/lverhoef/glp/glp_ri/train_model.py"

EXPERIMENT_DIR="/home/lverhoef/glp/glp_ri/experiments/starting_baseline/"

# Train CNN while creating the appropriate logs
printf "\nTraining CNN"
CFG_FILE="$EXPERIMENT_DIR/test_config_file.yml"
LOG_FILE="$EXPERIMENT_DIR/cnn_apply.log"
ERR_FILE="$EXPERIMENT_DIR/cnn_apply.err"
/home/lverhoef/.local/bin/safepython.sh $TRAIN_SCRIPT $CFG_FILE > $LOG_FILE 2> $ERR_FILE

# Then train GLP CNN
printf "\nTraining GLP CNN"
CFG_FILE="$EXPERIMENT_DIR/test_glp_config_file.yml"
LOG_FILE="$EXPERIMENT_DIR/glp_apply.log"
ERR_FILE="$EXPERIMENT_DIR/glp_apply.err"
/home/lverhoef/.local/bin/safepython.sh $TRAIN_SCRIPT $CFG_FILE > $LOG_FILE 2> $ERR_FILE
