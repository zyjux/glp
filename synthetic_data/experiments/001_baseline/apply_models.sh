#!/usr/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate glp

APPLY_SCRIPT="/home/lverhoef/glp/synthetic_data/apply_model.py"

EXPERIMENT_DIR="/home/lverhoef/glp/synthetic_data/experiments/001_baseline"
DATA_FILE="/mnt/data2/lverhoef/synthetic_ellipses/train_valid_ds.nc"

# Train CNN while creating the appropriate logs
printf "\nApplying CNN"
CFG_FILE="$EXPERIMENT_DIR/test_config_file.yml"
OUT_FILE="$EXPERIMENT_DIR/cnn_validation.nc"
LOG_FILE="$EXPERIMENT_DIR/cnn_apply.log"
ERR_FILE="$EXPERIMENT_DIR/cnn_apply.err"
/home/lverhoef/.local/bin/safepython.sh $APPLY_SCRIPT $CFG_FILE $OUT_FILE \
  --data_file=$DATA_FILE > $LOG_FILE 2> $ERR_FILE

# Then train GLP CNN
printf "\nApplying GLP CNN"
CFG_FILE="$EXPERIMENT_DIR/test_glp_config_file.yml"
OUT_FILE="$EXPERIMENT_DIR/glp_validation.nc"
LOG_FILE="$EXPERIMENT_DIR/glp_apply.log"
ERR_FILE="$EXPERIMENT_DIR/glp_apply.err"
/home/lverhoef/.local/bin/safepython.sh $APPLY_SCRIPT $CFG_FILE $OUT_FILE \
  --data_file=$DATA_FILE > $LOG_FILE 2> $ERR_FILE
