#!/usr/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate glp

APPLY_SCRIPT="/home/lverhoef/glp/glp_ri/apply_model.py"

DATA_DIR="/mnt/data2/lverhoef/RI/learning_examples/rotated_with_storm_motion/imputed/normalized"
DATA_FILE="valid_labels.json"

EXPERIMENT_DIR="/home/lverhoef/glp/glp_ri/experiments/006_updated_batch_size_old_encoder"

# Train CNN while creating the appropriate logs
printf "\nApplying CNN"
CFG_FILE="$EXPERIMENT_DIR/test_config_file.yml"
OUT_FILE="$EXPERIMENT_DIR/cnn_validation.nc"
LOG_FILE="$EXPERIMENT_DIR/cnn_apply.log"
ERR_FILE="$EXPERIMENT_DIR/cnn_apply.err"
/home/lverhoef/.local/bin/safepython.sh $APPLY_SCRIPT $CFG_FILE $OUT_FILE \
  --data_dir=$DATA_DIR --data_file=$DATA_FILE --batch_size=32 > $LOG_FILE 2> $ERR_FILE

# Then train GLP CNN
printf "\nApplying GLP CNN"
CFG_FILE="$EXPERIMENT_DIR/test_glp_config_file.yml"
OUT_FILE="$EXPERIMENT_DIR/glp_validation.nc"
LOG_FILE="$EXPERIMENT_DIR/glp_apply.log"
ERR_FILE="$EXPERIMENT_DIR/glp_apply.err"
/home/lverhoef/.local/bin/safepython.sh $APPLY_SCRIPT $CFG_FILE $OUT_FILE \
  --data_dir=$DATA_DIR --data_file=$DATA_FILE --batch_size=32 > $LOG_FILE 2> $ERR_FILE
