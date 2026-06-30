#!/usr/bin/bash

# Activate conda
eval "$(conda shell.bash hook)"
conda activate glp

RUN_SCRIPT="/home/lverhoef/glp/verifying_continuity/run_test.py"
EXPERIMENT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CFG_FILE="$EXPERIMENT_DIR/test_config.yaml"
LOG_FILE="$EXPERIMENT_DIR/test.log"
ERR_FILE="$EXPERIMENT_DIR/test.err"

python $RUN_SCRIPT $CFG_FILE $EXPERIMENT_DIR >$LOG_FILE 2> $ERR_FILE
