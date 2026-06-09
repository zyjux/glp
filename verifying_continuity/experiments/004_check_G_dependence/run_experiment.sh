#!/usr/bin/bash

# Activate conda
eval "$(conda shell.bash hook)"
conda activate glp

RUN_SCRIPT="/home/lverhoef/glp/verifying_continuity/run_test.py"
EXPERIMENT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for i in {1..3}
do
  CFG_FILE="$EXPERIMENT_DIR/test_${i}_config.yaml"
  LOG_FILE="$EXPERIMENT_DIR/test_${i}.log"
  ERR_FILE="$EXPERIMENT_DIR/test_${i}.err"

  python $RUN_SCRIPT $CFG_FILE $EXPERIMENT_DIR >$LOG_FILE 2> $ERR_FILE
done

