#!/usr/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate glp

train_script="/home/lverhoef/glp/synthetic_data/train_model.py"
save_file_base="/home/lverhoef/glp/synthetic_data/experiments/002_test_dataset_size/models/cnn"
glp_save_file_base="/home/lverhoef/glp/synthetic_data/experiments/002_test_dataset_size/models/glp_cnn"

experiment_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

dataset_ratios=(0.5 0.25 0.1 0.05 0.01 0.005 0.001)

for ratio in "${dataset_ratios[@]}"; do

  # Train CNN while creating the appropriate logs
  printf "\nTraining CNN"
  cfg_file="$experiment_dir/test_config_file.yml"
  sed -i "s/\(ds_size_ratio:\).*/\1 ${ratio}/" ${cfg_file}
  safe_ratio=${ratio//\./_}
  full_save_file="${save_file_base}_${safe_ratio}.pt"
  sed -i "s/\(model_save_file:\).*/\1 \"${full_save_file//\//\\\/}\"/" ${cfg_file}
  log_file="$experiment_dir/cnn.log"
  err_file="$experiment_dir/cnn.err"
  /home/lverhoef/.local/bin/safepython.sh $train_script $cfg_file > $log_file 2> $err_file

  # Then train GLP CNN
  printf "\nTraining GLP CNN"
  cfg_file="$experiment_dir/test_glp_config_file.yml"
  sed -i "s/\(ds_size_ratio:\).*/\1 ${ratio}/" ${cfg_file}
  full_save_file="${glp_save_file_base}_${safe_ratio}.pt"
  sed -i "s/\(model_save_file:\).*/\1 \"${full_save_file//\//\\\/}\"/" ${cfg_file}
  log_file="$experiment_dir/glp.log"
  err_file="$experiment_dir/glp.err"
  /home/lverhoef/.local/bin/safepython.sh $train_script $cfg_file > $log_file 2> $err_file

done
