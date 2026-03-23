#!/bin/bash
# All parameters read from environment variables with sensible local defaults.
# On Kaggle, export env vars in the notebook before calling this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 train.py \
  --exp_name "${EXP_NAME:-hitnet_custom}" \
  --model "${MODEL:-HITNet_SF}" \
  --data_augmentation "${DATA_AUG:-1}" \
  --data_type_train depth \
  --data_root_train "${DATA_ROOT:-data}" \
  --data_list_train "${TRAIN_CSV:-data/processed_data/train.csv}" \
  --data_size_train ${DATA_SIZE_H:-640} ${DATA_SIZE_W:-480} \
  --data_type_val depth \
  --data_root_val "${DATA_ROOT:-data}" \
  --data_list_val "${VAL_CSV:-data/processed_data/val.csv}" \
  --data_size_val ${DATA_SIZE_H:-640} ${DATA_SIZE_W:-480} \
  --batch_size "${BATCH_SIZE:-1}" \
  --batch_size_val "${BATCH_SIZE_VAL:-1}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-1e-3}" \
  --max_disp "${MAX_DISP:-160}" \
  --max_epochs "${MAX_EPOCHS:-100}"
