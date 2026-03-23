#!/bin/bash
# Auto-detect environment: Kaggle vs local
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "/kaggle/input" ]; then
    # Kaggle environment
    DATA_ROOT="/kaggle/input/stereo-smallbaseline"
    TRAIN_CSV="${DATA_ROOT}/train.csv"
    VAL_CSV="${DATA_ROOT}/val.csv"
    BATCH_SIZE=32
    BATCH_SIZE_VAL=16
    NUM_WORKERS=8
else
    # Local environment
    DATA_ROOT="data"
    TRAIN_CSV="data/processed_data/train.csv"
    VAL_CSV="data/processed_data/val.csv"
    BATCH_SIZE=1
    BATCH_SIZE_VAL=1
    NUM_WORKERS=4
fi

python3 train.py \
  --exp_name hitnet_custom \
  --model HITNet_SF \
  --data_augmentation 1 \
  --data_type_train depth \
  --data_root_train "$DATA_ROOT" \
  --data_list_train "$TRAIN_CSV" \
  --data_size_train 640 480 \
  --data_type_val depth \
  --data_root_val "$DATA_ROOT" \
  --data_list_val "$VAL_CSV" \
  --data_size_val 640 480 \
  --batch_size "$BATCH_SIZE" \
  --batch_size_val "$BATCH_SIZE_VAL" \
  --num_workers "$NUM_WORKERS" \
  --lr 1e-3 \
  --max_disp 160 \
  --max_epochs 100
