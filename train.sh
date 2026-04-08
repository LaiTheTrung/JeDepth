#!/usr/bin/env bash
set -e

CONFIG=${CONFIG:-configs/Real/stereo_smallbaseline.yaml}
SEED=${SEED:-42}
NUM_GPUS=${NUM_GPUS:-1}
BATCH=${BATCH:-4}
LR=${LR:-0.0002}
EPOCHS=${EPOCHS:-100}
EVAL_EVERY=${EVAL_EVERY:-5}

python main.py \
  --config-file "$CONFIG" \
  --num-gpus "$NUM_GPUS" \
  --seed "$SEED" \
  SOLVER.IMS_PER_BATCH "$BATCH" \
  SOLVER.BASE_LR "$LR" \
  SOLVER.MAX_EPOCH "$EPOCHS" \
  TEST.EVAL_EPOCH_PERIOD "$EVAL_EVERY"
