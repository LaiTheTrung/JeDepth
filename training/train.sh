#!/bin/bash
# FastFoundationStereo Training Script
#
# Usage:
#   bash train.sh                              # Train from scratch
#   bash train.sh --resume output/ckpt_epoch_010.pth  # Resume training
#   bash train.sh --pretrained path/to/model.pth      # Fine-tune from pretrained
#
# Requires: conda activate jedepth

set -e

cd "$(dirname "$0")"

python train.py \
    --cfg cfgs/ffstereo_custom.yaml \
    --output_dir output \
    --gpu 0 \
    --workers 4 \
    --seed 42 \
    --test_images test_images \
    "$@"
