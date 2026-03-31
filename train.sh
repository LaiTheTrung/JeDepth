#!/bin/bash
# ============================================================================
# train.sh - Script chạy huấn luyện mô hình IINet stereo depth estimation
#
# Sử dụng:
#   bash train.sh                    # Chạy mặc định
#   bash train.sh --resume <path>    # Resume từ checkpoint
# ============================================================================

set -e  # Dừng nếu có lỗi

# ── Cấu hình cơ bản ─────────────────────────────────────────────────────────
CFG="cfgs/iinet/iinet_custom.yaml"     # File config
OUTPUT_DIR="output"                     # Thư mục lưu kết quả
GPU=0                                   # GPU device ID
WORKERS=4                               # Số worker cho DataLoader
SEED=42                                 # Random seed

# ── Chạy huấn luyện ─────────────────────────────────────────────────────────
echo "============================================"
echo " IINet Stereo Depth Training"
echo " Config:     ${CFG}"
echo " Output:     ${OUTPUT_DIR}"
echo " GPU:        ${GPU}"
echo " Workers:    ${WORKERS}"
echo " Seed:       ${SEED}"
echo "============================================"

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --cfg ${CFG} \
    --output_dir ${OUTPUT_DIR} \
    --gpu 0 \
    --workers ${WORKERS} \
    --seed ${SEED} \
    "$@"

echo "Training finished!"
echo "Results saved to: ${OUTPUT_DIR}/$(basename ${CFG} .yaml)"
echo "TensorBoard: tensorboard --logdir ${OUTPUT_DIR}/$(basename ${CFG} .yaml)/tensorboard"
