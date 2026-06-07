#!/usr/bin/env bash
#
# run_lambda50_experiment.sh
#
# Optional ablation experiment: GAN with lambda_l1=50.
# Lower L1 weight → stronger adversarial influence → potentially more
# realistic textures, but PSNR may not improve.
#
# Uses --exp_name gan_lambda50 to avoid overwriting lambda_l1=100 results.
#
# Usage:
#   bash scripts/run_lambda50_experiment.sh
#

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

DATA_ROOT="./data/celeba"
MAX_SAMPLES=20000
OUTPUT_DIR="./outputs"
EXP_NAME="gan_lambda50"
LAMBDA_L1=50

echo "============================================"
echo "Launching lambda_l1=50 GAN Experiments"
echo "============================================"
echo "Exp name:    ${EXP_NAME}"
echo "lambda_l1:   ${LAMBDA_L1}"
echo "Output dir:  ${OUTPUT_DIR}/celeba_kaggle/${EXP_NAME}/"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# GPU 0: GAN center, lambda_l1=50
# ---------------------------------------------------------------------------
LOG0="logs/celeba_gan_center_lambda50.log"
CKPT0="${OUTPUT_DIR}/celeba_kaggle/${EXP_NAME}/center/checkpoints/generator_final.pth"

if [ -f "$CKPT0" ]; then
    echo "[SKIP] Center mask already done: ${CKPT0}"
else
    echo "[LAUNCH] GPU=0 center mask lambda_l1=50 → ${LOG0}"
    CUDA_VISIBLE_DEVICES=0 nohup python train.py \
        --dataset celeba_kaggle \
        --mode gan \
        --exp_name "${EXP_NAME}" \
        --mask_type center \
        --epochs 40 \
        --batch_size 16 \
        --image_size 128 \
        --data_root "${DATA_ROOT}" \
        --max_samples "${MAX_SAMPLES}" \
        --lambda_l1 "${LAMBDA_L1}" \
        --output_dir "${OUTPUT_DIR}" \
        > "${LOG0}" 2>&1 &
    echo "       PID=$!"
fi

# ---------------------------------------------------------------------------
# GPU 1: GAN random_box, lambda_l1=50
# ---------------------------------------------------------------------------
LOG1="logs/celeba_gan_random_box_lambda50.log"
CKPT1="${OUTPUT_DIR}/celeba_kaggle/${EXP_NAME}/random_box/checkpoints/generator_final.pth"

if [ -f "$CKPT1" ]; then
    echo "[SKIP] Random box already done: ${CKPT1}"
else
    echo "[LAUNCH] GPU=1 random_box lambda_l1=50 → ${LOG1}"
    CUDA_VISIBLE_DEVICES=1 nohup python train.py \
        --dataset celeba_kaggle \
        --mode gan \
        --exp_name "${EXP_NAME}" \
        --mask_type random_box \
        --epochs 40 \
        --batch_size 16 \
        --image_size 128 \
        --data_root "${DATA_ROOT}" \
        --max_samples "${MAX_SAMPLES}" \
        --lambda_l1 "${LAMBDA_L1}" \
        --output_dir "${OUTPUT_DIR}" \
        > "${LOG1}" 2>&1 &
    echo "       PID=$!"
fi

echo ""
echo "============================================"
echo "Monitor with:"
echo "  tail -f ${LOG0}"
echo "  tail -f ${LOG1}"
echo ""
echo "Output will be at:"
echo "  ${OUTPUT_DIR}/celeba_kaggle/${EXP_NAME}/center/"
echo "  ${OUTPUT_DIR}/celeba_kaggle/${EXP_NAME}/random_box/"
echo "============================================"
