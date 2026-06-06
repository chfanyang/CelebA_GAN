#!/bin/bash
# ============================================================================
# Recover all failed experiments:
#   2x CIFAR-10 L1 (failed due to download timeout)
#   4x Places2 (failed due to dataset __getitem__ bug, now fixed)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
OUTPUT="/dev/stdout"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================================================"
echo "  Recovery: CIFAR-10 L1 (2) + Places2 (4) = 6 experiments"
echo "  Start time: $(date)"
echo "============================================================================"

# ============================================================================
# Batch R1: CIFAR-10 L1 (30 epochs each)
# ============================================================================
echo -e "\n${YELLOW}=== Recovery Batch R1: CIFAR-10 L1 (30 epochs each) ===${NC}"
echo -e "${BLUE}[GPU 4] Starting: cifar10_l1_center (retry)${NC}"
CUDA_VISIBLE_DEVICES=4 python train.py \
    --dataset cifar10 --mode l1 --mask_type center \
    --epochs 30 --batch_size 128 --image_size 32 \
    > "$LOG_DIR/cifar10_l1_center.log" 2>&1 &
PID1=$!

echo -e "${BLUE}[GPU 5] Starting: cifar10_l1_random_box (retry)${NC}"
CUDA_VISIBLE_DEVICES=5 python train.py \
    --dataset cifar10 --mode l1 --mask_type random_box \
    --epochs 30 --batch_size 128 --image_size 32 \
    > "$LOG_DIR/cifar10_l1_random_box.log" 2>&1 &
PID2=$!

wait $PID1 && echo -e "${GREEN}[GPU 4] DONE: cifar10_l1_center${NC}" || echo "ERROR: cifar10_l1_center failed"
wait $PID2 && echo -e "${GREEN}[GPU 5] DONE: cifar10_l1_random_box${NC}" || echo "ERROR: cifar10_l1_random_box failed"

# ============================================================================
# Batch R2: Places2 L1 (20 epochs each)
# ============================================================================
echo -e "\n${YELLOW}=== Recovery Batch R2: Places2 L1 (20 epochs each) ===${NC}"
echo -e "${BLUE}[GPU 4] Starting: places2_l1_center (retry)${NC}"
CUDA_VISIBLE_DEVICES=4 python train.py \
    --dataset places2 --mode l1 --mask_type center \
    --epochs 20 --batch_size 16 --image_size 128 \
    --data_root ./data/places2_subset \
    > "$LOG_DIR/places2_l1_center.log" 2>&1 &
PID1=$!

echo -e "${BLUE}[GPU 5] Starting: places2_l1_random_box (retry)${NC}"
CUDA_VISIBLE_DEVICES=5 python train.py \
    --dataset places2 --mode l1 --mask_type random_box \
    --epochs 20 --batch_size 16 --image_size 128 \
    --data_root ./data/places2_subset \
    > "$LOG_DIR/places2_l1_random_box.log" 2>&1 &
PID2=$!

wait $PID1 && echo -e "${GREEN}[GPU 4] DONE: places2_l1_center${NC}" || echo "ERROR: places2_l1_center failed"
wait $PID2 && echo -e "${GREEN}[GPU 5] DONE: places2_l1_random_box${NC}" || echo "ERROR: places2_l1_random_box failed"

# ============================================================================
# Batch R3: Places2 GAN (40 epochs each)
# ============================================================================
echo -e "\n${YELLOW}=== Recovery Batch R3: Places2 GAN (40 epochs each) ===${NC}"
echo -e "${BLUE}[GPU 4] Starting: places2_gan_center (retry)${NC}"
CUDA_VISIBLE_DEVICES=4 python train.py \
    --dataset places2 --mode gan --mask_type center \
    --epochs 40 --batch_size 16 --image_size 128 \
    --data_root ./data/places2_subset --lambda_l1 100 \
    > "$LOG_DIR/places2_gan_center.log" 2>&1 &
PID1=$!

echo -e "${BLUE}[GPU 5] Starting: places2_gan_random_box (retry)${NC}"
CUDA_VISIBLE_DEVICES=5 python train.py \
    --dataset places2 --mode gan --mask_type random_box \
    --epochs 40 --batch_size 16 --image_size 128 \
    --data_root ./data/places2_subset --lambda_l1 100 \
    > "$LOG_DIR/places2_gan_random_box.log" 2>&1 &
PID2=$!

wait $PID1 && echo -e "${GREEN}[GPU 4] DONE: places2_gan_center${NC}" || echo "ERROR: places2_gan_center failed"
wait $PID2 && echo -e "${GREEN}[GPU 5] DONE: places2_gan_random_box${NC}" || echo "ERROR: places2_gan_random_box failed"

echo -e "\n${GREEN}============================================================================"
echo "  RECOVERY COMPLETE at $(date)"
echo "============================================================================${NC}"
