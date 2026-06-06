#!/bin/bash
# ============================================================================
# Re-run failed CIFAR-10 L1 experiments (failed due to download timeout)
# Run after main script completes
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "============================================================================"
echo "  Recovery: CIFAR-10 L1 Experiments (failed due to download)"
echo "  Start time: $(date)"
echo "============================================================================"

# Run on GPU 4
echo -e "${BLUE}[GPU 4] Starting: cifar10_l1_center (retry)${NC}"
CUDA_VISIBLE_DEVICES=4 python train.py \
    --dataset cifar10 --mode l1 --mask_type center \
    --epochs 30 --batch_size 128 --image_size 32 \
    >> "$LOG_DIR/cifar10_l1_center.log" 2>&1 &
PID1=$!

# Run on GPU 5
echo -e "${BLUE}[GPU 5] Starting: cifar10_l1_random_box (retry)${NC}"
CUDA_VISIBLE_DEVICES=5 python train.py \
    --dataset cifar10 --mode l1 --mask_type random_box \
    --epochs 30 --batch_size 128 --image_size 32 \
    >> "$LOG_DIR/cifar10_l1_random_box.log" 2>&1 &
PID2=$!

wait $PID1 && echo -e "${GREEN}[GPU 4] DONE: cifar10_l1_center${NC}" || echo "ERROR: cifar10_l1_center failed"
wait $PID2 && echo -e "${GREEN}[GPU 5] DONE: cifar10_l1_random_box${NC}" || echo "ERROR: cifar10_l1_random_box failed"

echo -e "\n${GREEN}CIFAR-10 L1 recovery completed at $(date)${NC}"
