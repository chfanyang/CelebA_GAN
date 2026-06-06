#!/bin/bash
# ============================================================================
# Run all GAN inpainting experiments on GPUs 4 and 5
# 12 experiments total: 3 datasets × 2 modes × 2 mask types
# Runs 2 experiments in parallel (one per GPU)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_SCRIPT="$SCRIPT_DIR/train.py"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

run_exp() {
    local gpu=$1
    local name=$2
    shift 2
    local log_file="$LOG_DIR/${name}.log"
    echo -e "${BLUE}[GPU $gpu] Starting: $name${NC}"
    echo "Command: CUDA_VISIBLE_DEVICES=$gpu python $TRAIN_SCRIPT $@" > "$log_file"
    CUDA_VISIBLE_DEVICES=$gpu python $TRAIN_SCRIPT "$@" >> "$log_file" 2>&1
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[GPU $gpu] DONE: $name${NC}"
        echo "SUCCESS" >> "$log_file"
    else
        echo -e "\033[0;31m[GPU $gpu] FAILED (exit=$exit_code): $name${NC}"
        echo "FAILED (exit=$exit_code)" >> "$log_file"
    fi
    return $exit_code
}

echo "============================================================================"
echo "  GAN Inpainting Experiments — 12 total, 2 at a time on GPUs 4 & 5"
echo "  Start time: $(date)"
echo "============================================================================"

# ============================================================================
# Batch 1: Fashion-MNIST L1 (both mask types, 20 epochs each)
# ============================================================================
echo -e "\n${YELLOW}=== Batch 1/6: Fashion-MNIST L1 (20 epochs each) ===${NC}"
run_exp 4 "fashion_mnist_l1_center" \
    --dataset fashion_mnist --mode l1 --mask_type center \
    --epochs 20 --batch_size 128 --image_size 32 &
PID1=$!

run_exp 5 "fashion_mnist_l1_random_box" \
    --dataset fashion_mnist --mode l1 --mask_type random_box \
    --epochs 20 --batch_size 128 --image_size 32 &
PID2=$!

wait $PID1 || echo "ERROR: fashion_mnist_l1_center failed"
wait $PID2 || echo "ERROR: fashion_mnist_l1_random_box failed"

# ============================================================================
# Batch 2: CIFAR-10 L1 (both mask types, 30 epochs each)
# ============================================================================
echo -e "\n${YELLOW}=== Batch 2/6: CIFAR-10 L1 (30 epochs each) ===${NC}"
run_exp 4 "cifar10_l1_center" \
    --dataset cifar10 --mode l1 --mask_type center \
    --epochs 30 --batch_size 128 --image_size 32 &
PID1=$!

run_exp 5 "cifar10_l1_random_box" \
    --dataset cifar10 --mode l1 --mask_type random_box \
    --epochs 30 --batch_size 128 --image_size 32 &
PID2=$!

wait $PID1 || echo "ERROR: cifar10_l1_center failed"
wait $PID2 || echo "ERROR: cifar10_l1_random_box failed"

# ============================================================================
# Batch 3: Fashion-MNIST GAN (both mask types, 30 epochs each)
# ============================================================================
echo -e "\n${YELLOW}=== Batch 3/6: Fashion-MNIST GAN (30 epochs each) ===${NC}"
run_exp 4 "fashion_mnist_gan_center" \
    --dataset fashion_mnist --mode gan --mask_type center \
    --epochs 30 --batch_size 128 --image_size 32 --lambda_l1 100 &
PID1=$!

run_exp 5 "fashion_mnist_gan_random_box" \
    --dataset fashion_mnist --mode gan --mask_type random_box \
    --epochs 30 --batch_size 128 --image_size 32 --lambda_l1 100 &
PID2=$!

wait $PID1 || echo "ERROR: fashion_mnist_gan_center failed"
wait $PID2 || echo "ERROR: fashion_mnist_gan_random_box failed"

# ============================================================================
# Batch 4: CIFAR-10 GAN (both mask types, 50 epochs each)
# ============================================================================
echo -e "\n${YELLOW}=== Batch 4/6: CIFAR-10 GAN (50 epochs each) ===${NC}"
run_exp 4 "cifar10_gan_center" \
    --dataset cifar10 --mode gan --mask_type center \
    --epochs 50 --batch_size 128 --image_size 32 --lambda_l1 100 &
PID1=$!

run_exp 5 "cifar10_gan_random_box" \
    --dataset cifar10 --mode gan --mask_type random_box \
    --epochs 50 --batch_size 128 --image_size 32 --lambda_l1 100 &
PID2=$!

wait $PID1 || echo "ERROR: cifar10_gan_center failed"
wait $PID2 || echo "ERROR: cifar10_gan_random_box failed"

# ============================================================================
# Batch 5: Places2 L1 (both mask types, 20 epochs each)
# ============================================================================
echo -e "\n${YELLOW}=== Batch 5/6: Places2 L1 (20 epochs each) ===${NC}"
run_exp 4 "places2_l1_center" \
    --dataset places2 --mode l1 --mask_type center \
    --epochs 20 --batch_size 16 --image_size 128 \
    --data_root ./data/places2_subset &
PID1=$!

run_exp 5 "places2_l1_random_box" \
    --dataset places2 --mode l1 --mask_type random_box \
    --epochs 20 --batch_size 16 --image_size 128 \
    --data_root ./data/places2_subset &
PID2=$!

wait $PID1 || echo "ERROR: places2_l1_center failed"
wait $PID2 || echo "ERROR: places2_l1_random_box failed"

# ============================================================================
# Batch 6: Places2 GAN (both mask types, 40 epochs each)
# ============================================================================
echo -e "\n${YELLOW}=== Batch 6/6: Places2 GAN (40 epochs each) ===${NC}"
run_exp 4 "places2_gan_center" \
    --dataset places2 --mode gan --mask_type center \
    --epochs 40 --batch_size 16 --image_size 128 \
    --data_root ./data/places2_subset --lambda_l1 100 &
PID1=$!

run_exp 5 "places2_gan_random_box" \
    --dataset places2 --mode gan --mask_type random_box \
    --epochs 40 --batch_size 16 --image_size 128 \
    --data_root ./data/places2_subset --lambda_l1 100 &
PID2=$!

wait $PID1 || echo "ERROR: places2_gan_center failed"
wait $PID2 || echo "ERROR: places2_gan_random_box failed"

# ============================================================================
# Summary
# ============================================================================
echo -e "\n${GREEN}============================================================================"
echo "  ALL EXPERIMENTS COMPLETED"
echo "  End time: $(date)"
echo "============================================================================${NC}"

# Print summary of results
echo -e "\n${BLUE}=== Experiment Outputs ===${NC}"
for exp in \
    fashion_mnist_l1_center fashion_mnist_l1_random_box \
    fashion_mnist_gan_center fashion_mnist_gan_random_box \
    cifar10_l1_center cifar10_l1_random_box \
    cifar10_gan_center cifar10_gan_random_box \
    places2_l1_center places2_l1_random_box \
    places2_gan_center places2_gan_random_box; do
    log="$LOG_DIR/${exp}.log"
    if grep -q "SUCCESS" "$log" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $exp"
    elif grep -q "FAILED" "$log" 2>/dev/null; then
        echo -e "  \033[0;31m✗${NC} $exp (see $log)"
    else
        echo -e "  ${YELLOW}?${NC} $exp (unknown status)"
    fi
done

echo -e "\nLogs: $LOG_DIR/"
echo "Done."
