#!/bin/bash
# ============================================================================
# Run ALL GAN inpainting experiments (v2 — mask_type separated in output dir)
# 12 experiments: 3 datasets × 2 modes × 2 mask types
# Uses GPUs 4 and 5 in parallel
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

run_one() {
    local gpu=$1 name=$2; shift 2
    local log="$LOG_DIR/${name}.log"
    echo -e "${BLUE}[GPU $gpu] $name${NC}"
    echo "CMD: CUDA_VISIBLE_DEVICES=$gpu python train.py $@" > "$log"
    CUDA_VISIBLE_DEVICES=$gpu python train.py "$@" >> "$log" 2>&1
    [ $? -eq 0 ] && echo "SUCCESS" >> "$log" && echo -e "${GREEN}[GPU $gpu] DONE: $name${NC}" || { echo "FAILED" >> "$log"; echo -e "\033[0;31m[GPU $gpu] FAILED: $name${NC}"; }
}

echo "============================================================================"
echo "  ALL GAN Inpainting Experiments — 12 total on GPUs 4 & 5"
echo "  Start: $(date)"
echo "============================================================================"

# ── Batch 1: Fashion-MNIST L1 ──────────────────────────────────────────────
echo -e "\n${YELLOW}=== Batch 1: Fashion-MNIST L1 ===${NC}"
run_one 4 fm_l1_ctr  --dataset fashion_mnist --mode l1 --mask_type center     --epochs 20 --batch_size 128 --image_size 32 &
run_one 5 fm_l1_rb   --dataset fashion_mnist --mode l1 --mask_type random_box --epochs 20 --batch_size 128 --image_size 32 &
wait

# ── Batch 2: CIFAR-10 L1 ───────────────────────────────────────────────────
echo -e "\n${YELLOW}=== Batch 2: CIFAR-10 L1 ===${NC}"
run_one 4 c10_l1_ctr --dataset cifar10 --mode l1 --mask_type center     --epochs 30 --batch_size 128 --image_size 32 &
run_one 5 c10_l1_rb  --dataset cifar10 --mode l1 --mask_type random_box --epochs 30 --batch_size 128 --image_size 32 &
wait

# ── Batch 3: Fashion-MNIST GAN ─────────────────────────────────────────────
echo -e "\n${YELLOW}=== Batch 3: Fashion-MNIST GAN ===${NC}"
run_one 4 fm_gan_ctr --dataset fashion_mnist --mode gan --mask_type center     --epochs 30 --batch_size 128 --image_size 32 --lambda_l1 100 &
run_one 5 fm_gan_rb  --dataset fashion_mnist --mode gan --mask_type random_box --epochs 30 --batch_size 128 --image_size 32 --lambda_l1 100 &
wait

# ── Batch 4: CIFAR-10 GAN ──────────────────────────────────────────────────
echo -e "\n${YELLOW}=== Batch 4: CIFAR-10 GAN ===${NC}"
run_one 4 c10_gan_ctr --dataset cifar10 --mode gan --mask_type center     --epochs 50 --batch_size 128 --image_size 32 --lambda_l1 100 &
run_one 5 c10_gan_rb  --dataset cifar10 --mode gan --mask_type random_box --epochs 50 --batch_size 128 --image_size 32 --lambda_l1 100 &
wait

# ── Batch 5: Places2 L1 ────────────────────────────────────────────────────
echo -e "\n${YELLOW}=== Batch 5: Places2 L1 ===${NC}"
run_one 4 p2_l1_ctr  --dataset places2 --mode l1 --mask_type center     --epochs 20 --batch_size 16 --image_size 128 --data_root ./data/places2_subset &
run_one 5 p2_l1_rb   --dataset places2 --mode l1 --mask_type random_box --epochs 20 --batch_size 16 --image_size 128 --data_root ./data/places2_subset &
wait

# ── Batch 6: Places2 GAN ───────────────────────────────────────────────────
echo -e "\n${YELLOW}=== Batch 6: Places2 GAN ===${NC}"
run_one 4 p2_gan_ctr --dataset places2 --mode gan --mask_type center     --epochs 40 --batch_size 16 --image_size 128 --data_root ./data/places2_subset --lambda_l1 100 &
run_one 5 p2_gan_rb  --dataset places2 --mode gan --mask_type random_box --epochs 40 --batch_size 16 --image_size 128 --data_root ./data/places2_subset --lambda_l1 100 &
wait

echo -e "\n${GREEN}============================================================================"
echo "  ALL 12 EXPERIMENTS COMPLETE — $(date)"
echo "============================================================================${NC}"

# Summary
echo -e "\n${BLUE}Summary:${NC}"
for log in "$LOG_DIR"/*.log; do
    name=$(basename "$log" .log)
    if grep -q "SUCCESS" "$log" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $name"
    else
        echo -e "  \033[0;31m✗${NC} $name"
    fi
done
