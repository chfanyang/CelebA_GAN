#!/bin/bash
# ============================================================================
# Run evaluation and L1 vs GAN comparison for all experiments
# Updated: checkpoint paths include mask_type
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

GPU=4

run_eval() {
    local name=$1 dataset=$2 mode=$3 mask_type=$4
    shift 4
    local ckpt="./outputs/${dataset}/${mode}/${mask_type}/checkpoints/generator_final.pth"
    local log_file="$LOG_DIR/eval2_${name}.log"
    echo -e "${BLUE}[EVAL] $name${NC}"
    CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
        --dataset "$dataset" --mode "$mode" --mask_type "$mask_type" \
        --checkpoint "$ckpt" "$@" > "$log_file" 2>&1
    [ $? -eq 0 ] && echo -e "${GREEN}[EVAL] DONE: $name${NC}" || echo -e "\033[0;31m[EVAL] FAILED: $name${NC}"
}

run_comparison() {
    local name=$1 dataset=$2 mask_type=$3
    shift 3
    local l1_ckpt="./outputs/${dataset}/l1/${mask_type}/checkpoints/generator_final.pth"
    local gan_ckpt="./outputs/${dataset}/gan/${mask_type}/checkpoints/generator_final.pth"
    local log_file="$LOG_DIR/cmp2_${name}.log"
    echo -e "${BLUE}[CMP] $name${NC}"
    CUDA_VISIBLE_DEVICES=$GPU python make_comparison.py \
        --dataset "$dataset" --mask_type "$mask_type" \
        --l1_checkpoint "$l1_ckpt" \
        --gan_checkpoint "$gan_ckpt" \
        "$@" > "$log_file" 2>&1
    [ $? -eq 0 ] && echo -e "${GREEN}[CMP] DONE: $name${NC}" || echo -e "\033[0;31m[CMP] FAILED: $name${NC}"
}

echo "============================================================================"
echo "  Post-Training Evaluation & Comparison"
echo "  Start time: $(date)"
echo "============================================================================"

for dataset in fashion_mnist cifar10; do
    echo -e "\n${YELLOW}=== $dataset Evaluations ===${NC}"
    for mode in l1 gan; do
        for mask in center random_box; do
            run_eval "${dataset}_${mode}_${mask}" "$dataset" "$mode" "$mask"
        done
    done
done

echo -e "\n${YELLOW}=== Places2 Evaluations ===${NC}"
for mode in l1 gan; do
    for mask in center random_box; do
        run_eval "places2_${mode}_${mask}" places2 "$mode" "$mask" \
            --data_root ./data/places2_subset --image_size 128
    done
done

echo -e "\n${YELLOW}=== L1 vs GAN Comparisons ===${NC}"
for dataset in fashion_mnist cifar10; do
    for mask in center random_box; do
        run_comparison "${dataset}_${mask}" "$dataset" "$mask"
    done
done

for mask in center random_box; do
    run_comparison "places2_${mask}" places2 "$mask" \
        --data_root ./data/places2_subset --image_size 128
done

echo -e "\n${GREEN}============================================================================"
echo "  ALL DONE at $(date)"
echo "============================================================================${NC}"
