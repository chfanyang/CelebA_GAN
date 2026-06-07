#!/usr/bin/env bash
#
# run_celeba_experiments.sh
#
# Launch CelebA inpainting experiments in parallel on 4 GPUs.
# Each experiment writes logs/*.log and outputs to outputs/celeba_kaggle/.
#
# Usage:
#   bash scripts/run_celeba_experiments.sh
#
# To stop all:
#   pkill -f "train.py --dataset celeba_kaggle"
#

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

DATA_ROOT="./data/celeba"
MAX_SAMPLES=20000
OUTPUT_DIR="./outputs"

# ---------------------------------------------------------------------------
# Helper: launch a training run on a specific GPU
# ---------------------------------------------------------------------------
launch() {
    local gpu="$1"
    local mode="$2"
    local mask_type="$3"
    local epochs="$4"
    local batch_size="$5"
    local image_size="$6"
    local extra_args="${7:-}"

    local log_name="celeba_${mode}_${mask_type}.log"
    local log_path="logs/${log_name}"

    # Skip if final checkpoint already exists
    local ckpt_path="${OUTPUT_DIR}/celeba_kaggle/${mode}/${mask_type}/checkpoints/generator_final.pth"
    if [ -f "$ckpt_path" ]; then
        echo "[SKIP] Checkpoint already exists: ${ckpt_path}"
        echo "       Remove it manually to re-run."
        return 0
    fi

    echo "[LAUNCH] GPU=${gpu} mode=${mode} mask=${mask_type} epochs=${epochs} bs=${batch_size} img=${image_size} → ${log_path}"

    CUDA_VISIBLE_DEVICES="${gpu}" nohup python train.py \
        --dataset celeba_kaggle \
        --mode "${mode}" \
        --mask_type "${mask_type}" \
        --epochs "${epochs}" \
        --batch_size "${batch_size}" \
        --image_size "${image_size}" \
        --data_root "${DATA_ROOT}" \
        --max_samples "${MAX_SAMPLES}" \
        ${extra_args} \
        --output_dir "${OUTPUT_DIR}" \
        > "${log_path}" 2>&1 &

    echo "       PID=$!"
}

# ---------------------------------------------------------------------------
# Launch all 4 experiments in parallel
# ---------------------------------------------------------------------------
echo "============================================"
echo "Launching CelebA Experiments"
echo "============================================"
echo "Data root:   ${DATA_ROOT}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "============================================"
echo ""

# GPU 0: L1 center
launch 0 "l1" "center" 20 16 128 ""

# GPU 1: GAN center
launch 1 "gan" "center" 40 16 128 "--lambda_l1 100"

# GPU 2: L1 random_box
launch 2 "l1" "random_box" 20 16 128 ""

# GPU 3: GAN random_box
launch 3 "gan" "random_box" 40 16 128 "--lambda_l1 100"

echo ""
echo "============================================"
echo "All experiments launched."
echo "Monitor with:  tail -f logs/*.log"
echo "Check GPUs:    nvidia-smi"
echo "============================================"
