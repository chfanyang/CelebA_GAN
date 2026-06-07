#!/usr/bin/env bash
#
# post_training.sh
#
# Run after all 4 training experiments complete.
# Generates comparison figures, metrics summaries, and experiment report.
#
# Usage:
#   bash scripts/post_training.sh
#

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

OUTPUT_DIR="./outputs"
BASE="${OUTPUT_DIR}/celeba_kaggle"
DATA_ROOT="./data/celeba"
IMAGE_SIZE=128

echo "============================================"
echo "Post-Training: Comparisons & Summaries"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. Check that all 4 checkpoints exist
# ---------------------------------------------------------------------------
MISSING=""
for exp in "l1/center" "gan/center" "l1/random_box" "gan/random_box"; do
    ckpt="${BASE}/${exp}/checkpoints/generator_final.pth"
    if [ ! -f "$ckpt" ]; then
        echo "[MISSING] $ckpt"
        MISSING="${MISSING} ${exp}"
    else
        echo "[OK] $ckpt"
    fi
done

if [ -n "$MISSING" ]; then
    echo ""
    echo "[WARN] Some checkpoints are missing:${MISSING}"
    echo "       Proceeding with available checkpoints..."
fi

# ---------------------------------------------------------------------------
# 2. Generate comparison figures
# ---------------------------------------------------------------------------
echo ""
echo "--- Generating comparison figures ---"

if [ -f "${BASE}/l1/center/checkpoints/generator_final.pth" ] && \
   [ -f "${BASE}/gan/center/checkpoints/generator_final.pth" ]; then
    echo "[RUN] Center mask comparison..."
    python make_comparison.py \
        --dataset celeba_kaggle \
        --mask_type center \
        --image_size ${IMAGE_SIZE} \
        --data_root "${DATA_ROOT}" \
        --l1_checkpoint "${BASE}/l1/center/checkpoints/generator_final.pth" \
        --gan_checkpoint "${BASE}/gan/center/checkpoints/generator_final.pth" \
        --output_path "${BASE}/comparison_center.png"
    echo "[OK] ${BASE}/comparison_center.png"
else
    echo "[SKIP] Center mask — missing checkpoints"
fi

if [ -f "${BASE}/l1/random_box/checkpoints/generator_final.pth" ] && \
   [ -f "${BASE}/gan/random_box/checkpoints/generator_final.pth" ]; then
    echo "[RUN] Random box comparison..."
    python make_comparison.py \
        --dataset celeba_kaggle \
        --mask_type random_box \
        --image_size ${IMAGE_SIZE} \
        --data_root "${DATA_ROOT}" \
        --l1_checkpoint "${BASE}/l1/random_box/checkpoints/generator_final.pth" \
        --gan_checkpoint "${BASE}/gan/random_box/checkpoints/generator_final.pth" \
        --output_path "${BASE}/comparison_random_box.png"
    echo "[OK] ${BASE}/comparison_random_box.png"
else
    echo "[SKIP] Random box — missing checkpoints"
fi

# ---------------------------------------------------------------------------
# 3. Generate metrics summary
# ---------------------------------------------------------------------------
echo ""
echo "--- Generating metrics summary ---"
python scripts/generate_summaries.py --output_dir "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# 4. Generate experiment summary markdown
# ---------------------------------------------------------------------------
echo ""
echo "--- Generating experiment summary ---"
python scripts/generate_experiment_summary.py --output_dir "${OUTPUT_DIR}"

echo ""
echo "============================================"
echo "Post-training complete!"
echo "Output files:"
echo "  ${BASE}/comparison_center.png"
echo "  ${BASE}/comparison_random_box.png"
echo "  ${BASE}/final_metrics_summary.csv"
echo "  ${BASE}/final_metrics_summary.md"
echo "  ${BASE}/experiment_summary.md"
echo "============================================"
