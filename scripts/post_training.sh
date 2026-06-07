#!/usr/bin/env bash
#
# post_training.sh
#
# Run after all training experiments complete.
# Generates comparison figures (final + best), metrics summaries, and experiment report.
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
# Helper: generate a comparison figure
# ---------------------------------------------------------------------------
make_comparison() {
    local mask="$1"
    local l1_ckpt="$2"
    local gan_ckpt="$3"
    local out_path="$4"
    local label="$5"

    if [ -f "$l1_ckpt" ] && [ -f "$gan_ckpt" ]; then
        echo "[RUN] ${label}..."
        python make_comparison.py \
            --dataset celeba_kaggle \
            --mask_type "${mask}" \
            --image_size ${IMAGE_SIZE} \
            --data_root "${DATA_ROOT}" \
            --l1_checkpoint "${l1_ckpt}" \
            --gan_checkpoint "${gan_ckpt}" \
            --output_path "${out_path}"
        echo "[OK] ${out_path}"
    else
        echo "[SKIP] ${label} — missing checkpoints"
    fi
}

# ---------------------------------------------------------------------------
# 1. Final checkpoint comparisons (lambda_l1=100)
# ---------------------------------------------------------------------------
echo ""
echo "--- Final checkpoint comparisons (lambda_l1=100) ---"

make_comparison "center" \
    "${BASE}/l1/center/checkpoints/generator_final.pth" \
    "${BASE}/gan/center/checkpoints/generator_final.pth" \
    "${BASE}/comparison_center.png" \
    "Center mask (final)"

make_comparison "random_box" \
    "${BASE}/l1/random_box/checkpoints/generator_final.pth" \
    "${BASE}/gan/random_box/checkpoints/generator_final.pth" \
    "${BASE}/comparison_random_box.png" \
    "Random box mask (final)"

# ---------------------------------------------------------------------------
# 2. Best checkpoint comparisons (lambda_l1=100)
# ---------------------------------------------------------------------------
echo ""
echo "--- Best checkpoint comparisons (lambda_l1=100) ---"

make_comparison "center" \
    "${BASE}/l1/center/checkpoints/generator_best.pth" \
    "${BASE}/gan/center/checkpoints/generator_best.pth" \
    "${BASE}/comparison_center_best.png" \
    "Center mask (best)"

make_comparison "random_box" \
    "${BASE}/l1/random_box/checkpoints/generator_best.pth" \
    "${BASE}/gan/random_box/checkpoints/generator_best.pth" \
    "${BASE}/comparison_random_box_best.png" \
    "Random box mask (best)"

# ---------------------------------------------------------------------------
# 3. Lambda50 comparisons (if available)
# ---------------------------------------------------------------------------
echo ""
echo "--- Lambda50 comparisons (if available) ---"

L50_BASE="${BASE}/gan_lambda50"

if [ -d "${L50_BASE}" ]; then
    make_comparison "center" \
        "${BASE}/l1/center/checkpoints/generator_best.pth" \
        "${L50_BASE}/center/checkpoints/generator_best.pth" \
        "${BASE}/comparison_center_lambda50.png" \
        "Center mask lambda50 (best)"

    make_comparison "random_box" \
        "${BASE}/l1/random_box/checkpoints/generator_best.pth" \
        "${L50_BASE}/random_box/checkpoints/generator_best.pth" \
        "${BASE}/comparison_random_box_lambda50.png" \
        "Random box lambda50 (best)"
else
    echo "[SKIP] Lambda50 experiments not found at ${L50_BASE}"
fi

# ---------------------------------------------------------------------------
# 4. Generate metrics summary (standard)
# ---------------------------------------------------------------------------
echo ""
echo "--- Generating metrics summary ---"
python scripts/generate_summaries.py --output_dir "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# 5. Generate extended metrics summary with lambda50
# ---------------------------------------------------------------------------
echo ""
echo "--- Generating extended summary (with lambda50) ---"
python scripts/generate_summaries_extended.py --output_dir "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# 6. Generate experiment summary markdown
# ---------------------------------------------------------------------------
echo ""
echo "--- Generating experiment summary ---"
python scripts/generate_experiment_summary.py --output_dir "${OUTPUT_DIR}"

echo ""
echo "============================================"
echo "Post-training complete!"
echo "Output files:"
echo "  ${BASE}/comparison_center.png"
echo "  ${BASE}/comparison_center_best.png"
echo "  ${BASE}/comparison_random_box.png"
echo "  ${BASE}/comparison_random_box_best.png"
echo "  ${BASE}/final_metrics_summary.csv"
echo "  ${BASE}/final_metrics_summary.md"
echo "  ${BASE}/final_metrics_summary_with_lambda50.csv"
echo "  ${BASE}/final_metrics_summary_with_lambda50.md"
echo "  ${BASE}/experiment_summary.md"
echo "============================================"
