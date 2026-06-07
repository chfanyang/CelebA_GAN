#!/usr/bin/env python3
"""
Generate experiment_summary.md from training results.

Usage:
  python scripts/generate_experiment_summary.py [--output_dir ./outputs]
"""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path


def read_metrics_csv(path: str) -> list:
    """Read all rows from a metrics CSV."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def get_final_metrics(path: str) -> dict:
    """Get the last row of a metrics CSV."""
    rows = read_metrics_csv(path)
    return rows[-1] if rows else {}


def fmt(v: str, decimals: int = 4) -> str:
    try:
        return f"{float(v):.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)


def count_checkpoints(checkpoint_dir: str) -> int:
    """Count generator epoch checkpoints."""
    if not os.path.isdir(checkpoint_dir):
        return 0
    return len([f for f in os.listdir(checkpoint_dir) if f.startswith("generator_epoch")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    base = Path(args.output_dir) / "celeba_kaggle"

    # Read results
    results = {}
    for mode in ["l1", "gan"]:
        for mask in ["center", "random_box"]:
            key = f"{mode}_{mask}"
            csv_path = base / mode / mask / "metrics.csv"
            metrics = get_final_metrics(str(csv_path))
            results[key] = {
                "csv_path": str(csv_path),
                "metrics": metrics,
                "checkpoints": count_checkpoints(
                    str(base / mode / mask / "checkpoints")
                ),
            }

    # Determine what's available
    has_center = (bool(results["l1_center"]["metrics"]) and
                  bool(results["gan_center"]["metrics"]))
    has_random = (bool(results["l1_random_box"]["metrics"]) and
                  bool(results["gan_random_box"]["metrics"]))

    out_path = base / "experiment_summary.md"
    with open(out_path, "w") as f:
        f.write("# CelebA 人脸图像填充实验总结\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # ---- 1. Experiment Setup ----
        f.write("## 1. 实验设置\n\n")
        f.write("| 参数 | 值 |\n")
        f.write("|------|----|\n")
        f.write("| Dataset | celeba_kaggle (Kaggle 本地版) |\n")
        f.write("| Image Size | 128 × 128 |\n")
        f.write("| Max Samples | 20,000 |\n")
        f.write("| Batch Size | 16 |\n")
        f.write("| L1 Epochs | 20 |\n")
        f.write("| GAN Epochs | 40 |\n")
        f.write("| λ_l1 (GAN) | 100 |\n")
        f.write("| Optimizer | Adam |\n")
        f.write("| GAN Type | WGAN-GP (TTUR, n_critic=3, D_lr=3×G_lr) |\n")
        f.write("| Mask Types | center (48×48), random_box (48×48) |\n")
        f.write("| GPU | NVIDIA A800 80GB × 4 |\n\n")

        # ---- 2. Model Architecture ----
        f.write("## 2. 模型架构\n\n")
        f.write("- **生成器**: U-Net with skip connections (5 encoder + 4 decoder layers)\n")
        f.write("  - 参数量: 22,530,883\n")
        f.write("  - 输入: [masked_image, mask] → 4 通道\n")
        f.write("- **判别器** (仅 GAN 模式): PatchGAN (WGAN-GP)\n")
        f.write("  - 参数量: 6,960,577\n")
        f.write("  - 输入: [image, mask] → 4 通道 (mask-conditioned)\n")
        f.write("  - 无 BatchNorm (保证 gradient penalty 有效性)\n\n")

        # ---- 3. Final Metrics ----
        f.write("## 3. 最终指标\n\n")

        if has_center:
            f.write("### Center Mask\n\n")
            f.write("| Model | Full L1 | Full PSNR | **Hole L1** | **Hole PSNR** |\n")
            f.write("|-------|---------|-----------|-------------|---------------|\n")
            for mode_label, mode_key in [("L1 Baseline", "l1"), ("L1 + GAN", "gan")]:
                m = results[f"{mode_key}_center"]["metrics"]
                if m:
                    f.write(f"| {mode_label} | {fmt(m.get('full_l1','N/A'))} | "
                            f"{fmt(m.get('full_psnr','N/A'),2)} dB | "
                            f"**{fmt(m.get('hole_l1','N/A'))}** | "
                            f"**{fmt(m.get('hole_psnr','N/A'),2)} dB** |\n")
            f.write("\n")

        if has_random:
            f.write("### Random Box Mask\n\n")
            f.write("| Model | Full L1 | Full PSNR | **Hole L1** | **Hole PSNR** |\n")
            f.write("|-------|---------|-----------|-------------|---------------|\n")
            for mode_label, mode_key in [("L1 Baseline", "l1"), ("L1 + GAN", "gan")]:
                m = results[f"{mode_key}_random_box"]["metrics"]
                if m:
                    f.write(f"| {mode_label} | {fmt(m.get('full_l1','N/A'))} | "
                            f"{fmt(m.get('full_psnr','N/A'),2)} dB | "
                            f"**{fmt(m.get('hole_l1','N/A'))}** | "
                            f"**{fmt(m.get('hole_psnr','N/A'),2)} dB** |\n")
            f.write("\n")

        # ---- 4. Analysis Draft ----
        f.write("## 4. 结果分析\n\n")

        f.write("### 4.1 L1 Baseline vs L1+GAN\n\n")
        f.write("- **L1 模型**结果通常更平滑，但容易模糊——纯 L1 损失倾向于产生像素级的平均结果，修复区域过渡自然但缺乏纹理细节。\n")
        f.write("- **L1+GAN 模型**视觉真实感更强——对抗损失鼓励生成器产生逼真的纹理和细节，修复区域更自然。\n")
        f.write("- GAN 训练更不稳定：观察 loss 曲线中 G_loss 和 D_loss 的波动，Wasserstein distance 的变化趋势。\n\n")

        f.write("### 4.2 Hole 指标 vs Full Image 指标\n\n")
        f.write("- `hole_l1` 和 `hole_psnr` 比 `full_l1` / `full_psnr` 更能反映修复质量。\n")
        f.write("- 原因：已知区域（未遮挡部分）会稀释全图指标——即使 hole 区域修复得很好，全图指标变化也不明显。\n")
        f.write("- **报告中主要使用 hole_l1 和 hole_psnr 作为评估指标。**\n\n")

        f.write("### 4.3 Center Mask vs Random Box Mask\n\n")
        f.write("- **Center mask** 更难修复：中心区域通常覆盖人脸关键结构（眼、鼻、嘴），修复难度高。\n")
        f.write("- **Random box mask** 相对容易：遮挡可能落在背景或非关键区域，修复难度较低。\n")
        f.write("- 不同 mask 类型对 GAN 的 advantage 影响不同——GAN 在 center mask 上的相对提升通常更显著。\n\n")

        # ---- 5. Generated Assets ----
        f.write("## 5. 生成产物\n\n")
        f.write("```\n")
        f.write("outputs/celeba_kaggle/\n")
        f.write("├── l1/\n")
        f.write("│   ├── center/\n")
        f.write("│   │   ├── checkpoints/generator_final.pth\n")
        f.write("│   │   ├── samples/epoch_*.png\n")
        f.write("│   │   ├── metrics.csv\n")
        f.write("│   │   └── loss_curve.png\n")
        f.write("│   └── random_box/\n")
        f.write("│       └── ...\n")
        f.write("├── gan/\n")
        f.write("│   ├── center/\n")
        f.write("│   │   ├── checkpoints/{generator,discriminator}_final.pth\n")
        f.write("│   │   ├── samples/epoch_*.png\n")
        f.write("│   │   ├── metrics.csv\n")
        f.write("│   │   └── loss_curve.png\n")
        f.write("│   └── random_box/\n")
        f.write("│       └── ...\n")
        f.write("├── comparison_center.png\n")
        f.write("├── comparison_random_box.png\n")
        f.write("├── final_metrics_summary.csv\n")
        f.write("├── final_metrics_summary.md\n")
        f.write("└── experiment_summary.md  ← 本文件\n")
        f.write("```\n\n")

        # ---- 6. Reproduction Commands ----
        f.write("## 6. 复现命令\n\n")
        f.write("### 训练\n\n")
        f.write("```bash\n")
        f.write("bash scripts/run_celeba_experiments.sh\n")
        f.write("```\n\n")
        f.write("### 生成对比图与总结\n\n")
        f.write("```bash\n")
        f.write("bash scripts/post_training.sh\n")
        f.write("```\n")

    print(f"[INFO] Experiment summary saved to {out_path}")


if __name__ == "__main__":
    main()
