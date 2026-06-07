#!/usr/bin/env python3
"""
Extended metrics summary including lambda_l1=50 experiments.

Scans:
  outputs/celeba_kaggle/l1/{center,random_box}/metrics.csv
  outputs/celeba_kaggle/gan/{center,random_box}/metrics.csv
  outputs/celeba_kaggle/gan_lambda50/{center,random_box}/metrics.csv (if exists)

Produces:
  outputs/celeba_kaggle/final_metrics_summary_with_lambda50.csv
  outputs/celeba_kaggle/final_metrics_summary_with_lambda50.md

Usage:
  python scripts/generate_summaries_extended.py [--output_dir ./outputs]
"""

import argparse
import csv
import os
from pathlib import Path


def read_all_rows(path: str) -> list:
    """Read all rows from a metrics CSV."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def get_best_row(rows: list) -> dict:
    """Find the row with best hole_psnr (or lowest hole_l1)."""
    if not rows:
        return {}
    # Prefer hole_psnr
    if "hole_psnr" in rows[0]:
        best = max(rows, key=lambda r: float(r.get("hole_psnr", -1e9)))
    elif "hole_l1" in rows[0]:
        best = min(rows, key=lambda r: float(r.get("hole_l1", 1e9)))
    else:
        best = rows[-1]  # fallback: last row
    return best


def fmt(v, decimals=4):
    try:
        return f"{float(v):.{decimals}f}"
    except (ValueError, TypeError):
        return str(v) if v else "N/A"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    base = Path(args.output_dir) / "celeba_kaggle"

    # Define experiments to scan: (exp_name, mask_type, lambda_l1, label)
    experiments = [
        ("l1",           "center",     100, "L1 center λ=100"),
        ("gan",          "center",     100, "GAN center λ=100"),
        ("l1",           "random_box", 100, "L1 random_box λ=100"),
        ("gan",          "random_box", 100, "GAN random_box λ=100"),
        ("gan_lambda50", "center",      50, "GAN center λ=50"),
        ("gan_lambda50", "random_box",  50, "GAN random_box λ=50"),
    ]

    rows = []
    for exp_name, mask_type, lambda_l1, _label in experiments:
        csv_path = base / exp_name / mask_type / "metrics.csv"
        all_rows = read_all_rows(str(csv_path))
        if not all_rows:
            continue

        last = all_rows[-1]
        best = get_best_row(all_rows)

        row = {
            "dataset": "celeba_kaggle",
            "mask_type": mask_type,
            "exp_name": exp_name,
            "lambda_l1": lambda_l1,
            "final_full_l1": last.get("full_l1", "N/A"),
            "final_full_mse": last.get("full_mse", "N/A"),
            "final_full_psnr": last.get("full_psnr", "N/A"),
            "final_hole_l1": last.get("hole_l1", "N/A"),
            "final_hole_mse": last.get("hole_mse", "N/A"),
            "final_hole_psnr": last.get("hole_psnr", "N/A"),
            "best_epoch": best.get("epoch", "N/A"),
            "best_hole_psnr": best.get("hole_psnr", "N/A"),
            "best_hole_l1": best.get("hole_l1", "N/A"),
        }
        rows.append(row)

    if not rows:
        print("[ERROR] No metrics.csv files found.")
        return

    # ---- Save CSV ----
    csv_out = base / "final_metrics_summary_with_lambda50.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    all_fields = list(rows[0].keys())
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV saved to {csv_out}")

    # ---- Save Markdown ----
    md_out = base / "final_metrics_summary_with_lambda50.md"
    with open(md_out, "w") as f:
        f.write("# CelebA Inpainting — Final Metrics Summary (with λ=50)\n\n")

        for mask in ["center", "random_box"]:
            mask_label = "Center Mask" if mask == "center" else "Random Box Mask"
            f.write(f"## {mask_label}\n\n")
            f.write("| Exp | λ | Full PSNR | **Hole L1** | **Hole PSNR** | Best Epoch | Best Hole PSNR | Best Hole L1 |\n")
            f.write("|-----|---|-----------|-------------|---------------|------------|----------------|---------------|\n")
            for r in rows:
                if r["mask_type"] != mask:
                    continue
                f.write(
                    f"| {r['exp_name']} | {r['lambda_l1']} | "
                    f"{fmt(r['final_full_psnr'], 2)} dB | "
                    f"**{fmt(r['final_hole_l1'])}** | "
                    f"**{fmt(r['final_hole_psnr'], 2)} dB** | "
                    f"{r['best_epoch']} | "
                    f"{fmt(r['best_hole_psnr'], 2)} dB | "
                    f"{fmt(r['best_hole_l1'])} |\n"
                )
            f.write("\n")

        f.write("## Notes\n\n")
        f.write("- λ=100: stronger L1 constraint → more pixel-accurate, may be slightly blurry.\n")
        f.write("- λ=50: stronger adversarial influence → potentially more realistic textures, PSNR may be lower.\n")
        f.write("- Best epoch is selected by highest validation hole_psnr.\n")
        f.write("- **Hole metrics** are the primary evaluation — full image metrics are diluted by known regions.\n")

    print(f"[INFO] Markdown saved to {md_out}")


if __name__ == "__main__":
    main()
