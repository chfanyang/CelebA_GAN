#!/usr/bin/env python3
"""
Generate final metrics summary from experiment output directories.

Scans outputs/celeba_kaggle/{l1,gan}/{center,random_box}/metrics.csv
and produces:
  - outputs/celeba_kaggle/final_metrics_summary.csv
  - outputs/celeba_kaggle/final_metrics_summary.md

Usage:
  python scripts/generate_summaries.py [--output_dir ./outputs]
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def read_final_metrics(metrics_path: str) -> dict:
    """Read the last row of a metrics CSV file."""
    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    return rows[-1]  # Last epoch


def read_all_rows(metrics_path: str) -> list:
    """Read all rows from a metrics CSV file."""
    if not os.path.exists(metrics_path):
        return []
    with open(metrics_path, "r") as f:
        return list(csv.DictReader(f))


def get_best_metrics(rows: list) -> dict:
    """Find the best row by hole_psnr (higher is better) or hole_l1 (lower is better)."""
    if not rows:
        return {}
    if "hole_psnr" in rows[0]:
        best = max(rows, key=lambda r: float(r.get("hole_psnr", -1e9)))
    elif "hole_l1" in rows[0]:
        best = min(rows, key=lambda r: float(r.get("hole_l1", 1e9)))
    else:
        best = rows[-1]
    return {
        "best_epoch": best.get("epoch", "N/A"),
        "best_hole_psnr": best.get("hole_psnr", "N/A"),
        "best_hole_l1": best.get("hole_l1", "N/A"),
    }


def format_value(v: str) -> str:
    """Format a metric value to 4 decimal places if float."""
    try:
        return f"{float(v):.4f}"
    except (ValueError, TypeError):
        return str(v)


def main():
    parser = argparse.ArgumentParser(description="Generate final metrics summary")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    base = Path(args.output_dir) / "celeba_kaggle"

    experiments = [
        ("celeba_kaggle",  "center",     "l1",  base / "l1"  / "center"     / "metrics.csv"),
        ("celeba_kaggle",  "center",     "gan", base / "gan" / "center"     / "metrics.csv"),
        ("celeba_kaggle",  "random_box", "l1",  base / "l1"  / "random_box" / "metrics.csv"),
        ("celeba_kaggle",  "random_box", "gan", base / "gan" / "random_box" / "metrics.csv"),
    ]

    # Collect results
    rows = []
    found_any = False

    for dataset, mask_type, mode, csv_path in experiments:
        all_rows_list = read_all_rows(str(csv_path))
        if not all_rows_list:
            print(f"[WARN] Not found: {csv_path}")
            continue

        metrics = all_rows_list[-1]  # last epoch for final values
        best = get_best_metrics(all_rows_list)

        found_any = True
        row = {
            "dataset": dataset,
            "mask_type": mask_type,
            "mode": mode,
            "best_epoch": best.get("best_epoch", "N/A"),
            "best_hole_psnr": best.get("best_hole_psnr", "N/A"),
            "best_hole_l1": best.get("best_hole_l1", "N/A"),
        }

        # Extract key metrics (handle naming variances)
        metric_map = {
            "final_full_l1":  ["full_l1"],
            "final_full_mse": ["full_mse"],
            "final_full_psnr": ["full_psnr"],
            "final_hole_l1":  ["hole_l1"],
            "final_hole_mse": ["hole_mse"],
            "final_hole_psnr": ["hole_psnr"],
        }

        for out_key, candidates in metric_map.items():
            for c in candidates:
                if c in metrics:
                    row[out_key] = metrics[c]
                    break
            else:
                row[out_key] = "N/A"

        # Add loss info if available
        if "loss" in metrics:
            row["final_loss"] = metrics["loss"]
        elif "l1_loss" in metrics:
            row["final_l1_loss"] = metrics["l1_loss"]
        if "g_loss" in metrics:
            row["final_g_loss"] = metrics["g_loss"]
        if "d_loss" in metrics:
            row["final_d_loss"] = metrics["d_loss"]
        if "wasserstein_d" in metrics:
            row["final_w_dist"] = metrics["wasserstein_d"]

        rows.append(row)

    if not found_any:
        print("[ERROR] No metrics.csv files found. Make sure training has completed.")
        sys.exit(1)

    # ---- Save CSV ----
    csv_out = base / "final_metrics_summary.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    # Collect all field names from all rows
    all_fields_set = set()
    for r in rows:
        all_fields_set.update(r.keys())
    all_fields = sorted(all_fields_set)
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV summary saved to {csv_out}")

    # ---- Save Markdown ----
    md_out = base / "final_metrics_summary.md"
    with open(md_out, "w") as f:
        f.write("# CelebA Inpainting — Final Metrics Summary\n\n")

        f.write("## Center Mask\n\n")
        f.write("| Mode | Full L1 | Full PSNR | **Hole L1** | **Hole PSNR** | Notes |\n")
        f.write("|------|---------|-----------|-------------|---------------|-------|\n")
        for r in rows:
            if r["mask_type"] != "center":
                continue
            notes = ""
            if r["mode"] == "gan":
                notes = "L1 + PatchGAN (WGAN-GP)"
            else:
                notes = "L1 baseline"
            f.write(f"| {r['mode']} | {format_value(r['final_full_l1'])} | "
                    f"{format_value(r['final_full_psnr'])} dB | "
                    f"**{format_value(r['final_hole_l1'])}** | "
                    f"**{format_value(r['final_hole_psnr'])} dB** | "
                    f"{notes} |\n")

        f.write("\n## Random Box Mask\n\n")
        f.write("| Mode | Full L1 | Full PSNR | **Hole L1** | **Hole PSNR** | Notes |\n")
        f.write("|------|---------|-----------|-------------|---------------|-------|\n")
        for r in rows:
            if r["mask_type"] != "random_box":
                continue
            notes = ""
            if r["mode"] == "gan":
                notes = "L1 + PatchGAN (WGAN-GP)"
            else:
                notes = "L1 baseline"
            f.write(f"| {r['mode']} | {format_value(r['final_full_l1'])} | "
                    f"{format_value(r['final_full_psnr'])} dB | "
                    f"**{format_value(r['final_hole_l1'])}** | "
                    f"**{format_value(r['final_hole_psnr'])} dB** | "
                    f"{notes} |\n")

        f.write("\n## Notes\n\n")
        f.write("- **Hole metrics** (hole_l1, hole_psnr) are the primary evaluation — full image metrics are diluted by known regions.\n")
        f.write("- GAN models typically achieve better visual realism despite potentially worse L1 scores.\n")
        f.write("- Center mask is harder than random_box because it consistently occludes facial structure.\n")

    print(f"[INFO] Markdown summary saved to {md_out}")


if __name__ == "__main__":
    main()
