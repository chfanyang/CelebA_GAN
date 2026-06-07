#!/usr/bin/env python3
"""Evaluate a trained inpainting model on the test set.

Usage:
    python evaluate.py --dataset fashion_mnist --mode gan \\
        --checkpoint ./outputs/fashion_mnist/gan/checkpoints/generator_final.pth

    python evaluate.py --dataset cifar10 --mode l1 \\
        --checkpoint ./outputs/cifar10/l1/checkpoints/generator_final.pth
"""

import argparse
import os
import csv

import torch
from tqdm import tqdm

from src.utils import get_device, set_seed, ensure_dir
from src.datasets import get_dataloader
from src.models import Generator
from src.masks import generate_mask
from src.metrics import compute_metrics, AverageMeter
from src.visualize import save_sample_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained inpainting model"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["celeba", "celeba_kaggle", "fashion_mnist", "cifar10", "places2"])
    parser.add_argument("--mode", type=str, required=True,
                        choices=["l1", "gan"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to generator checkpoint")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mask_type", type=str, default="center",
                        choices=["center", "random_box"])
    parser.add_argument("--mask_size", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # Default image size
    if args.image_size is None:
        args.image_size = 128 if args.dataset in ("celeba", "celeba_kaggle", "places2") else 32

    # Default mask size
    if args.mask_size is None:
        default_masks = {32: 14, 64: 24, 128: 48}
        args.mask_size = default_masks.get(args.image_size, args.image_size // 2)

    # Image channels
    image_channels = 1 if args.dataset == "fashion_mnist" else 3

    # Load model
    generator = Generator(image_channels=image_channels,
                          image_size=args.image_size).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=True)
    generator.load_state_dict(checkpoint["model_state_dict"])
    generator.eval()
    print(f"[INFO] Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Data
    test_loader = get_dataloader(
        name=args.dataset,
        root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        train=False,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )

    # Evaluate
    metric_keys = ["full_l1", "full_mse", "full_psnr", "hole_l1", "hole_mse", "hole_psnr"]
    meters = {k: AverageMeter() for k in metric_keys}
    all_originals = []
    all_completed = []
    all_masked = []
    n_saved = 0

    print(f"[INFO] Evaluating on {len(test_loader)} batches...")
    with torch.no_grad():
        for images, _ in tqdm(test_loader):
            images = images.to(device)
            batch_size = images.size(0)

            mask = generate_mask(batch_size, args.image_size, args.image_size,
                                 args.mask_type, args.mask_size, device)
            masked = images * mask
            predicted = generator(masked, mask)
            completed = masked * mask + predicted * (1 - mask)

            metrics = compute_metrics(completed, images, mask=mask)
            for k, v in metrics.items():
                meters[k].update(v, batch_size)

            # Save some samples
            if n_saved < 8:
                all_originals.append(images[:1])
                all_completed.append(completed[:1])
                all_masked.append(masked[:1])
                n_saved += 1

    # Print results
    print("\n" + "=" * 60)
    print(f"Evaluation Results — {args.dataset} ({args.mode}, {args.mask_type})")
    print("-" * 60)
    for k, meter in meters.items():
        if "psnr" in k:
            print(f"  {k:>12s}: {meter.avg:.2f} dB")
        else:
            print(f"  {k:>12s}: {meter.avg:.6f}")
    print("=" * 60)

    # Save samples
    eval_dir = os.path.join(args.output_dir, args.dataset, args.mode, args.mask_type)
    ensure_dir(eval_dir)

    sample_images = torch.cat(all_originals + all_masked + all_completed, dim=0)
    samples_dict = {
        "Original": torch.cat(all_originals, dim=0),
        "Masked": torch.cat(all_masked, dim=0),
        "Completed": torch.cat(all_completed, dim=0),
    }
    save_sample_grid(samples_dict,
                     os.path.join(eval_dir, "evaluation_samples.png"),
                     n_samples=len(all_originals))

    # Save metrics
    metrics_path = os.path.join(eval_dir, "test_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for k, meter in meters.items():
            writer.writerow({"metric": k, "value": meter.avg})

    print(f"[INFO] Samples saved to {eval_dir}/evaluation_samples.png")
    print(f"[INFO] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
