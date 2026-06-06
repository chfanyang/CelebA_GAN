#!/usr/bin/env python3
"""Generate comparison figures between L1 baseline and GAN models.

Usage:
    python make_comparison.py --dataset fashion_mnist \\
        --l1_checkpoint ./outputs/fashion_mnist/l1/checkpoints/generator_final.pth \\
        --gan_checkpoint ./outputs/fashion_mnist/gan/checkpoints/generator_final.pth

    python make_comparison.py --dataset cifar10 \\
        --l1_checkpoint ./outputs/cifar10/l1/checkpoints/generator_final.pth \\
        --gan_checkpoint ./outputs/cifar10/gan/checkpoints/generator_final.pth
"""

import argparse
import os

import torch
from tqdm import tqdm

from src.utils import get_device, set_seed, ensure_dir
from src.datasets import get_dataloader
from src.models import Generator
from src.masks import generate_mask
from src.visualize import make_comparison_figure


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate L1 vs GAN comparison figures"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["fashion_mnist", "cifar10", "places2"])
    parser.add_argument("--l1_checkpoint", type=str, required=True)
    parser.add_argument("--gan_checkpoint", type=str, required=True)
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


def load_generator(checkpoint_path, image_channels, image_size, device):
    """Load a generator from checkpoint."""
    generator = Generator(image_channels=image_channels,
                          image_size=image_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=True)
    generator.load_state_dict(checkpoint["model_state_dict"])
    generator.eval()
    return generator


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # Defaults
    if args.image_size is None:
        args.image_size = 128 if args.dataset == "places2" else 32
    if args.mask_size is None:
        default_masks = {32: 14, 64: 28, 128: 56}
        args.mask_size = default_masks.get(args.image_size, args.image_size // 2)

    image_channels = 1 if args.dataset == "fashion_mnist" else 3

    # Load models
    print("[INFO] Loading L1 model...")
    l1_generator = load_generator(
        args.l1_checkpoint, image_channels, args.image_size, device
    )
    print("[INFO] Loading GAN model...")
    gan_generator = load_generator(
        args.gan_checkpoint, image_channels, args.image_size, device
    )

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

    # Generate comparison
    n_samples = 8
    all_originals = []
    all_masked = []
    all_l1_completed = []
    all_gan_completed = []
    collected = 0

    print(f"[INFO] Generating comparison on {len(test_loader)} batches...")
    with torch.no_grad():
        for images, _ in test_loader:
            if collected >= n_samples:
                break

            images = images.to(device)
            batch_size = images.size(0)

            mask = generate_mask(batch_size, args.image_size, args.image_size,
                                 args.mask_type, args.mask_size, device)
            masked = images * mask

            l1_pred = l1_generator(masked, mask)
            l1_completed = masked * mask + l1_pred * (1 - mask)

            gan_pred = gan_generator(masked, mask)
            gan_completed = masked * mask + gan_pred * (1 - mask)

            # Collect samples
            for i in range(batch_size):
                if collected >= n_samples:
                    break
                all_originals.append(images[i:i+1])
                all_masked.append(masked[i:i+1])
                all_l1_completed.append(l1_completed[i:i+1])
                all_gan_completed.append(gan_completed[i:i+1])
                collected += 1

    # Create comparison figure
    comparison_dir = os.path.join(args.output_dir, args.dataset)
    ensure_dir(comparison_dir)

    save_path = os.path.join(
        comparison_dir, f"comparison_l1_vs_gan_{args.mask_type}.png"
    )
    make_comparison_figure(
        torch.cat(all_originals, dim=0),
        torch.cat(all_masked, dim=0),
        torch.cat(all_l1_completed, dim=0),
        torch.cat(all_gan_completed, dim=0),
        save_path,
        n_samples=n_samples,
    )

    print(f"[INFO] Comparison figure saved to {save_path}")


if __name__ == "__main__":
    main()
