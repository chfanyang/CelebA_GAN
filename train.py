#!/usr/bin/env python3
"""Train image inpainting models with L1 or L1+GAN loss.

Main experiment: CelebA face image inpainting.

Usage:
    # Kaggle CelebA L1 baseline (local files, no annotation needed)
    python train.py --dataset celeba_kaggle --mode l1 --mask_type center \\
        --epochs 20 --batch_size 16 --image_size 128 \\
        --data_root ./data/celeba --max_samples 20000

    # Kaggle CelebA GAN
    python train.py --dataset celeba_kaggle --mode gan --mask_type center \\
        --epochs 40 --batch_size 16 --image_size 128 \\
        --lambda_l1 100 --data_root ./data/celeba --max_samples 20000

    # Kaggle CelebA GAN (lower memory: 64x64)
    python train.py --dataset celeba_kaggle --mode gan --mask_type center \\
        --epochs 40 --batch_size 32 --image_size 64 \\
        --lambda_l1 100 --data_root ./data/celeba --max_samples 20000

    # CelebA (torchvision) L1 baseline
    python train.py --dataset celeba --mode l1 --mask_type center \\
        --epochs 20 --batch_size 16 --image_size 128 \\
        --data_root ./data --max_samples 20000

    # CelebA (torchvision) GAN
    python train.py --dataset celeba --mode gan --mask_type center \\
        --epochs 40 --batch_size 16 --image_size 128 \\
        --lambda_l1 100 --data_root ./data --max_samples 20000

    # Fashion-MNIST L1 baseline
    python train.py --dataset fashion_mnist --mode l1 --mask_type center \\
        --epochs 20 --batch_size 128 --image_size 32

    # CIFAR-10 GAN
    python train.py --dataset cifar10 --mode gan --mask_type center \\
        --epochs 50 --batch_size 128 --image_size 32 --lambda_l1 100
"""

import argparse
import math
import os

from src.utils import get_device, set_seed, ensure_dir
from src.datasets import get_dataloader
from src.models import Generator, Discriminator
from src.trainer import L1Trainer, GANTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train image inpainting models (L1 baseline / L1+GAN)"
    )
    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["celeba", "celeba_kaggle", "fashion_mnist", "cifar10", "places2"],
                        help="Dataset name")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--image_size", type=int, default=None,
                        help="Image size (auto-selected if not set)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max training samples (for Places2 subset)")

    # Model / Mode
    parser.add_argument("--mode", type=str, required=True,
                        choices=["l1", "gan"],
                        help="Training mode: l1 (baseline) or gan (L1+GAN)")

    # Mask
    parser.add_argument("--mask_type", type=str, default="center",
                        choices=["center", "random_box"],
                        help="Mask type")
    parser.add_argument("--mask_size", type=int, default=None,
                        help="Mask square side length (auto-computed if None)")

    # Training
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (default depends on dataset+mode)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (default depends on dataset)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lambda_l1", type=float, default=10.0,
                        help="L1 loss weight in GAN mode (lower = more GAN influence)")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output root directory")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--sample_interval", type=int, default=5,
                        help="Save sample images every N epochs")

    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Data loading workers")

    return parser.parse_args()


def auto_defaults(args):
    """Set default values based on dataset and mode where not specified."""
    # Image size defaults
    if args.image_size is None:
        if args.dataset in ("celeba", "celeba_kaggle", "places2"):
            args.image_size = 128
        else:
            args.image_size = 32

    # Batch size defaults
    if args.batch_size is None:
        if args.dataset in ("celeba", "celeba_kaggle"):
            args.batch_size = 16 if args.image_size >= 128 else 32
        elif args.dataset == "places2":
            args.batch_size = 16 if args.image_size >= 128 else 32
        else:
            args.batch_size = 128

    # Epochs defaults
    if args.epochs is None:
        defaults = {
            ("celeba", "l1"): 20,
            ("celeba", "gan"): 40,
            ("celeba_kaggle", "l1"): 20,
            ("celeba_kaggle", "gan"): 40,
            ("fashion_mnist", "l1"): 20,
            ("fashion_mnist", "gan"): 30,
            ("cifar10", "l1"): 30,
            ("cifar10", "gan"): 50,
            ("places2", "l1"): 20,
            ("places2", "gan"): 40,
        }
        args.epochs = defaults.get((args.dataset, args.mode), 30)

    return args


def main():
    args = parse_args()
    args = auto_defaults(args)

    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Dataset: {args.dataset}, Mode: {args.mode}, "
          f"Image size: {args.image_size}, Epochs: {args.epochs}")

    # Output directory (includes mask_type to separate experiments)
    output_dir = os.path.join(
        args.output_dir, args.dataset, args.mode, args.mask_type
    )
    ensure_dir(output_dir)
    print(f"[INFO] Output directory: {output_dir}")

    # Data
    is_places2 = args.dataset == "places2"
    data_path = args.data_root if is_places2 else args.data_root
    if is_places2:
        # For places2, data_root points directly to the image folder
        data_path = args.data_root

    train_loader = get_dataloader(
        name=args.dataset,
        root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        train=True,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        name=args.dataset,
        root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        train=False,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )

    print(f"[INFO] Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}")

    # Determine image channels
    if args.dataset == "fashion_mnist":
        image_channels = 1
    else:
        image_channels = 3

    # Create model(s)
    generator = Generator(
        image_channels=image_channels,
        image_size=args.image_size,
    )
    print(f"[INFO] Generator: {sum(p.numel() for p in generator.parameters()):,} params")

    if args.mode == "gan":
        # Discriminator input: image + mask channel (mask-conditioned)
        discriminator = Discriminator(
            input_channels=image_channels + 1,
            image_size=args.image_size,
        )
        print(f"[INFO] Discriminator: {sum(p.numel() for p in discriminator.parameters()):,} params")
    else:
        discriminator = None

    # Create trainer and train
    if args.mode == "l1":
        trainer = L1Trainer(
            generator=generator,
            dataloader=train_loader,
            val_dataloader=val_loader,
            output_dir=output_dir,
            device=device,
            image_size=args.image_size,
            mask_type=args.mask_type,
            mask_size=args.mask_size,
            lr=args.lr,
            sample_interval=args.sample_interval,
            save_interval=args.save_interval,
        )
    else:
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            dataloader=train_loader,
            val_dataloader=val_loader,
            output_dir=output_dir,
            device=device,
            image_size=args.image_size,
            mask_type=args.mask_type,
            mask_size=args.mask_size,
            lr=args.lr,
            sample_interval=args.sample_interval,
            save_interval=args.save_interval,
            lambda_l1=args.lambda_l1,
            d_lr_mult=3.0,      # TTUR: critic learns 3x faster
            n_critic=3,          # 3 critic updates per generator update
            lambda_gp=10.0,      # Gradient penalty weight
        )

    print(f"[INFO] Starting training for {args.epochs} epochs...")
    trainer.train(args.epochs)

    print(f"[INFO] Training complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
