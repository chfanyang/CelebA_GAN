"""Visualization utilities for inpainting results."""

import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils

from .utils import denormalize


def tensor_to_numpy_grid(tensor: torch.Tensor, nrow: int = 8) -> np.ndarray:
    """Convert a batch of image tensors to a numpy grid image.

    Args:
        tensor: [B, C, H, W] image tensor in [-1, 1]
        nrow: number of images per row

    Returns:
        numpy array of shape [H_grid, W_grid, 3] in [0, 1]
    """
    tensor = denormalize(tensor)
    grid = vutils.make_grid(tensor, nrow=nrow, padding=2, normalize=False)
    grid = grid.cpu().numpy().transpose(1, 2, 0)
    # Handle grayscale
    if grid.shape[2] == 1:
        grid = np.repeat(grid, 3, axis=2)
    return np.clip(grid, 0, 1)


def save_sample_grid(images: dict, save_path: str, n_samples: int = 8):
    """Save a grid of sample images.

    Args:
        images: dict mapping name -> tensor [B, C, H, W] or None
        save_path: file path to save
        n_samples: number of samples to show
    """
    # Filter out None values
    images = {k: v for k, v in images.items() if v is not None}
    if not images:
        return

    n_cols = len(images)
    fig, axes = plt.subplots(n_samples, n_cols,
                              figsize=(n_cols * 3, n_samples * 3))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for col, (name, tensor) in enumerate(images.items()):
        tensor_np = tensor_to_numpy_grid(tensor[:n_samples], nrow=1)
        for row in range(n_samples):
            ax = axes[row, col]
            # Extract single image patch from the grid
            if tensor.shape[1] == 1:  # grayscale
                single = denormalize(tensor[row:row+1])
                single = single.cpu().squeeze().numpy()
                ax.imshow(single, cmap="gray", vmin=0, vmax=1)
            else:
                single = denormalize(tensor[row:row+1])
                single = single.cpu().squeeze(0).permute(1, 2, 0).numpy()
                ax.imshow(np.clip(single, 0, 1))
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_curve(csv_path: str, save_path: str):
    """Plot loss curves from a metrics CSV file.

    Args:
        csv_path: path to metrics.csv
        save_path: path to save the plot
    """
    import csv

    if not os.path.exists(csv_path):
        return

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return

    epochs = [int(r["epoch"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax = axes[0]
    # Keys that are metrics (not losses) — exclude from loss plot
    metric_keys = {"epoch", "full_l1", "full_mse", "full_psnr",
                   "hole_l1", "hole_mse", "hole_psnr",
                   "l1", "mse", "psnr"}
    for key in rows[0].keys():
        if key not in metric_keys:
            vals = [float(r[key]) for r in rows]
            ax.plot(epochs, vals, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Metrics plot
    ax = axes[1]
    for key in ["full_l1", "full_mse", "full_psnr",
                "hole_l1", "hole_mse", "hole_psnr"]:
        if key in rows[0]:
            vals = [float(r[key]) for r in rows]
            ax.plot(epochs, vals, label=key.upper())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_comparison_figure(originals, masked, l1_completed, gan_completed,
                           save_path: str, n_samples: int = 8):
    """Generate a 4-column comparison: Original | Masked | L1 | GAN.

    Args:
        originals: tensor [B, C, H, W]
        masked: tensor [B, C, H, W]
        l1_completed: tensor [B, C, H, W]
        gan_completed: tensor [B, C, H, W]
        save_path: file path
        n_samples: number of samples
    """
    images = {
        "Original": originals,
        "Masked": masked,
        "L1 Completed": l1_completed,
        "GAN Completed": gan_completed,
    }
    save_sample_grid(images, save_path, n_samples)
