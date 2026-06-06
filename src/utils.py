"""Utility functions for the GAN inpainting project."""

import random
import os
from pathlib import Path

import numpy as np
import torch


def get_device():
    """Return the available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [-1, 1] to [0, 1]."""
    return tensor.clamp(-1, 1).add(1).mul(0.5)
