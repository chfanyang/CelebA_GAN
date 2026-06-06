"""Evaluation metrics for image inpainting."""

import math
import torch

from .utils import denormalize


def compute_metrics(predicted: torch.Tensor,
                    original: torch.Tensor) -> dict:
    """Compute L1, MSE, and PSNR between predicted and original images.

    Images are denormalized from [-1, 1] to [0, 1] before computing metrics.

    Args:
        predicted: predicted/completed image tensor [B, C, H, W]
        original: ground truth image tensor [B, C, H, W]

    Returns:
        dict with keys "l1", "mse", "psnr"
    """
    pred = denormalize(predicted)
    orig = denormalize(original)

    l1 = torch.nn.functional.l1_loss(pred, orig).item()
    mse = torch.nn.functional.mse_loss(pred, orig).item()
    psnr = 20 * math.log10(1.0) - 10 * math.log10(max(mse, 1e-10))

    return {"l1": l1, "mse": mse, "psnr": psnr}


class AverageMeter:
    """Track running averages of scalar values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)
