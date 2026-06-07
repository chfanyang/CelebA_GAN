"""Evaluation metrics for image inpainting."""

import math
import torch

from .utils import denormalize


def compute_metrics(predicted: torch.Tensor,
                    original: torch.Tensor,
                    mask: torch.Tensor = None) -> dict:
    """Compute L1, MSE, and PSNR between predicted and original images.

    Images are denormalized from [-1, 1] to [0, 1] before computing metrics.

    Args:
        predicted: predicted/completed image tensor [B, C, H, W]
        original: ground truth image tensor [B, C, H, W]
        mask: optional binary mask [B, 1, H, W], 1=known, 0=missing.
              If provided, also computes hole-only metrics.

    Returns:
        dict with keys "full_l1", "full_mse", "full_psnr",
        and optionally "hole_l1", "hole_mse", "hole_psnr"
    """
    pred = denormalize(predicted)
    orig = denormalize(original)

    # Full image metrics
    full_l1 = torch.nn.functional.l1_loss(pred, orig).item()
    full_mse = torch.nn.functional.mse_loss(pred, orig).item()
    full_psnr = 20 * math.log10(1.0) - 10 * math.log10(max(full_mse, 1e-10))

    result = {"full_l1": full_l1, "full_mse": full_mse, "full_psnr": full_psnr}

    # Hole-only metrics
    if mask is not None:
        hole = 1.0 - mask  # 1=missing, 0=known
        hole_sum = hole.sum(dim=(1, 2, 3))  # [B] pixels per sample
        total_hole = hole_sum.sum() + 1e-8

        diff = torch.abs(pred - orig)
        hole_l1 = (diff * hole).sum() / (total_hole * pred.size(1) + 1e-8)

        sq_diff = (pred - orig) ** 2
        hole_mse = (sq_diff * hole).sum() / (total_hole * pred.size(1) + 1e-8)

        hole_psnr = 20 * math.log10(1.0) - 10 * math.log10(max(hole_mse.item(), 1e-10))

        result.update({
            "hole_l1": hole_l1.item(),
            "hole_mse": hole_mse.item(),
            "hole_psnr": hole_psnr,
        })

    return result


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
