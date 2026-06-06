"""Mask generation for image inpainting."""

import torch


def generate_center_mask(batch_size: int, h: int, w: int,
                         mask_size: int, device: torch.device) -> torch.Tensor:
    """Generate center square mask.

    Args:
        batch_size: number of masks
        h, w: image height and width
        mask_size: side length of the square hole
        device: torch device

    Returns:
        mask tensor of shape [B, 1, H, W], 1=known, 0=missing
    """
    mask = torch.ones(batch_size, 1, h, w, device=device)
    top = (h - mask_size) // 2
    left = (w - mask_size) // 2
    mask[:, :, top:top + mask_size, left:left + mask_size] = 0
    return mask


def generate_random_box_mask(batch_size: int, h: int, w: int,
                              mask_size: int, device: torch.device) -> torch.Tensor:
    """Generate random-position square mask.

    Args:
        batch_size: number of masks
        h, w: image height and width
        mask_size: side length of the square hole
        device: torch device

    Returns:
        mask tensor of shape [B, 1, H, W], 1=known, 0=missing
    """
    mask = torch.ones(batch_size, 1, h, w, device=device)
    for i in range(batch_size):
        top = torch.randint(0, h - mask_size + 1, (1,)).item()
        left = torch.randint(0, w - mask_size + 1, (1,)).item()
        mask[i, :, top:top + mask_size, left:left + mask_size] = 0
    return mask


def generate_mask(batch_size: int, h: int, w: int,
                  mask_type: str, mask_size: int,
                  device: torch.device) -> torch.Tensor:
    """Dispatch to the correct mask generator.

    Args:
        batch_size: number of masks
        h, w: image height and width
        mask_type: "center" or "random_box"
        mask_size: side length of the square hole
        device: torch device

    Returns:
        mask tensor of shape [B, 1, H, W], 1=known, 0=missing
    """
    if mask_type == "center":
        return generate_center_mask(batch_size, h, w, mask_size, device)
    elif mask_type == "random_box":
        return generate_random_box_mask(batch_size, h, w, mask_size, device)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}. "
                         f"Expected 'center' or 'random_box'.")
