"""WGAN-GP loss functions for stable GAN training.

Wasserstein GAN with Gradient Penalty:
- Replaces BCE loss with Earth Mover's distance
- Gradient penalty enforces 1-Lipschitz constraint on critic
- No label smoothing needed; no sigmoid on critic output

Masked L1 loss:
- Computes L1 only in the hole (missing) region
- Prevents known-region pixels from diluting the loss
"""

import torch
import torch.nn as nn
import torch.autograd as autograd


def masked_l1_loss(predicted: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute L1 loss only in the hole (missing) region.

    Args:
        predicted: generator output [B, C, H, W] in [-1, 1]
        target: ground truth image [B, C, H, W] in [-1, 1]
        mask: binary mask [B, 1, H, W], 1=known, 0=missing (hole)
        eps: small value to avoid division by zero

    Returns:
        scalar L1 loss averaged over hole pixels only
    """
    hole = 1.0 - mask  # 1=missing, 0=known
    loss = torch.abs(predicted - target) * hole
    return loss.sum() / (hole.sum() * target.size(1) + eps)


def compute_gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """Compute WGAN-GP gradient penalty.

    Penalizes deviation of critic gradient norm from 1 at random
    interpolations between real and fake samples.

    Args:
        critic: discriminator/critic model
        real: real images [B, C, H, W]
        fake: generated images [B, C, H, W] (detached)
        device: torch device
        lambda_gp: gradient penalty weight (default 10.0)

    Returns:
        gradient penalty loss (scalar)
    """
    batch_size = real.size(0)
    # Random interpolation coefficient per sample
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    epsilon = epsilon.expand_as(real)

    # Interpolated samples
    interpolated = epsilon * real + (1.0 - epsilon) * fake
    interpolated.requires_grad_(True)

    # Critic output on interpolated samples
    d_interpolated = critic(interpolated)

    # Gradients of critic output w.r.t. interpolated samples
    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Compute gradient penalty: (||grad||_2 - 1)^2
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1.0) ** 2).mean()

    return lambda_gp * gradient_penalty


def critic_loss(
    d_real: torch.Tensor,
    d_fake: torch.Tensor,
    gradient_penalty: torch.Tensor,
) -> torch.Tensor:
    """WGAN critic loss: maximize D(real) - D(fake).

    D_loss = mean(D(fake)) - mean(D(real)) + gradient_penalty

    The critic tries to push real scores high and fake scores low.

    Args:
        d_real: critic output for real images
        d_fake: critic output for generated images (detached)
        gradient_penalty: computed gradient penalty

    Returns:
        critic loss (scalar)
    """
    return d_fake.mean() - d_real.mean() + gradient_penalty


def generator_loss_wgan(
    predicted: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor,
    d_fake: torch.Tensor,
    lambda_l1: float,
) -> torch.Tensor:
    """WGAN generator loss: minimize -D(G(z)) + lambda_l1 * masked_L1.

    G_loss = -mean(D(fake)) + lambda_l1 * masked_l1_loss(predicted, original, mask)

    L1 loss is computed only in the hole region (where mask == 0). This
    prevents known-region pixels from dominating the reconstruction loss.

    Args:
        predicted: raw generator output [B, C, H, W] (before blending)
        original: original ground truth image [B, C, H, W]
        mask: binary mask [B, 1, H, W], 1=known, 0=missing
        d_fake: critic output for completed image (with mask channel)
        lambda_l1: weight for L1 reconstruction loss

    Returns:
        total generator loss (scalar)
    """
    l1_loss = masked_l1_loss(predicted, original, mask)
    # WGAN generator: negative of critic score on fake
    wasserstein_loss = -d_fake.mean()
    return lambda_l1 * l1_loss + wasserstein_loss


def wasserstein_distance(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Estimate Wasserstein distance: D(real) - D(fake).

    Higher value = better critic (more separation).
    Approaching 0 = generator is matching real distribution.

    This is used for monitoring; not used in gradient computation.

    Args:
        d_real: critic output for real images
        d_fake: critic output for generated images

    Returns:
        estimated Wasserstein distance (scalar, detached)
    """
    with torch.no_grad():
        return (d_real.mean() - d_fake.mean()).item()
