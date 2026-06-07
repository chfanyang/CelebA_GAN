"""Training loops for L1 baseline and L1+GAN inpainting."""

import csv
import os

import torch
import torch.optim as optim
from tqdm import tqdm

from .masks import generate_mask
from .losses import (
    compute_gradient_penalty,
    critic_loss,
    generator_loss_wgan,
    wasserstein_distance,
    masked_l1_loss,
)
from .metrics import compute_metrics, AverageMeter
from .visualize import save_sample_grid, plot_loss_curve
from .utils import denormalize, ensure_dir


# Default mask sizes for different image dimensions
# CelebA recommended: 128→48, 64→24
DEFAULT_MASK_SIZES = {32: 14, 64: 24, 128: 48}


def _get_inpainted(masked_image: torch.Tensor, mask: torch.Tensor,
                   predicted: torch.Tensor) -> torch.Tensor:
    """Combine masked image with predicted missing region.

    completed = masked_image * mask + predicted * (1 - mask)
    """
    return masked_image * mask + predicted * (1 - mask)


class BaseTrainer:
    """Common functionality for L1 and GAN trainers."""

    def __init__(self, generator, dataloader, val_dataloader,
                 output_dir, device, image_size, mask_type, mask_size,
                 lr, sample_interval, save_interval, lambda_l1=100):
        self.generator = generator.to(device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = output_dir
        self.device = device
        self.image_size = image_size
        self.mask_type = mask_type
        self.lambda_l1 = lambda_l1

        if mask_size is None:
            self.mask_size = DEFAULT_MASK_SIZES.get(image_size, image_size // 2)
        else:
            self.mask_size = mask_size

        self.sample_interval = sample_interval
        self.save_interval = save_interval

        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self.sample_dir = os.path.join(output_dir, "samples")
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.sample_dir)

        self.metrics_path = os.path.join(output_dir, "metrics.csv")
        self._init_metrics_csv()

        # Best checkpoint tracking: prefer hole_psnr (higher is better)
        self.best_psnr = -float("inf")
        self.best_l1 = float("inf")
        self.best_epoch = 0

        # Fixed images for consistent sample visualization
        self.fixed_images = None
        self.fixed_mask = None

    def _init_metrics_csv(self):
        """Initialize the metrics CSV file with header."""
        # Will be written per epoch; header set on first write
        self._csv_header_written = False

    def _save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save model checkpoint."""
        suffix = "final" if is_final else f"epoch_{epoch}"
        path = os.path.join(self.checkpoint_dir, f"generator_{suffix}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.generator.state_dict(),
        }, path)

    def _save_best_checkpoint(self, epoch: int):
        """Save best model checkpoint based on validation metric.

        Subclasses may override to save additional models (e.g., discriminator).
        """
        path = os.path.join(self.checkpoint_dir, "generator_best.pth")
        torch.save({
            "epoch": epoch,
            "best_psnr": self.best_psnr,
            "best_l1": self.best_l1,
            "model_state_dict": self.generator.state_dict(),
        }, path)

    def _save_samples(self, epoch: int):
        """Generate and save sample inpainting results."""
        self.generator.eval()

        if self.fixed_images is None:
            # Get a batch of validation images
            try:
                images, _ = next(iter(self.val_dataloader))
            except StopIteration:
                return
            images = images.to(self.device)
            n = min(images.size(0), 8)
            self.fixed_images = images[:n]
            mask = generate_mask(
                n, self.image_size, self.image_size,
                self.mask_type, self.mask_size, self.device
            )
            self.fixed_mask = mask

        images = self.fixed_images
        mask = self.fixed_mask
        masked = images * mask

        with torch.no_grad():
            predicted = self.generator(masked, mask)
            completed = _get_inpainted(masked, mask, predicted)

        images_dict = {
            "Original": images,
            "Mask": mask.repeat(1, images.size(1), 1, 1),
            "Masked": masked,
            "Prediction": predicted,
            "Completed": completed,
        }

        save_path = os.path.join(self.sample_dir, f"epoch_{epoch:03d}.png")
        save_sample_grid(images_dict, save_path, n_samples=images.size(0))
        self.generator.train()

    def _validate(self):
        """Compute validation metrics on a batch (full-image and hole-only)."""
        self.generator.eval()
        total_metrics = {}
        n_batches = 0

        with torch.no_grad():
            for images, _ in self.val_dataloader:
                images = images.to(self.device)
                batch_size = images.size(0)
                mask = generate_mask(
                    batch_size, self.image_size, self.image_size,
                    self.mask_type, self.mask_size, self.device
                )
                masked = images * mask
                predicted = self.generator(masked, mask)
                completed = _get_inpainted(masked, mask, predicted)
                metrics = compute_metrics(completed, images, mask=mask)
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0.0
                    total_metrics[k] += v * batch_size
                n_batches += 1
                if n_batches >= 10:  # Limit validation to 10 batches
                    break

        total_samples = n_batches * self.dataloader.batch_size
        for k in total_metrics:
            total_metrics[k] /= max(total_samples, 1)

        self.generator.train()
        return total_metrics

    def _write_metrics_row(self, epoch: int, loss_dict: dict, val_metrics: dict):
        """Write a row to the metrics CSV, marking best epoch."""
        row = {"epoch": epoch}
        row.update(loss_dict)
        row.update(val_metrics)

        # Determine if this is the best epoch
        is_best = False
        if "hole_psnr" in val_metrics:
            if val_metrics["hole_psnr"] > self.best_psnr:
                self.best_psnr = val_metrics["hole_psnr"]
                self.best_epoch = epoch
                is_best = True
        elif "hole_l1" in val_metrics:
            if val_metrics["hole_l1"] < self.best_l1:
                self.best_l1 = val_metrics["hole_l1"]
                self.best_epoch = epoch
                is_best = True

        row["is_best"] = is_best
        row["best_epoch"] = self.best_epoch

        if is_best:
            self._save_best_checkpoint(epoch)

        if not self._csv_header_written:
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            self._csv_header_written = True

        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)


class L1Trainer(BaseTrainer):
    """Trainer for L1-only autoencoder inpainting.

    Loss is computed only in the hole region (mask == 0) to prevent
    known-region pixels from diluting the training signal.
    """

    def __init__(self, generator, dataloader, val_dataloader,
                 output_dir, device, image_size, mask_type, mask_size,
                 lr, sample_interval, save_interval, lambda_l1=100):
        super().__init__(generator, dataloader, val_dataloader,
                         output_dir, device, image_size, mask_type, mask_size,
                         lr, sample_interval, save_interval, lambda_l1)
        self.optimizer = optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.9, 0.999)
        )

    def train(self, epochs: int):
        """Run L1 training loop."""
        self.generator.train()

        for epoch in range(1, epochs + 1):
            loss_meter = AverageMeter()
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{epochs} [L1]")

            for images, _ in pbar:
                images = images.to(self.device)
                batch_size = images.size(0)

                # Generate masks
                mask = generate_mask(
                    batch_size, self.image_size, self.image_size,
                    self.mask_type, self.mask_size, self.device
                )
                masked = images * mask

                # Forward
                predicted = self.generator(masked, mask)
                completed = _get_inpainted(masked, mask, predicted)

                # Masked L1 loss: only penalize the hole region
                loss = masked_l1_loss(predicted, images, mask)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_meter.update(loss.item(), batch_size)
                pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

            # End of epoch
            val_metrics = self._validate()
            self._write_metrics_row(epoch, {"loss": loss_meter.avg}, val_metrics)

            tqdm.write(
                f"Epoch {epoch}/{epochs} | L1 Loss: {loss_meter.avg:.4f} | "
                f"Hole PSNR: {val_metrics.get('hole_psnr', val_metrics.get('full_psnr', 0)):.2f} dB"
            )

            if epoch % self.save_interval == 0:
                self._save_checkpoint(epoch)
                self._save_samples(epoch)

        # Final save
        self._save_checkpoint(epochs, is_final=True)
        self._save_samples(epochs)

        # Plot loss curve
        plot_loss_curve(self.metrics_path,
                        os.path.join(self.output_dir, "loss_curve.png"))


class GANTrainer(BaseTrainer):
    """WGAN-GP Trainer with TTUR (Two Time-scale Update Rule).

    Key improvements over standard GAN:
    - Wasserstein loss + gradient penalty (stable, mode-collapse resistant)
    - TTUR: critic learning rate > generator learning rate
    - n_critic: multiple critic updates per generator update
    - No BCE, no sigmoid, no label smoothing needed
    """

    def __init__(self, generator, discriminator, dataloader, val_dataloader,
                 output_dir, device, image_size, mask_type, mask_size,
                 lr, sample_interval, save_interval, lambda_l1=100,
                 d_lr_mult=4.0, n_critic=5, lambda_gp=10.0):
        super().__init__(generator, dataloader, val_dataloader,
                         output_dir, device, image_size, mask_type, mask_size,
                         lr, sample_interval, save_interval, lambda_l1)
        self.discriminator = discriminator.to(device)

        # TTUR: critic learns faster than generator
        g_lr = lr
        d_lr = lr * d_lr_mult
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), lr=g_lr, betas=(0.0, 0.9)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=d_lr, betas=(0.0, 0.9)
        )
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp

        print(f"[INFO] WGAN-GP: G_lr={g_lr:.0e}, D_lr={d_lr:.0e}, "
              f"n_critic={n_critic}, lambda_gp={lambda_gp}")

    def _save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save both generator and discriminator checkpoints."""
        suffix = "final" if is_final else f"epoch_{epoch}"
        g_path = os.path.join(self.checkpoint_dir, f"generator_{suffix}.pth")
        d_path = os.path.join(self.checkpoint_dir, f"discriminator_{suffix}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.generator.state_dict(),
        }, g_path)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.discriminator.state_dict(),
        }, d_path)

    def _save_best_checkpoint(self, epoch: int):
        """Save best generator and discriminator checkpoints."""
        super()._save_best_checkpoint(epoch)
        d_path = os.path.join(self.checkpoint_dir, "discriminator_best.pth")
        torch.save({
            "epoch": epoch,
            "best_psnr": self.best_psnr,
            "best_l1": self.best_l1,
            "model_state_dict": self.discriminator.state_dict(),
        }, d_path)

    def train(self, epochs: int):
        """Run WGAN-GP training loop with TTUR and mask-conditioned critic."""
        self.generator.train()
        self.discriminator.train()

        for epoch in range(1, epochs + 1):
            g_loss_meter = AverageMeter()
            d_loss_meter = AverageMeter()
            l1_loss_meter = AverageMeter()
            w_dist_meter = AverageMeter()

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{epochs} [WGAN-GP]")

            for images, _ in pbar:
                images = images.to(self.device)
                batch_size = images.size(0)

                # Generate masks
                mask = generate_mask(
                    batch_size, self.image_size, self.image_size,
                    self.mask_type, self.mask_size, self.device
                )
                masked = images * mask

                # Prepare mask-conditioned real input for critic
                real_input = torch.cat([images, mask], dim=1)

                # ============================================================
                # Critic (Discriminator) updates — n_critic times per G update
                # ============================================================
                d_loss_total = 0.0
                for _ in range(self.n_critic):
                    with torch.no_grad():
                        predicted = self.generator(masked, mask)
                        completed = _get_inpainted(masked, mask, predicted)
                        fake_input = torch.cat([completed, mask], dim=1)

                    d_real = self.discriminator(real_input)
                    d_fake = self.discriminator(fake_input.detach())

                    # Gradient penalty on mask-conditioned inputs
                    gp = compute_gradient_penalty(
                        self.discriminator, real_input, fake_input.detach(),
                        self.device, self.lambda_gp
                    )

                    # WGAN critic loss
                    d_loss = critic_loss(d_real, d_fake, gp)

                    self.d_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    d_loss_total += d_loss.item()

                d_loss_avg = d_loss_total / self.n_critic

                # ============================================================
                # Generator update — only once per batch
                # ============================================================
                predicted = self.generator(masked, mask)
                completed = _get_inpainted(masked, mask, predicted)

                # Mask-conditioned fake input for critic
                fake_input_for_g = torch.cat([completed, mask], dim=1)
                d_fake_for_g = self.discriminator(fake_input_for_g)

                # Generator loss: adversarial + masked L1
                g_loss = generator_loss_wgan(
                    predicted, images, mask, d_fake_for_g, self.lambda_l1
                )
                l1_val = masked_l1_loss(predicted, images, mask).item()

                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Wasserstein distance for monitoring
                w_dist = wasserstein_distance(d_real, d_fake.detach())

                g_loss_meter.update(g_loss.item(), batch_size)
                d_loss_meter.update(d_loss_avg, batch_size)
                l1_loss_meter.update(l1_val, batch_size)
                w_dist_meter.update(w_dist, batch_size)

                pbar.set_postfix({
                    "G": f"{g_loss_meter.avg:.2f}",
                    "D": f"{d_loss_meter.avg:.2f}",
                    "L1": f"{l1_loss_meter.avg:.4f}",
                    "W": f"{w_dist_meter.avg:.3f}",
                })

            # End of epoch
            val_metrics = self._validate()
            loss_dict = {
                "g_loss": g_loss_meter.avg,
                "d_loss": d_loss_meter.avg,
                "l1_loss": l1_loss_meter.avg,
                "wasserstein_d": w_dist_meter.avg,
            }
            self._write_metrics_row(epoch, loss_dict, val_metrics)

            tqdm.write(
                f"Epoch {epoch}/{epochs} | G: {g_loss_meter.avg:.2f} | "
                f"D: {d_loss_meter.avg:.2f} | L1: {l1_loss_meter.avg:.4f} | "
                f"W-dist: {w_dist_meter.avg:.3f} | "
                f"Hole PSNR: {val_metrics.get('hole_psnr', val_metrics.get('full_psnr', 0)):.2f} dB"
            )

            if epoch % self.save_interval == 0:
                self._save_checkpoint(epoch)
                self._save_samples(epoch)

        # Final save
        self._save_checkpoint(epochs, is_final=True)
        self._save_samples(epochs)

        # Plot loss curve
        plot_loss_curve(self.metrics_path,
                        os.path.join(self.output_dir, "loss_curve.png"))
