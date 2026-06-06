"""Generator and Discriminator models for image inpainting."""

import math

import torch
import torch.nn as nn


class Generator(nn.Module):
    """U-Net Generator for image inpainting with skip connections.

    Input: concatenation of [masked_image, mask] along channel dim.
    Output: predicted full image (Tanh activation, range [-1, 1]).

    Skip connections from encoder to decoder allow direct gradient flow
    and preserve fine spatial details from known regions.

    Number of down/up layers auto-adjusts based on image_size:
      32x32  -> 3 layers (bottleneck 4x4)
      64x64  -> 4 layers (bottleneck 4x4)
      128x128 -> 5 layers (bottleneck 4x4)
    """

    def __init__(self, image_channels: int = 3, image_size: int = 32):
        super().__init__()
        self.image_channels = image_channels
        self.input_channels = image_channels + 1  # +1 for mask
        self.image_size = image_size

        # Number of downsampling layers: log2(32)=5, minus 2 → 3 layers
        num_layers = int(math.log2(image_size)) - 2
        self.num_layers = num_layers

        # Channel progression: 64 → 128 → 256 → 512 (capped)
        base_ch = 64
        ch_list = [min(base_ch * (2 ** i), 512) for i in range(num_layers)]

        # --- Encoder (downsampling path) ---
        encoder_blocks = []
        in_ch = self.input_channels
        for out_ch in ch_list:
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            encoder_blocks.append(block)
            in_ch = out_ch
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        # --- Bottleneck ---
        bottleneck_ch = ch_list[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

        # --- Decoder (upsampling path with skip connections) ---
        decoder_blocks = []
        dec_ch_list = list(reversed(ch_list))  # e.g. [256, 128, 64] for 3 layers
        for i in range(num_layers - 1):
            in_ch = dec_ch_list[i]
            skip_ch = dec_ch_list[i + 1]  # from corresponding encoder level
            out_ch = dec_ch_list[i + 1]
            block = nn.ModuleDict({
                'up': nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                'norm': nn.InstanceNorm2d(out_ch),
                'act': nn.ReLU(inplace=True),
                # After concat with skip: out_ch + skip_ch → out_ch
                'conv': nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1),
                'conv_norm': nn.InstanceNorm2d(out_ch),
                'conv_act': nn.ReLU(inplace=True),
            })
            decoder_blocks.append(block)

        # Final upsampling block (no skip at the last level)
        self.final_up = nn.ConvTranspose2d(
            dec_ch_list[-1], image_channels, kernel_size=4, stride=2, padding=1
        )
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.tanh = nn.Tanh()

    def forward(self, masked_image: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with U-Net skip connections.

        Args:
            masked_image: [B, C, H, W] masked image (image * mask)
            mask: [B, 1, H, W] binary mask (1=known, 0=missing)

        Returns:
            predicted: [B, C, H, W] predicted full image
        """
        x = torch.cat([masked_image, mask], dim=1)

        # Encoder path — store intermediate features for skip connections
        skip_features = []
        for enc_block in self.encoder_blocks:
            x = enc_block(x)
            skip_features.append(x)

        # Bottleneck
        x = self.bottleneck(skip_features[-1])

        # Decoder path with skip connections
        # skip_features: [E1, E2, E3] (low→high resolution)
        # decoder uses: E2, E1 (reversed, excluding bottleneck)
        num_skips = len(skip_features)
        for i, dec_block in enumerate(self.decoder_blocks):
            x = dec_block['up'](x)
            x = dec_block['norm'](x)
            x = dec_block['act'](x)
            # Concatenate skip features from encoder (going backwards)
            skip = skip_features[num_skips - 2 - i]  # E2, then E1
            x = torch.cat([x, skip], dim=1)
            x = dec_block['conv'](x)
            x = dec_block['conv_norm'](x)
            x = dec_block['conv_act'](x)

        x = self.final_up(x)
        return self.tanh(x)


class Discriminator(nn.Module):
    """PatchGAN Critic for WGAN-GP.

    Input: an image (original or completed).
    Output: patch-level critic scores (real-valued, no Sigmoid).

    NO normalization layers (BatchNorm/InstanceNorm break the
    gradient penalty's per-sample Lipschitz constraint).

    Number of layers auto-adjusts to match the Generator depth.
    """

    def __init__(self, image_channels: int = 3, image_size: int = 32):
        super().__init__()
        self.image_channels = image_channels
        self.image_size = image_size

        num_layers = int(math.log2(image_size)) - 2

        base_ch = 64
        ch_list = [min(base_ch * (2 ** i), 512) for i in range(num_layers)]

        layers = []
        in_ch = image_channels
        for out_ch in ch_list:
            layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch

        # Final patch output layer (stride=1) — single critic score per patch
        layers.append(
            nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            image: [B, C, H, W] input image

        Returns:
            scores: [B, 1, h', w'] per-patch critic scores (unbounded real values)
        """
        return self.model(image)
