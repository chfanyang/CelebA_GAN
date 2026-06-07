# CelebA Inpainting — Final Metrics Summary

## Center Mask

| Mode | Full L1 | Full PSNR | **Hole L1** | **Hole PSNR** | Notes |
|------|---------|-----------|-------------|---------------|-------|
| l1 | 0.0075 | 30.2285 dB | **0.0530** | **21.7092 dB** | L1 baseline |
| gan | 0.0077 | 30.0296 dB | **0.0547** | **21.5102 dB** | L1 + PatchGAN (WGAN-GP) |

## Random Box Mask

| Mode | Full L1 | Full PSNR | **Hole L1** | **Hole PSNR** | Notes |
|------|---------|-----------|-------------|---------------|-------|
| l1 | 0.0089 | 28.7790 dB | **0.0634** | **20.2596 dB** | L1 baseline |
| gan | 0.0085 | 28.8917 dB | **0.0603** | **20.3724 dB** | L1 + PatchGAN (WGAN-GP) |

## Notes

- **Hole metrics** (hole_l1, hole_psnr) are the primary evaluation — full image metrics are diluted by known regions.
- GAN models typically achieve better visual realism despite potentially worse L1 scores.
- Center mask is harder than random_box because it consistently occludes facial structure.
