# CelebA Inpainting — Final Metrics Summary (with λ=50)

## Center Mask

| Exp | λ | Full PSNR | **Hole L1** | **Hole PSNR** | Best Epoch | Best Hole PSNR | Best Hole L1 |
|-----|---|-----------|-------------|---------------|------------|----------------|---------------|
| l1 | 100 | 30.23 dB | **0.0530** | **21.71 dB** | 17 | 21.77 dB | 0.0527 |
| gan | 100 | 30.03 dB | **0.0547** | **21.51 dB** | 28 | 21.76 dB | 0.0537 |
| gan_lambda50 | 50 | 30.13 dB | **0.0536** | **21.61 dB** | 40 | 21.61 dB | 0.0536 |

## Random Box Mask

| Exp | λ | Full PSNR | **Hole L1** | **Hole PSNR** | Best Epoch | Best Hole PSNR | Best Hole L1 |
|-----|---|-----------|-------------|---------------|------------|----------------|---------------|
| l1 | 100 | 28.78 dB | **0.0634** | **20.26 dB** | 20 | 20.26 dB | 0.0634 |
| gan | 100 | 28.89 dB | **0.0603** | **20.37 dB** | 38 | 20.84 dB | 0.0583 |
| gan_lambda50 | 50 | 28.87 dB | **0.0604** | **20.35 dB** | 38 | 20.83 dB | 0.0581 |

## Notes

- λ=100: stronger L1 constraint → more pixel-accurate, may be slightly blurry.
- λ=50: stronger adversarial influence → potentially more realistic textures, PSNR may be lower.
- Best epoch is selected by highest validation hole_psnr.
- **Hole metrics** are the primary evaluation — full image metrics are diluted by known regions.
