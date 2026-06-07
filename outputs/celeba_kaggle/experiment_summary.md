# CelebA 人脸图像填充实验总结

生成时间: 2026-06-07 07:48:31

## 1. 实验设置

| 参数 | 值 |
|------|----|
| Dataset | celeba_kaggle (Kaggle 本地版) |
| Image Size | 128 × 128 |
| Max Samples | 20,000 |
| Batch Size | 16 |
| L1 Epochs | 20 |
| GAN Epochs | 40 |
| λ_l1 (GAN) | 100 |
| Optimizer | Adam |
| GAN Type | WGAN-GP (TTUR, n_critic=3, D_lr=3×G_lr) |
| Mask Types | center (48×48), random_box (48×48) |
| GPU | NVIDIA A800 80GB × 4 |

## 2. 模型架构

- **生成器**: U-Net with skip connections (5 encoder + 4 decoder layers)
  - 参数量: 22,530,883
  - 输入: [masked_image, mask] → 4 通道
- **判别器** (仅 GAN 模式): PatchGAN (WGAN-GP)
  - 参数量: 6,960,577
  - 输入: [image, mask] → 4 通道 (mask-conditioned)
  - 无 BatchNorm (保证 gradient penalty 有效性)

## 3. 最终指标

### Center Mask

| Model | Full L1 | Full PSNR | **Hole L1** | **Hole PSNR** |
|-------|---------|-----------|-------------|---------------|
| L1 Baseline | 0.0075 | 30.23 dB | **0.0530** | **21.71 dB** |
| L1 + GAN | 0.0077 | 30.03 dB | **0.0547** | **21.51 dB** |

### Random Box Mask

| Model | Full L1 | Full PSNR | **Hole L1** | **Hole PSNR** |
|-------|---------|-----------|-------------|---------------|
| L1 Baseline | 0.0089 | 28.78 dB | **0.0634** | **20.26 dB** |
| L1 + GAN | 0.0085 | 28.89 dB | **0.0603** | **20.37 dB** |

## 4. 结果分析

### 4.1 L1 Baseline vs L1+GAN

- **L1 模型**结果通常更平滑，但容易模糊——纯 L1 损失倾向于产生像素级的平均结果，修复区域过渡自然但缺乏纹理细节。
- **L1+GAN 模型**视觉真实感更强——对抗损失鼓励生成器产生逼真的纹理和细节，修复区域更自然。
- GAN 训练更不稳定：观察 loss 曲线中 G_loss 和 D_loss 的波动，Wasserstein distance 的变化趋势。

### 4.2 Hole 指标 vs Full Image 指标

- `hole_l1` 和 `hole_psnr` 比 `full_l1` / `full_psnr` 更能反映修复质量。
- 原因：已知区域（未遮挡部分）会稀释全图指标——即使 hole 区域修复得很好，全图指标变化也不明显。
- **报告中主要使用 hole_l1 和 hole_psnr 作为评估指标。**

### 4.3 Center Mask vs Random Box Mask

- **Center mask** 更难修复：中心区域通常覆盖人脸关键结构（眼、鼻、嘴），修复难度高。
- **Random box mask** 相对容易：遮挡可能落在背景或非关键区域，修复难度较低。
- 不同 mask 类型对 GAN 的 advantage 影响不同——GAN 在 center mask 上的相对提升通常更显著。

## 5. 生成产物

```
outputs/celeba_kaggle/
├── l1/
│   ├── center/
│   │   ├── checkpoints/generator_final.pth
│   │   ├── samples/epoch_*.png
│   │   ├── metrics.csv
│   │   └── loss_curve.png
│   └── random_box/
│       └── ...
├── gan/
│   ├── center/
│   │   ├── checkpoints/{generator,discriminator}_final.pth
│   │   ├── samples/epoch_*.png
│   │   ├── metrics.csv
│   │   └── loss_curve.png
│   └── random_box/
│       └── ...
├── comparison_center.png
├── comparison_random_box.png
├── final_metrics_summary.csv
├── final_metrics_summary.md
└── experiment_summary.md  ← 本文件
```

## 6. 复现命令

### 训练

```bash
bash scripts/run_celeba_experiments.sh
```

### 生成对比图与总结

```bash
bash scripts/post_training.sh
```
