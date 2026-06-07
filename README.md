# 基于 GAN 的 CelebA 人脸图像填充实验

Image Inpainting 图像填充任务：给定被遮挡的人脸图像，预测缺失区域，生成完整修复图像。

## 主实验

本项目主实验是 **CelebA 人脸图像填充**，对比两种方法：

| 方法 | 说明 |
|------|------|
| **L1 Baseline** | 仅使用 hole 区域 L1 损失的 Conv Encoder-Decoder |
| **L1 + GAN** | 同上生成器 + PatchGAN 判别器（WGAN-GP），L1 + 对抗损失 |

其他数据集（Fashion-MNIST, CIFAR-10, Places2）保留支持但非主实验。

## 项目结构

```
├── README.md
├── requirements.txt
├── train.py              # 训练入口
├── evaluate.py           # 评估脚本
├── make_comparison.py    # L1 vs GAN 对比图生成
├── src/
│   ├── datasets.py       # 数据加载 (CelebA, Fashion-MNIST, CIFAR-10, Places2)
│   ├── masks.py          # 遮挡生成 (center, random_box)
│   ├── models.py         # 生成器 (U-Net) + 判别器 (PatchGAN, mask-conditioned)
│   ├── losses.py         # GAN 损失函数 + masked L1 loss
│   ├── metrics.py        # 全图 + hole 区域指标 (L1, MSE, PSNR)
│   ├── trainer.py        # L1 / GAN 训练循环
│   ├── visualize.py      # 可视化工具
│   └── utils.py          # 设备、随机种子等工具函数
└── outputs/              # 训练输出 (自动创建)
```

## 环境要求

```bash
pip install -r requirements.txt
```

- PyTorch >= 2.0
- torchvision >= 0.15
- CUDA (可选，支持 CPU 训练)

## 数据集

### CelebA（主实验）

CelebA 数据集通过 `torchvision.datasets.CelebA` 自动下载。

**如果自动下载失败**（Google Drive 配额限制），需要手动准备：

1. 从 https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html 下载 `img_align_celeba.zip`
2. 解压到 `./data/celeba/img_align_celeba/`
3. 下载 `list_eval_partition.txt` 和 `identity_CelebA.txt` 放到 `./data/celeba/`

预处理：
- CenterCrop 成正方形（原始图像 178×218）
- Resize 到 `--image_size`（默认 128）
- Normalize 到 [-1, 1]

### 其他数据集

Fashion-MNIST 和 CIFAR-10 自动下载。Places2 需要手动准备本地子集。

## 使用方法

### 主实验：CelebA 人脸填充

#### L1 基线

```bash
python train.py \
  --dataset celeba \
  --mode l1 \
  --mask_type center \
  --epochs 20 \
  --batch_size 16 \
  --image_size 128 \
  --data_root ./data \
  --max_samples 20000 \
  --output_dir ./outputs
```

#### L1 + GAN

```bash
python train.py \
  --dataset celeba \
  --mode gan \
  --mask_type center \
  --epochs 40 \
  --batch_size 16 \
  --image_size 128 \
  --lambda_l1 100 \
  --data_root ./data \
  --max_samples 20000 \
  --output_dir ./outputs
```

#### 显存不足时（64×64）

```bash
python train.py \
  --dataset celeba \
  --mode gan \
  --mask_type center \
  --epochs 40 \
  --batch_size 32 \
  --image_size 64 \
  --lambda_l1 100 \
  --data_root ./data \
  --max_samples 20000 \
  --output_dir ./outputs
```

### 其他数据集（保留支持）

<details>
<summary>Fashion-MNIST / CIFAR-10 / Places2 命令</summary>

#### Fashion-MNIST

```bash
# L1 基线
python train.py --dataset fashion_mnist --mode l1 --mask_type center \
    --epochs 20 --batch_size 128 --image_size 32

# L1 + GAN
python train.py --dataset fashion_mnist --mode gan --mask_type center \
    --epochs 30 --batch_size 128 --image_size 32 --lambda_l1 100
```

#### CIFAR-10

```bash
# L1 基线
python train.py --dataset cifar10 --mode l1 --mask_type center \
    --epochs 30 --batch_size 128 --image_size 32

# L1 + GAN
python train.py --dataset cifar10 --mode gan --mask_type center \
    --epochs 50 --batch_size 128 --image_size 32 --lambda_l1 100
```

#### Places2-Subset

```bash
# L1
python train.py --dataset places2 --mode l1 --mask_type center \
    --epochs 20 --batch_size 16 --image_size 128 \
    --data_root ./data/places2_subset --max_samples 5000

# GAN
python train.py --dataset places2 --mode gan --mask_type center \
    --epochs 40 --batch_size 16 --image_size 128 \
    --data_root ./data/places2_subset --max_samples 5000 --lambda_l1 100
```

</details>

### 评估

```bash
python evaluate.py --dataset celeba --mode gan \
  --checkpoint ./outputs/celeba/gan/center/checkpoints/generator_final.pth \
  --image_size 128
```

### 生成 L1 vs GAN 对比图

```bash
python make_comparison.py --dataset celeba \
  --l1_checkpoint ./outputs/celeba/l1/center/checkpoints/generator_final.pth \
  --gan_checkpoint ./outputs/celeba/gan/center/checkpoints/generator_final.pth \
  --image_size 128
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集: celeba / fashion_mnist / cifar10 / places2 | 必填 |
| `--mode` | 训练模式: l1 / gan | 必填 |
| `--mask_type` | 遮挡类型: center / random_box | center |
| `--epochs` | 训练轮数 | 按数据集自动设置 |
| `--batch_size` | 批次大小 | 按数据集自动设置 |
| `--image_size` | 图像尺寸 | CelebA/Places2: 128, 其他: 32 |
| `--lambda_l1` | L1 损失权重 (GAN 模式) | 100 |
| `--lr` | 学习率 | 2e-4 |
| `--data_root` | 数据根目录 | ./data |
| `--max_samples` | 限制训练样本数 | None |
| `--mask_size` | 遮挡方块边长 | 自动计算 (128→48, 64→24, 32→14) |
| `--output_dir` | 输出根目录 | ./outputs |
| `--seed` | 随机种子 | 42 |
| `--num_workers` | 数据加载线程数 | 4 |
| `--save_interval` | 每隔 N 个 epoch 保存 checkpoint | 5 |
| `--sample_interval` | 每隔 N 个 epoch 保存样例图 | 5 |

## Mask 说明

| 图像尺寸 | center mask 大小 |
|----------|-----------------|
| 128×128 | 48×48 |
| 64×64 | 24×24 |
| 32×32 | 14×14 |

Mask 约定：
- `mask = 1` → 已知区域
- `mask = 0` → 缺失区域（hole）
- `masked_image = image * mask`
- 生成器输入 = `concat([masked_image, mask])` → 4 通道 (RGB + mask)
- 判别器输入 = `concat([image, mask])` → 4 通道（mask-conditioned）

## 关键设计

### Masked L1 Loss

L1 损失**只在 hole 区域**计算，避免已知区域像素稀释训练信号：

```python
hole = 1 - mask
loss = |predicted - target| * hole  / (hole_pixels + eps)
```

### Mask-Conditioned Discriminator

判别器接收 `[image, mask]` 拼接输入，让判别器知道哪些区域是修复的，从而给出更有针对性的判断。

### 评估指标

| 指标 | 说明 |
|------|------|
| `full_l1 / full_mse / full_psnr` | 全图指标 |
| `hole_l1 / hole_mse / hole_psnr` | 仅 hole 区域指标（**更重要**） |

报告中主要使用 **hole_l1** 和 **hole_psnr**。

## 输出内容

每个实验在 `outputs/{dataset}/{mode}/{mask_type}/` 下保存：

```
checkpoints/          # 模型权重 (generator_epoch_N.pth, generator_final.pth)
samples/              # 样例图像 (epoch_NNN.png)
metrics.csv           # 训练指标 (loss, full_psnr, hole_psnr, ...)
loss_curve.png        # 损失曲线
```

样例图格式：`Original | Mask | Masked | Prediction | Completed`

对比图格式：`Original | Masked | L1 Completed | GAN Completed`

## 模型架构

- **生成器**: U-Net with skip connections，层数根据 image_size 自动调整
  - 128×128 → 5 层 encoder + 4 层 decoder
  - 64×64 → 4 层 encoder + 3 层 decoder
  - 32×32 → 3 层 encoder + 2 层 decoder
- **判别器**: PatchGAN (WGAN-GP)，无 BatchNorm（保证 gradient penalty 有效性）
  - 使用 TTUR (Two Time-scale Update Rule)：判别器学习率 = 生成器 × 3
  - n_critic = 3：每 3 次判别器更新后更新 1 次生成器
- **L1 基线**: 仅使用 masked L1 损失训练生成器
- **GAN 改进**: masked L1 + 对抗损失联合训练

## 报告分析要点

训练完成后，报告中可以从以下角度进行分析：

1. **L1 结果更平滑但容易模糊**：纯 L1 损失倾向于产生像素级的平均结果，修复区域过渡自然但缺乏纹理细节
2. **L1+GAN 视觉真实感更强**：对抗损失鼓励生成器产生逼真的纹理和细节，修复区域更自然
3. **GAN 训练更不稳定**：观察 loss 曲线中 G_loss 和 D_loss 的波动，Wasserstein distance 的变化趋势
4. **hole 区域指标 vs 全图指标**：hole_l1 和 hole_psnr 比 full image 指标更能反映修复质量，因为已知区域会稀释全图指标
5. **不同 mask 类型的差异**：center mask vs random_box mask 对修复难度的影响不同
