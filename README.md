# 基于 GAN 的跨数据集图像填充实验

Image Inpainting 图像填充任务：给定被遮挡的图像，预测缺失区域，生成完整修复图像。

## 项目结构

```
├── README.md
├── requirements.txt
├── train.py              # 训练入口
├── evaluate.py           # 评估脚本
├── make_comparison.py    # L1 vs GAN 对比图生成
├── src/
│   ├── datasets.py       # 数据加载 (Fashion-MNIST, CIFAR-10, Places2)
│   ├── masks.py          # 遮挡生成 (center, random_box)
│   ├── models.py         # 生成器 (Encoder-Decoder) + 判别器 (PatchGAN)
│   ├── losses.py         # GAN 损失函数
│   ├── metrics.py        # L1, MSE, PSNR 指标
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

### Fashion-MNIST & CIFAR-10

自动下载到 `--data_root` 目录。

### Places2-Subset

需要手动准备本地子集，放置在某个目录下（如 `./data/places2_subset/`），包含 jpg/png/jpeg 图片。

## 使用方法

### 训练

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

#### Places2-Subset (128×128)

```bash
# L1 基线
python train.py --dataset places2 --mode l1 --mask_type center \
    --epochs 20 --batch_size 16 --image_size 128 \
    --data_root ./data/places2_subset --max_samples 5000

# L1 + GAN
python train.py --dataset places2 --mode gan --mask_type center \
    --epochs 40 --batch_size 16 --image_size 128 \
    --data_root ./data/places2_subset --max_samples 5000 --lambda_l1 100
```

#### 显存不足时用 64×64

```bash
python train.py --dataset places2 --mode gan --mask_type center \
    --epochs 40 --batch_size 32 --image_size 64 \
    --data_root ./data/places2_subset --max_samples 5000 --lambda_l1 100
```

### 评估

```bash
python evaluate.py --dataset fashion_mnist --mode gan \
    --checkpoint ./outputs/fashion_mnist/gan/checkpoints/generator_final.pth
```

### 生成对比图

```bash
python make_comparison.py --dataset fashion_mnist \
    --l1_checkpoint ./outputs/fashion_mnist/l1/checkpoints/generator_final.pth \
    --gan_checkpoint ./outputs/fashion_mnist/gan/checkpoints/generator_final.pth
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集: fashion_mnist / cifar10 / places2 | 必填 |
| `--mode` | 训练模式: l1 / gan | 必填 |
| `--mask_type` | 遮挡类型: center / random_box | center |
| `--epochs` | 训练轮数 | 按数据集自动设置 |
| `--batch_size` | 批次大小 | 按数据集自动设置 |
| `--image_size` | 图像尺寸 | Fashion-MNIST/CIFAR-10: 32, Places2: 128 |
| `--lambda_l1` | L1 损失权重 | 100 |
| `--lr` | 学习率 | 2e-4 |
| `--data_root` | 数据根目录 | ./data |
| `--output_dir` | 输出根目录 | ./outputs |
| `--seed` | 随机种子 | 42 |
| `--max_samples` | Places2 最大样本数 | None |
| `--mask_size` | 遮挡方块边长 | 自动计算 |
| `--num_workers` | 数据加载线程数 | 4 |

## 输出内容

每个实验在 `outputs/{dataset}/{mode}/` 下保存：

```
checkpoints/          # 模型权重
samples/              # 样例图像 (epoch_N.png)
metrics.csv           # 训练指标
loss_curve.png        # 损失曲线
```

样例图格式：`Original | Mask | Masked | Prediction | Completed`

## 模型架构

- **L1 基线**: Conv Encoder → ConvTranspose Decoder, 仅 L1 损失
- **GAN 改进**: 同上生成器 + PatchGAN 判别器, L1 + 对抗损失
- 生成器层数根据 image_size 自动调整（32→3层, 64→4层, 128→5层）
