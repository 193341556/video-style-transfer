# 视频风格迁移系统

基于 **AdaIN 自适应实例归一化** 与 **PWC-Net 光流时序一致性** 的视频风格迁移系统，支持固定风格与任意风格两种迁移模式。

## 效果展示

> 将普通视频迁移为指定艺术风格，同时通过光流约束保持帧间时序一致性，避免画面闪烁。

## 技术架构

| 模块 | 技术方案 |
|------|---------|
| 特征提取 | VGG16（relu1_2 / relu2_2 / relu3_3 / relu4_3） |
| 风格迁移 | AdaIN 自适应实例归一化（对齐内容特征的均值/方差） |
| 时序一致性 | PWC-Net 光流估计 + 时序一致性损失（CuPy CUDA 加速） |
| 损失函数 | 内容损失 + 风格损失（Gram 矩阵）+ 时序一致性损失 |
| 评估指标 | SSIM / PSNR / LPIPS / 光流一致性 |

## 网络结构

```
Encoder（3 × Conv）
    ↓
AdaIN（内嵌 MLP，输入风格向量）
    ↓
5 × Residual Block
    ↓
Decoder（2 × Deconv）
    ↓
风格化帧输出
```

## 文件结构

```
├── adain.py              # AdaIN 自适应实例归一化模块
├── vgg.py                # VGG16 / VGG19 特征提取网络
├── transformer.py        # 固定风格 Transformer 网络
├── transformer_any.py    # 任意风格 Transformer 网络
├── train.py              # 固定风格训练脚本
├── train_any.py          # 任意风格训练脚本
├── cstyletrain_any.py    # 从检查点继续训练
├── stylize.py            # 固定风格逐帧风格化
├── stylize_any.py        # 任意风格逐帧风格化
├── video.py              # 固定风格视频迁移入口
├── video_any.py          # 任意风格视频迁移入口
├── evaluater.py          # 视频质量评估
├── pwc.py                # PWC-Net 光流估计网络
├── utils.py              # 工具函数（Gram 矩阵、图像读写等）
├── correlation/
│   └── correlation.py    # CuPy CUDA 相关性计算
├── images/               # 风格参考图像
└── style_images/         # 任意风格训练图像库
```

## 环境要求

- Python 3.x
- CUDA GPU（PWC-Net 光流计算需要）
- CuPy（版本需与 CUDA 版本匹配）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 固定风格迁移

```bash
python video.py
```

### 任意风格迁移

```bash
python video_any.py
```

### 训练

```bash
# 固定风格训练
python train.py

# 任意风格训练
python train_any.py

# 从检查点继续训练
python cstyletrain_any.py
```

### 质量评估

```bash
python evaluater.py
```

输出 SSIM、PSNR、LPIPS 风格相似性及光流一致性指标。
