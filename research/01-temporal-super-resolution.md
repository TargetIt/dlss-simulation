# 时序超分辨率 (Temporal Super Resolution) 原理

> 素材卡片 | 日期: 2026-06-08 | 来源: egan-wu/Super-Resolution + NVIDIA DLSS 文档

## 一句话总结

时序超分辨率利用历史帧的信息来增强当前帧的重建质量。核心思路是：相邻帧包含互补的亚像素信息，通过运动补偿对齐后融合，可以显著优于单帧超分。

## 核心原理

### 1. 为什么时序信息有用

```
单帧超分： LR(t) → CNN → HR(t)        只能从空间模式猜测高频细节
时序超分： LR(t-1), LR(t) → Warp+Fuse → HR(t)  历史帧提供了额外采样点
```

真实游戏渲染中，每帧的渲染位置有亚像素级别偏移（jitter），所以 LR(t-1) 和 LR(t) 包含了同一场景的不同采样。把这些采样对齐融合，等价于增加了有效分辨率。

### 2. 三种演进架构（来自 egan-wu/Super-Resolution）

| Phase | 架构 | 输入 | PSNR | 复杂度 |
|-------|------|------|:----:|:-----:|
| 1 | Early Fusion | LR(t-1) ⊕ LR(t) = 6ch | 29.01 | 低 |
| 2 | Warp-then-Fuse | backward_warp(LR(t-1)) ⊕ LR(t) ⊕ mask = 7ch | **33.37** | 中 |
| 3 | Recurrent Feedback | LR(t) + warped HR(t-1) 双分支融合 | 28.69* | 高 |

### 3. 关键技术细节

#### 3.1 Backward Warp（反向扭曲）

```python
# 核心操作：grid_sample
# 对于输出图像上每个像素 (i,j)，从输入帧的 (j+flow_x, i+flow_y) 位置采样
warped = F.grid_sample(frame, sample_grid, mode='bilinear')
```

**为什么用 backward warp 而不是 forward warp？**
- Forward warp：源像素映射到多个目标像素 → 产生空洞和重叠
- Backward warp：每个目标像素反向查找源像素 → 无空洞，每个像素都有值

**遮挡处理 (Occlusion Mask)**
- 采样位置超出边界 → 该像素被遮挡 → mask=0
- 网络看到 mask 知道"这里没有有效历史信息"，可以退化为只用当前帧

#### 3.2 运动向量 (Motion Vectors)

真实 DLSS 从 GPU 光栅化器接收逐像素运动向量。在图像数据集上模拟时：

```
方法1 (训练): 从 HR 图裁两个不同偏移的 crop → 已知像素偏移 → 作为刚性运动向量
方法2 (推理): Farneback 稠密光流估计 → 逐像素运动场 → 作为稠密运动向量
```

#### 3.3 Halton Jitter（Phase 3 关键技巧）

DLSS 的真精髓：每帧渲染时对投影矩阵施加亚像素抖动，使相邻帧采样不同位置。

```
Frame t-1: 采样位置 (x+0.3, y+0.7)
Frame t:   采样位置 (x+0.8, y+0.2)
Frame t+1: 采样位置 (x+0.5, y+0.9)
```

Halton 序列（一种准随机低差异序列）确保抖动均匀覆盖整个亚像素空间。循环网络通过多帧累积这些不同采样，重建出超越单帧分辨率的细节。

#### 3.4 Recurrent Feedback Loop（Phase 3 架构）

```
                    ┌──────────────────┐
                    │   HR(t-1) 输出    │ ←── 上一个时间步的输出
                    │   (128×128)      │
                    └────────┬─────────┘
                             │ warp (HR空间)
                             ▼
LR(t) ──► LR Branch ──► feature(64ch @ HR) ──┐
        (32×32)    PixelShuffle ×4            ├─► concat(128ch) ──► refine ──► HR(t)
                                              │
                    HR History Branch ────────┘
                    feature(64ch @ HR)
```

关键设计：
- **LR 分支**：在低分辨率空间提取特征 → PixelShuffle 上采样到 HR 特征空间
- **History 分支**：在 HR 空间 warp 上一帧的 HR 输出 → 提取特征
- **融合**：concat 两个分支的特征 → 残差块 refine → 输出 HR(t)

**Teacher Forcing (训练) vs True Recurrence (推理)**
- 训练时：HR(t-1) 使用 Ground Truth（避免误差累积）
- 推理时：HR(t-1) 使用模型自身上一帧输出（真正的循环）
- Scheduled Sampling 逐步从 Teacher Forcing 过渡到自反馈

### 4. DLSS 版本对照

| DLSS 版本 | 超分模型 | 关键改进 |
|-----------|---------|---------|
| DLSS 2 (2020) | CNN (每游戏训练) → 通用 CNN | 引入时序信息，替代 DLSS 1 的逐游戏训练 |
| DLSS 3 (2022) | CNN + 光流帧生成 | 加入 Frame Generation |
| DLSS 4 (2025) | Vision Transformer | CNN → Transformer，更好时序稳定性 |
| DLSS 4.5 (2026) | 第二代 Vision Transformer | 改进抗鬼影、时序稳定性、动态帧生成 |

## 参考来源

- [egan-wu/Super-Resolution](https://github.com/egan-wu/Super-Resolution) — PyTorch DLSS-style TSR 实现
- NVIDIA DLSS 官方开发者页面 — transformer model 替代 CNN
- [NVIDIA DLSS 4 技术博客](https://www.nvidia.com/en-us/geforce/news/dlss4-multi-frame-generation/) — Vision Transformer 架构细节
