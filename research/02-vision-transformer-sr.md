# Vision Transformer 在超分辨率中的应用

> 素材卡片 | 日期: 2026-06-08 | 来源: NVIDIA DLSS 4 文档 + 学术文献

## 一句话总结

DLSS 4 将超分辨率模型从卷积神经网络 (CNN) 升级为 Vision Transformer (ViT)，通过自注意力机制捕获全局依赖关系，显著减少鬼影和提升时序稳定性。

## CNN vs Transformer 在超分中的对比

| 特性 | CNN (DLSS 1-3) | Vision Transformer (DLSS 4+) |
|------|---------------|---------------------------|
| 感受野 | 局部（由卷积核大小和层数决定） | 全局（自注意力看全图） |
| 参数效率 | 高（权值共享） | 中等（需要更多数据） |
| 时序一致性 | 依赖显式 warp 对齐 | 自注意力直接建模跨帧关系 |
| 鬼影问题 | 常见（局部感受野误判） | 显著减少 |
| 推理速度 | 快 | 较慢（需要优化） |

## Vision Transformer 基本原理

### 标准 ViT 架构

```
输入图像 (H×W×3)
    ↓
Patch Embedding：切成 N 个 patch，每个映射为 D 维向量
    ↓
Position Embedding：加入位置编码
    ↓
Transformer Encoder × L 层
    ├── Multi-Head Self-Attention (MHSA)
    │   Q,K,V = Linear(x), Attention = softmax(QK^T/√d)V
    ├── Layer Norm
    └── MLP (两层全连接)
    ↓
上采样 Head → HR 输出
```

### DLSS 4 如何使用 ViT

DLSS 4 不是简单的"输入一张图输出一张超分图"，而是：

1. **输入**：当前低分辨率帧 + warp 对齐的历史帧 + 运动向量 + 深度/法线等 G-Buffer
2. **ViT 处理**：将所有这些信息拼成 token 序列，Transformer 自行学习哪些信息对重建有用
3. **输出**：高分辨率帧

**为什么 Transformer 对 DLSS 特别有效：**

1. **全局一致性**：CNN 只看局部，一个像素的鬼影可能来自画面另一端的光源变化。Transformer 能看到全图。
2. **跨模态融合**：运动向量、深度、颜色是不同的"模态"，Transformer 的多头注意力天然适合融合异质信息。
3. **时序注意力**：可以把多帧的 token 拼在一起，让注意力机制直接学习"哪些历史像素对当前像素有用"。

### SwinIR / HAT 等学术 ViT 超分模型

学术界领先的 ViT 超分模型：

1. **SwinIR** (Swin Transformer for Image Restoration)
   - 使用 Swin Transformer（基于窗口的局部注意力 + 跨窗口交互）
   - 在多个图像恢复任务上 SOTA
   - 比全局 ViT 更快（窗口注意力 O(NW²) vs 全局 O(N²)）

2. **HAT** (Hybrid Attention Transformer)
   - 结合 Channel Attention + Spatial Attention
   - 更好的高频细节恢复

3. **Restormer**
   - 在 Channel 维度做注意力（而非 Spatial）
   - 计算效率更高

## DLSS 4.5 第二代 Transformer 改进

根据 NVIDIA 官方信息：
- **更好的时序稳定性**：减少了连续帧之间的细微抖动
- **改进的抗鬼影**：第二代模型对快速运动和遮挡场景更鲁棒
- **与动态帧生成协同**：Transformer 超分结果作为帧生成的输入，更好的超分质量 → 更好的插值帧
- **FP8 精度**：在 RTX 40/50 上用 FP8 降低推理开销（相比 FP16 近乎减半）

## 对我们的模拟项目

我们的超分模块可以走两条路：

**路线 A（推荐：由浅入深）**
1. 先用 egan-wu/Super-Resolution 的 CNN Phase 2 (WarpTSRNet) 理解时序超分
2. 替换骨干网络为轻量 ViT（如 SwinIR-small）
3. 对比 CNN vs ViT 效果

**路线 B（学术路线）**
1. 直接用 SwinIR / HAT 预训练模型
2. 加入时序 warp 对齐
3. 训练时序 ViT 超分

建议走路线 A，因为 M4 Pro 上训练全量 ViT 有难度（24GB 可以推理，训练可能不够）。

## 参考来源

- [NVIDIA DLSS 4 官方页面](https://developer.nvidia.com/rtx/dlss) — Transformer model 说明
- SwinIR: Liang et al., "SwinIR: Image Restoration Using Swin Transformer", ICCVW 2021
- HAT: Chen et al., "Activating More Pixels in Image Super-Resolution Transformer", CVPR 2023
