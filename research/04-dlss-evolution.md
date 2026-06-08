# DLSS 演进史：从 1.0 到 4.5

> 素材卡片 | 日期: 2026-06-08 | 来源: Wikipedia + NVIDIA 官方文档

## 版本时间线

| 版本 | 发布时间 | 核心创新 | 硬件要求 |
|------|---------|---------|---------|
| DLSS 1.0 | 2019.02 | 空间上采样 + 逐游戏训练 | RTX 20+ |
| DLSS 2.0 | 2020.04 | AI 加速的 TAAU，通用训练，引入时序信息 | RTX 20+ |
| DLSS 3.0 | 2022.09 | 光流帧生成 (Frame Gen)，AI 插帧 | RTX 40+（帧生成） |
| DLSS 3.5 | 2023.09 | Ray Reconstruction，AI 单模型替代多降噪器 | RTX 20+ |
| DLSS 4.0 | 2025.01 | Vision Transformer 超分 + Multi Frame Gen (3-4×) | RTX 50（多帧生成） |
| **DLSS 4.5** | **2026.01** | **第二代 ViT + 动态多帧生成 (15)×** | **RTX 50（动态生成）** |
| DLSS 5.0 (计划) | 2026 秋季 | 神经渲染，增强光照和材质 | 待公布 |

## 各版本详细技术

### DLSS 1.0 — 空间超分
- 纯空间上采样器，无时序信息
- 需要为每个游戏单独训练
- 效果一般，被广泛批评

### DLSS 2.0 — 时序超分（质变）
- 第一个"真正好用"的版本
- **通用模型**：不再逐游戏训练
- **时序反走样 (TAAU)**：利用历史帧 + 运动向量的 AI 增强版
- 架构：CNN-based autoencoder

### DLSS 3.0 — 帧生成
- 引入 **Optical Flow Accelerator (OFA)** 硬件
- 在两帧之间插入 AI 生成的中间帧
- 光流场 + 游戏引擎运动向量 → 扭曲当前帧 → 融合 → 输出
- 只有 RTX 40 系列支持（需要 OFA 硬件）

### DLSS 3.5 — 光线重建
- 替代路径追踪中的多个手工降噪器
- 单一 AI 模型处理所有降噪（反射、阴影、全局光照）
- 训练数据是 DLSS 3 的 5 倍
- 适用于所有 RTX GPU

### DLSS 4.0 — Transformer + 多帧生成（2025 最大飞跃）
- **CNN → Vision Transformer**：超分模型架构换代
- **Multi Frame Generation**：1 帧渲染 → 生成 3-4 帧（RTX 50）
- 相比 CNN：减少鬼影、提升时序稳定性
- VRAM 使用减少 30%（以 Warhammer 40K: Darktide 为例，4K 节省 400MB）

### DLSS 4.5 — 第二代 Transformer + 动态帧生成（当前最新）

**三大改进：**

1. **第二代 Transformer 超分模型**
   - 更好的时序稳定性
   - 改进的抗鬼影能力
   - 更好的反走样质量

2. **动态多帧生成 (Dynamic MFG)**
   - 从固定 3-4× → 动态最多 **6×**（约等于 15 帧总共）
   - 根据场景复杂度自适应调整生成倍数
   - 简单场景可以生成更多帧，复杂场景少生成保证质量

3. **FP8 精度推理**
   - RTX 40/50 利用 FP8 张量核心
   - 性能开销显著降低
   - RTX 30 系列及更老：FP8 不支持 → 性能下降 20%+（Tom's Hardware 2026.01 测试）

## 技术演进总览

```
DLSS 1.0                DLSS 2.0               DLSS 3.0/3.5            DLSS 4.0/4.5
┌─────────────┐        ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│ 空间上采样   │   +    │ 时序信息     │   +    │ 光流帧生成   │   +    │ Transformer │
│ 逐游戏训练   │   →    │ 通用模型     │   →    │ Ray Recon   │   →    │ 动态帧生成   │
│ CNN         │        │ CNN + Warp  │        │ CNN + OFA   │        │ ViT + FP8   │
└─────────────┘        └─────────────┘        └─────────────┘        └─────────────┘
    2019                   2020                   2022-2023              2025-2026
```

## 对我们的模拟项目

我们模拟 DLSS 4.5 的三个核心模块正好对应三个 DLSS 版本的技术：

| 我们的模块 | 对应的 DLSS 技术 | 可用的开源基础 |
|-----------|-----------------|---------------|
| 超分辨率模块 | DLSS 4.5 第二代 ViT 超分 | egan-wu/Super-Resolution + SwinIR |
| 帧生成模块 | DLSS 4.5 动态多帧生成 | RIFE / Practical-RIFE |
| 降噪模块 | DLSS 3.5 Ray Reconstruction | U-Net 降噪 / 轻量 ViT |

**关键差距认知：**
- 我们只能用通用图像/视频数据训练，没有游戏引擎的精确运动向量
- 用 Farneback/IFNet 估计的光流不如 GPU 光栅化器提供的精确
- 推理速度做不到实时（M4 Pro 无专用 AI 加速器对标 Tensor Core）
- 但**原理验证**和**学习目标**完全可行

## 参考资料

- [Wikipedia: Deep Learning Super Sampling](https://en.wikipedia.org/wiki/Deep_Learning_Super_Sampling)
- [NVIDIA DLSS 开发者页面](https://developer.nvidia.com/rtx/dlss)
- Tom's Hardware: "DLSS 4.5 yields 20%+ performance loss on older RTX 30/20 GPUs" (2026.01.06)
