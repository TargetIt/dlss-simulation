# 光流帧插值原理 (RIFE / IFNet)

> 素材卡片 | 日期: 2026-06-08 | 来源: hzwer/Practical-RIFE + ECCV 2022 论文

## 一句话总结

光流帧插值 (Video Frame Interpolation, VFI) 在两张真实帧之间合成中间帧，DLSS Frame Generation 的学术基础。RIFE (Real-time Intermediate Flow Estimation) 是目前最快的 VFI 方案。

## RIFE 核心架构

```
Frame 0 ──┐                        ┌──► Intermediate Frame (t=0.5)
           ├──► IFNet ──► Flow_0→1, Flow_1→0, Fusion Map
Frame 1 ──┘
```

### IFNet (Intermediate Flow Estimation Network)

RIFE 的核心创新：直接估计**中间光流**而非双向光流。传统方法先算 Flow_0→1，然后反转得到 Flow_t→0 和 Flow_t→1，RIFE 一步到位。

### 完整 Pipeline

```
Input: img0, img1, timestep t ∈ [0,1]

1. IFNet(img0, img1, t) → flow_0→t, flow_1→t, fusion_map
2. warped_img0 = backward_warp(img0, flow_0→t)
3. warped_img1 = backward_warp(img1, flow_1→t)  
4. output = fusion_map * warped_img0 + (1-fusion_map) * warped_img1
```

### 关键细节

1. **Fusion Map**：不是简单平均两个 warp 结果，而是学习逐像素的融合权重（处理遮挡区域）
2. **多尺度推理**：在多个分辨率上估计光流，从粗到细
3. **实时性能**：RIFE 可以在 1080p 视频上跑到实时（30fps+），这是它叫 "Real-time" 的原因

### 与 DLSS Frame Generation 的关系

| | RIFE | DLSS 3 Frame Gen | DLSS 4.5 Dynamic MFG |
|---|---|---|---|
| 输入帧数 | 2 帧 | 2 帧（当前+前一帧） | 2 帧 + 历史 |
| 生成帧数 | 1 或 N（级联） | 1 帧 | 最多 5-6 帧（动态） |
| 光流 | IFNet | 硬件光流加速器 (OFA) | OFA + 改进 |
| 运动向量 | 估计 | GPU 光栅化器提供（更准） | GPU 提供 + Transformer 增强 |

## 为什么 DLSS Frame Generation 更好

1. **运动向量来源**：游戏引擎提供精确的逐像素运动向量（包括粒子、阴影、反射），RIFE 只能从像素估计
2. **HUD/UI 分离**：DLSS 知道哪些是 UI 元素（不应插值），RIFE 会错误地扭曲 UI
3. **时序一致性**：DLSS 有前后帧的完整运动链，RIFE 只考虑两帧

## 学习价值

对我们的模拟项目：RIFE 是帧生成模块的最佳起点，因为：
1. 开源、有预训练模型
2. 代码简洁（~100 行推理）
3. warp 操作与 Super-Resolution 项目的 warp 模块可以直接复用
4. 可以理解光流插值的基本原理

## 参考来源

- [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) — 实用化版本
- [arXiv RIFE](https://github.com/hzwer/arXiv2020-RIFE) — 原始论文实现
- Huang et al., "Real-Time Intermediate Flow Estimation for Video Frame Interpolation", ECCV 2022
