# Development Plan — Low Precision FFT

> Han 统筹 · OpenClaw 调度 · Claude Code 执行

## Phase 1：环境 & 验证 ← 当前

| # | 任务 | 负责 | 状态 |
|---|------|------|------|
| 1.1 | CUDA Toolkit 安装 (nvcc + cuFFT headers) | Han | ⬜ |
| 1.2 | 验证 CuFFTPlanCache.h FP16 路径可用 | CC | ⬜ |
| 1.3 | FP32 vs FP16 cuFFT 基准：吞吐+精度 | CC | ⬜ |
| 1.4 | 产出基准报告 (markdown) | CC | ⬜ |

**定义完成**: nvcc --version 正常 + FP16 cuFFT 跑通 + 基准数据完整

## Phase 2：GPU 内核

| # | 任务 | 负责 |
|---|------|------|
| 2.1 | FP16 cuFFT 封装内核 (动态缩放策略) | CC |
| 2.2 | 共享内存优化 + 内存合并 | CC |
| 2.3 | 误差补偿 → 精度 vs 性能 trade-off 分析 | CC |
| 2.4 | FP8 路径探索（如 E4M3/E5M2 可用） | CC |

**定义完成**: FP16 FFT 封装可用，相对误差 ≤ 1e-3，性能优于 FP32 基线

## Phase 3：PyTorch 集成

| # | 任务 | 负责 |
|---|------|------|
| 3.1 | ATen C++ 扩展 → `torch.fft.fft_lowp()` | CC |
| 3.2 | autograd 支持 (forward + backward) | CC |
| 3.3 | 设备迁移 + 多 GPU 支持 | CC |
| 3.4 | 单元测试 → CUDA + CPU 全覆盖 | CC |

**定义完成**: `import torch; torch.fft.fft_lowp(x)` 可调用，测试通过

## Phase 4：CPU 多架构

| # | 任务 | 负责 |
|---|------|------|
| 4.1 | Intel AVX-512 FP16 SIMD 实现 | CC |
| 4.2 | ARM NEON/SVE 实现 | CC |
| 4.3 | 编译时指令集自动检测 | CC |
| 4.4 | CPU 端基准测试 | CC |

**定义完成**: 三大 CPU 架构均有可用的低精度 FFT

## Phase 5：社区提交

| # | 任务 | 负责 |
|---|------|------|
| 5.1 | 整理 PR → PyTorch 官方仓库 | Han + CC |
| 5.2 | 基准报告 → PyTorch Discuss | Han |
| 5.3 | 社区反馈迭代 | CC |

**定义完成**: PR 被 PyTorch 维护者 accept 或进入 review 流程
