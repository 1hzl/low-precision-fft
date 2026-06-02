# Low Precision FFT — 产品规格

## 一句话

为 PyTorch 社区补齐 FP16/FP8 低精度 FFT 算子，覆盖 GPU CUDA 和 CPU 多架构 SIMD，服务 LLM 微调等频域计算场景。

## 目标平台

| 平台 | 精度 | 后端 |
|------|------|------|
| NVIDIA GPU (SM_120+) | FP16, FP8 | CUDA cuFFT |
| Intel x86_64 | FP16, FP8 | AVX-512 |
| ARM AArch64 | FP16, FP8 | NEON / SVE |
| RISC-V | FP16 | RVV 0.7+ |

## 核心 API

```python
torch.fft.fft_lowp(input, n=None, dim=-1, norm=None, dtype=torch.float16)
torch.fft.ifft_lowp(input, n=None, dim=-1, norm=None, dtype=torch.float16)
```

行为：输入/输出低精度 Tensor，内部处理动态缩放和误差补偿，精度损失相比 FP32 ≤ 1e-3（相对误差）。

## 技术策略

1. **复用现有路径** — PyTorch `CuFFTPlanCache.h` 已有 `kHalf→CUDA_C_16F` 映射，先验证再拓展
2. **动态缩放** — 仿 FP8 训练思路，每次 FFT 前根据数据范围自动缩放
3. **可合并适配器** — 参考 Block Circulant Adapter，设计即插即用的轻量封装

## 非目标

- 不追求击败 cuFFT 原生 FP32/C2C 性能（那是硬件厂的事）
- 不覆盖 BF16（PyTorch 已有部分支持）
- 不实现全新 FFT 算法（复用 cuFFT + FFTW 等成熟库）
