# Low Precision FFT for PyTorch

为 PyTorch 实现 FP16/FP8 低精度 FFT（GPU CUDA + CPU 多架构 SIMD），支撑 LLM 微调场景。
最终目标：贡献代码到 PyTorch 社区。

## 参考

- arXiv 2505.00582 — Block Circulant Adapter for LLMs（核心灵感论文）
- PyTorch CuFFTPlanCache.h — kHalf → CUDA_C_16F 已有映射，待验证
- GPU: NVIDIA RTX 5070 Ti (SM_120, 12GB, CUDA 13.0)

## 项目状态

Phase 1: 环境验证 & 基准测试
