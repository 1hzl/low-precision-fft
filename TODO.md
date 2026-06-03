# TODO — Low Precision FFT

## Phase 1: 环境 & 验证 ✅

- [x] 1.1 CUDA Toolkit 安装确认
- [x] 1.2 验证 CuFFTPlanCache.h FP16 路径
- [x] 1.3 FP32 vs FP16 cuFFT 基准测试
- [x] 1.4 基准报告

## Phase 2: FP16 cuFFT → PyTorch 封装

- [x] 2.1 Sprint 1: PyTorch C++ 扩展封装 cuFFT FP16 ← ✅ 已完成
- [ ] 2.2 Sprint 2: backward 自动微分 ← 🔄 已委派 (HANDSHAKE)
- [ ] 2.3 Sprint 3: FP16 vs FP32 精度基准
- [ ] 2.4 Sprint 4: 性能基准吞吐量对比
