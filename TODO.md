# TODO — Low Precision FFT

## Phase 1: 环境 & 验证 ✅

- [x] 1.1 CUDA Toolkit 安装确认
- [x] 1.2 验证 CuFFTPlanCache.h FP16 路径
- [x] 1.3 FP32 vs FP16 cuFFT 基准测试
- [x] 1.4 基准报告

## Phase 2: FP16 cuFFT → PyTorch 封装

- [x] 2.1 Sprint 1: PyTorch C++ 扩展封装 cuFFT FP16 ← ✅ 已完成
- [x] 2.2 Sprint 2: backward 自动微分 ← ✅ 已完成 (gradcheck PASSED)
- [x] 2.3 Sprint 3: FP16 vs FP32 精度基准 ← ✅ 已完成 (max rel_err < 0.5%)
- [x] 2.4 Sprint 4: 性能基准吞吐量对比 ← ✅ 已完成 (raw cuFFT 1.4x-2.8x batched)
- [x] 2.4 优化: 消除 Python wrapper overhead (方案A+B) ← ✅ 已完成 (单FFT FP16≥1.0x)

## Phase 3: FP8 自研 kernel (6/4 - 6/30)

- [x] 3.1a FP8 E4M3 表示能力分析 → `docs/fp8-e4m3-basics.md`
- [x] 3.1b FFT 蝶形误差传播模型 + Python 仿真 → `docs/fp8-fft-error-model.md` + `tests/sim_fp8_fft_error.py`
- [x] 3.1c 三种候选方案对比 → `docs/fp8-strategy-comparison.md` (推荐: 块浮点数)
- [ ] 3.2 块浮点 FFT — CPU Python 原型
- [ ] 3.3 块浮点 FFT — CUDA kernel v0
- [ ] 3.4 精度-性能调优
