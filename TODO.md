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

- [x] 3.1a 论文研读 2605.28451 → `paper-notes/2605.28451-analysis.md`
- [x] 3.1b FP8 E4M3 表示能力分析 → `docs/fp8-e4m3-basics.md`
- [x] 3.1c FFT 蝶形误差传播模型 + Python 仿真 → `docs/fp8-fft-error-model.md` + `tests/sim_fp8_fft_error.py`
- [x] 3.1d 三种候选方案对比（含论文验证） → `docs/fp8-strategy-comparison.md` (推荐: 块浮点数)
- [x] 3.1e NVIDIA RTX 5070 Ti FP8 验证 → `src/cuda/fp8_verification.cu` + `build_fp8.bat`
- [x] 3.2 块浮点 FFT — CPU Python 原型
- [x] 3.3 块浮点 FFT — CUDA kernel v0
- [x] 3.3 审查修复 — 消除 device sync + twiddle 差异文档
- [x] 3.4 精度-性能调优 & 最终报告

## Phase 4: BF16 cuFFT 集成

- [x] 4.1 BF16 测试 + 编译验证 + LAPTOP-CHANGES
- [x] 4.1 审查修复 — BF16 SQNR 基准对齐 (FP64 reference)

## Phase 4: Sprint 4.2 — BFP 功能补全 + 性能优化 + 项目收尾 ✅

- [x] 4.2.1 BFP 接入公共 API (fp8 → BFPFFT)
- [x] 4.2.2 BFP CUDA inverse FFT (roundtrip verified)
- [x] 4.2.3 内存带宽 benchmark (large batch × large N)
- [x] 4.2.4 BFP 边界测试 (DC/极值/全零/N=2,4)
- [x] 4.2.5 README 更新
- [x] 4.3 BFP 消融：尾数位宽（mantissa bits）
- [x] 4.4 BFP 消融：指数共享粒度（group size）

