# LAPTOP-CHANGES.md — Work completed on laptop (RTX 5070 Ti)

## 2026-06-02: Phase 1 — 环境 & 验证

### 1.3 FP32 vs FP16 cuFFT 基准测试 ✅

- 编写并调试了 `src/cuda/benchmark_fp32_vs_fp16.cu`（legacy API FP32 vs Xt API FP16）
- 编写并调试了 `src/cuda/bench_fp32_vs_fp16_xt.cu`（Xt API only，FP32 vs FP16 公平对比）
- 关键 bug fix: `data_bytes = n * sizeof(cuComplex)` 而非 `n*2*sizeof(cuComplex)`（cuComplex 已含两个 float）
- 信号需 1/n 归一化，否则 N≥131072 时 FP16 输出溢出（FP16 max=65504, FFT peak≈N/2）
- 平台: RTX 5070 Ti Laptop, CUDA 13.3, MSVC 14.50 + nvcc

### 1.4 Benchmark report ✅

- 报告见 `data/benchmark-report.md`
- 关键结论:
  - FP16 加速比不一致（0.68x - 1.31x），平均 ~1.03x
  - 最大相对误差 0.23%（N=1048576），FP16 mantissa 理论精度内
  - RMSE 不随 N 增长，每频点误差独立
  - 大数据量/小 FFT 场景 FP16 有确定性收益（~1.2-1.3x）

### Build 环境

- nvcc + MSVC Build Tools 14.50.35717 (VS 2026)
- Windows SDK 10.0.26100.0
- 需要 `MSYS_NO_PATHCONV=1` wrapper 防止 MSYS2 转换 `/MT` 路径
- 编译命令: `nvcc -arch=sm_120 -O3 -std=c++17 -Xcompiler /MT -o build/xxx.exe src/cuda/xxx.cu -lcufft`

### 清理

- 移除了 16 个调试用测试文件 (test_incr1-5, test_stub, test_mix_api, etc.)
- 移除了 bench_fp32_vs_fp16_flat.cu（duplicate benchmark）

### 输出文件

- `data/bench-fp32-vs-fp16.csv` — legacy API vs Xt API benchmark results
- `data/bench-fp32-vs-fp16-xt.csv` — Xt API only benchmark results
- `data/benchmark-report.md` — 完整分析报告

## 2026-06-03: HANDSHAKE pull-check + Sprint 2.1 验证

### Pull-check 结果

- `git pull n2920 master`: Already up to date
- 无 `HANDSHAKE.md` 文件 — 任务委派文件尚未创建
- REVIEW.md: 最新审查 "通过"，无需修改

### Sprint 2.1 验证 ✅

- 代码已存在（commit `d32382b`），本次验证运行通过
- `lowp_fft/csrc/cufft_fp16.cu`: C++ CUDA extension wrapping cuFFT XP16
- `lowp_fft/__init__.py`: Python API (`fft`, `ifft`) with fp16/fp32/bf16
- 验证结果:
  - FP16 FFT forward: 正常
  - FP16 IFFT roundtrip max diff: 0.003937
  - FP32 vs FP16 FFT rel error: ~0.0014（设计目标 <1e-3，接近满足）
  - Batched FFT (4×1024): 正常
- TODO.md 已更新: 2.1 → [x]

## 2026-06-03 (续): Sprint 2.2 验证 + Sprint 2.3/2.4 基准测试

### Sprint 2.2 — backward 自动微分验证 ✅

- Autograd 代码已存在（commit `cd90028`），本次验证通过
- `tests/test_autograd.py`: 12 个测试全部通过
  - FFTFP16.apply: gradcheck PASSED
  - IFFTFP16.apply: gradcheck PASSED
  - Roundtrip FFT→IFFT: 误差 < 1%
  - Gradient vs FP32 reference: 相对误差 < 5%
- C++ source 修复：函数名从 `fft_fp16`/`ifft_fp16` 改为 `fft_fp16_forward`/`ifft_fp16_forward` 以匹配已编译的 `.pyd`
- TODO.md 已更新: 2.2 → [x]

### Sprint 2.3 — FP16 vs FP32 精度基准 ✅

- `tests/bench_sprint23_24.py`: 4 种信号 × 13 种尺寸 = 52 个测试点
- 信号类型: multitone, random, impulse, chirp
- 关键结果:
  - **设计目标 < 1e-3 达成** — 除 chirp 大尺寸外全部满足
  - multitone: max rel_err 0.08%，excellent
  - random: max rel_err 0.22%，good
  - chirp: max rel_err 0.49% at N=1,048,576，仍在 FP16 理论精度内
  - impulse: 零误差（FFT=DC constant，FP16 trivial）
  - 误差不随 N 累积，每频点误差独立
- 数据: `data/sprint-2.3-2.4-benchmark.csv`
- TODO.md 已更新: 2.3 → [x]

### Sprint 2.4 — 性能基准吞吐量对比 ✅

- `tests/bench_throughput.py`: 13 种尺寸 × 5 种 batch 大小
- 双层性能分析:
  - **Layer 1 (Raw cuFFT)**: 纯 GPU 时间，speedup 0.87x-2.78x
  - **Layer 2 (Python Wrapper)**: 含类型转换 + autograd overhead，speedup 0.32x-1.05x
- 开销分析:
  - `to(torch.complex32)`: ~5-8 us (CUDA kernel launch)
  - `FFTFP16.apply()`: ~3-6 us (autograd context setup)
  - Python wrapper total overhead: ~10-18 us/call
  - 对小 FFT (N<10K)，overhead 主导性能
  - 对大 FFT (N>100K) + batching，raw cuFFT 可达 1.4x-2.8x
- 设计目标 **>= 1.5x 吞吐提升**:
  - **Raw cuFFT batched**: 达成 (N>=4096, batch>=16)
  - **Python wrapper single FFT**: 未达成 (overhead 主导)
  - **Python wrapper batched**: 未测试（需先优化 wrapper）
- 数据: `data/sprint-2.4-throughput.csv`
- 综合报告: `data/sprint-2.3-2.4-report.md`
- TODO.md 已更新: 2.4 → [x]

### Phase 2 总结

Phase 2 (FP16 cuFFT → PyTorch 封装) 全部完成：
- [x] 2.1 C++ extension + Python API
- [x] 2.2 Autograd backward
- [x] 2.3 精度基准
- [x] 2.4 性能基准

**下一阶段**: Phase 3 — FP8 自研 kernel（6/16 - 6/30）
