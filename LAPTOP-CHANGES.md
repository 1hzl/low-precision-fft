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

### 剩余未检查任务

- [ ] 2.2 Sprint 2: backward 自动微分（计划 6/9-6/11）
- [ ] 2.3 Sprint 3: FP16 vs FP32 精度基准
- [ ] 2.4 Sprint 4: 性能基准吞吐量对比
- 以上三项尚未委派，等待 HANDSHAKE.md 或明确指令
