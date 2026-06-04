# Sprint 3.4 — BFP FP8 FFT Final Report

**Date**: 2026-06-04  
**Platform**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB VRAM), CUDA 13.3, SM_120  
**Project**: Low-Precision FFT — Phase 3 (FP8 Custom Kernel)

---

## 1. Accuracy Summary

SQNR (Signal-to-Quantization-Noise Ratio) measured against FP64 reference FFT.
Values reported for random normal input (most representative signal type).

| Method | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|--------|-------|-------|--------|--------|--------|
| cuFFT FP32 (complex64) | 138 dB | 136 dB | 135 dB | 136 dB | 135 dB |
| cuFFT FP16 (complex32) | 61 dB | 61 dB | 60 dB | 60 dB | 57 dB |
| **BFP FP8 (CUDA v0)** | **22 dB** | **22 dB** | **21 dB** | **21 dB** | **20 dB** |
| Naive FP8 (every-op quantize) | ~0 dB | ~0 dB | ~0 dB | ~0 dB | ~0 dB |
| FP8 Hardware (naive) | 9 dB | — | — | — | — |

### Key Observations

1. **BFP provides ~20-30 dB gain over naive FP8** — The per-stage shared exponent
   reduces quantization events from N·log₂(N) to log₂(N) per value, preventing the
   catastrophic collapse seen in naive FP8 (0 dB SQNR at N ≥ 256).

2. **FP16 cuFFT is the practical precision floor today** — 57-61 dB SQNR is adequate
   for most signal processing and ML workloads.

3. **BFP FP8 at ~20 dB is sufficient for some applications** — Comparable to 3-4 bit
   effective precision. Useful for memory-bandwidth-bound scenarios where 2× storage
   reduction vs FP16 matters.

4. **N scaling behavior** — BFP SQNR decreases slightly with N (22→20 dB from
   N=256→4096), consistent with per-stage quantization error accumulation.
   The decay is much slower than naive FP8's O(N) collapse.

---

## 2. Throughput Benchmark

GPU kernel execution time measured with CUDA events. BFP times include all GPU work
(stage loop + dequant); cuFFT times include the full library call.

### Per-FFT Latency (μs)

| N | Batch | BFP FP8 | cuFFT FP16 | cuFFT FP32 | BFP vs FP16 Ratio |
|---|-------|---------|------------|------------|-------------------|
| 256 | 1 | 111.5 | 16.3 | 34.3 | 0.15× |
| 256 | 16 | 111.5 | 1.4 | 1.2 | 0.01× |
| 256 | 64 | 111.5 | 0.1 | 0.2 | 0.00× |
| 256 | 256 | 111.5 | 0.03 | 0.05 | 0.00× |
| 512 | 1 | 126.5 | 13.1 | 15.1 | 0.10× |
| 512 | 16 | 126.5 | 0.7 | 1.5 | 0.01× |
| 512 | 64 | 126.5 | 0.1 | 0.3 | 0.00× |
| 512 | 256 | 126.5 | 0.07 | 0.13 | 0.00× |
| 1024 | 1 | 137.1 | 16.8 | 12.4 | 0.12× |
| 1024 | 16 | 137.1 | 0.6 | 0.8 | 0.00× |
| 1024 | 64 | 137.1 | 0.1 | 0.2 | 0.00× |
| 1024 | 256 | 137.1 | 0.03 | 0.04 | 0.00× |
| 2048 | 1 | 143.7 | 8.2 | 17.9 | 0.06× |
| 2048 | 16 | 143.7 | 1.0 | 0.9 | 0.01× |
| 2048 | 64 | 143.7 | 0.2 | 0.2 | 0.00× |
| 2048 | 256 | 143.7 | 0.04 | 0.05 | 0.00× |
| 4096 | 1 | 154.8 | 9.4 | 10.3 | 0.06× |
| 4096 | 16 | 154.8 | 0.5 | 0.6 | 0.00× |
| 4096 | 64 | 154.8 | 0.2 | 0.2 | 0.00× |
| 4096 | 256 | 154.8 | 0.04 | 0.08 | 0.00× |

### Throughput (GFLOPS)

| N | Batch | BFP FP8 | cuFFT FP16 | cuFFT FP32 |
|---|-------|---------|------------|------------|
| 256 | 1 | 0.09 | 0.63 | 0.30 |
| 4096 | 1 | 1.59 | 26.07 | 23.85 |
| 4096 | 256 | 1.59 | 6692 | 2906 |

### Key Observations

1. **BFP v0 is ~7-17× slower than cuFFT FP16 for single FFT** — Expected for a
   research-grade kernel with no optimization (no shared memory, no warp-level
   parallelism tuning, no occupancy optimization).

2. **cuFFT batching provides massive throughput** — Amortized per-FFT latency drops
   from ~10 μs to ~0.04 μs with batch=256 at N=4096. cuFFT parallelizes across the
   batch dimension efficiently.

3. **BFP has no batch acceleration** — Each FFT runs independently on the GPU.
   Adding batch support (multiple FFTs in parallel) would require a different kernel
   launch strategy.

4. **Performance ceiling is far above current BFP** — cuFFT FP16 achieves ~6700 GFLOPS
   at N=4096 batch=256, while BFP achieves ~1.6 GFLOPS. The gap indicates substantial
   optimization headroom.

---

## 3. Analysis

### BFP vs Naive FP8: The Big Win

The per-stage shared exponent strategy is validated. Naive FP8 FFT collapses to
~0 dB SQNR at N ≥ 256 (every arithmetic op quantized → error dominates signal).
BFP reduces quantization to once per value per stage, preserving ~20 dB SQNR
across all tested sizes.

### Performance Gap: Research Kernel vs Production Library

The current BFP CUDA kernel (v0) prioritizes correctness over performance.
Key optimization opportunities for future iterations:

1. **Shared memory**: Load twiddle factors and FP8 data into shared memory
2. **Warp-level intrinsics**: Use warp shuffle for butterfly pairs within a warp
3. **Occupancy tuning**: Current BLOCK_SIZE=256 may not be optimal for all N
4. **Kernel fusion**: Merge requantize into the butterfly kernel via `__syncthreads()`
5. **Vectorized loads**: Load FP8 data as `uchar4` or wider types
6. **Multi-FFT batching**: Process multiple independent FFTs in parallel

### When BFP FP8 Makes Sense

- **Memory-bandwidth-bound scenarios**: 2× data reduction vs FP16, 4× vs FP32
- **Tolerance for ~20 dB SQNR**: Coarse signal analysis, feature detection
- **Large batch inference**: Where memory capacity is the bottleneck
- **Not suitable for**: High-precision scientific computing, audio processing,
  or any application requiring >30 dB SQNR

---

## 4. Project Status

### Phase 1-2: FP16 cuFFT (Complete)

- [x] cuFFT FP16 extension for PyTorch
- [x] Autograd backward pass
- [x] Accuracy: <0.5% relative error vs FP32
- [x] Performance: 1.0-2.8× speedup vs FP32

### Phase 3: FP8 Custom Kernel (Complete)

- [x] Sprint 3.1: Theory & error analysis (FP8 E4M3, BFP model, strategy comparison)
- [x] Sprint 3.2: Python BFP prototype (validates algorithm)
- [x] Sprint 3.3: CUDA kernel v0 (correctness verified, no per-stage sync)
- [x] Sprint 3.4: Precision-performance analysis (this report)

### Next Steps (Phase 4: July 2026)

- [ ] BF16 cuFFT wrapper (reuse FP16 framework)
- [ ] BFP CUDA kernel optimization (shared memory, warp intrinsics, occupancy)
- [ ] CPU SIMD fallback (x86 AVX2)
- [ ] PyTorch PR preparation

---

## 5. Artifacts

| File | Description |
|------|-------------|
| `data/sprint-3.4-throughput.csv` | Throughput benchmark data (20 configs) |
| `data/sprint-3.2-bfp-bench.csv` | BFP vs naive FP8 SQNR (27 configs) |
| `data/sprint-2.4-throughput.csv` | FP16 vs FP32 cuFFT throughput |
| `tests/bench_bfp_throughput.py` | Throughput benchmark script |
| `src/cuda/bfp_fft.cu` | BFP CUDA kernel (v0, with --bench mode) |
| `tests/test_bfp_cuda.py` | BFP CUDA test suite (14 tests) |

---

*End of Sprint 3.4 report. Phase 3 complete.*
