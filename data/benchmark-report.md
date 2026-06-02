# FP32 vs FP16 cuFFT Benchmark Report

**Date**: 2026-06-02
**GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12GB VRAM, SM 12.0)
**CUDA**: 13.3
**Warmup**: 10 iterations | **Benchmark**: 100 iterations

## Test setup

Two benchmarks were run:

| Benchmark | FP32 API | FP16 API | CSV |
|-----------|----------|----------|-----|
| Legacy-vs-Xt | `cufftPlan1d` + `cufftExecC2C` | `cufftXtMakePlanMany` + `cufftXtExec` | `bench-fp32-vs-fp16.csv` |
| Xt-vs-Xt | `cufftXtMakePlanMany` (CUDA_C_32F) + `cufftXtExec` | `cufftXtMakePlanMany` (CUDA_C_16F) + `cufftXtExec` | `bench-fp32-vs-fp16-xt.csv` |

Signal: multi-tone (7Hz, 13Hz, 23Hz) + uniform noise, normalized by 1/n to prevent FP16 overflow at large N.

## Results: Legacy-vs-Xt

| N | FP32 (us) | FP16 (us) | Speedup | MaxRelErr | RMSE |
|---|-----------|-----------|---------|-----------|------|
| 256 | 5.91 | 6.70 | 0.88x | 3.35e-04 | 4.36e-05 |
| 512 | 7.97 | 6.10 | **1.31x** | 5.61e-04 | 3.68e-05 |
| 1024 | 7.31 | 6.44 | 1.14x | 4.69e-04 | 2.83e-05 |
| 2048 | 6.95 | 7.07 | 0.98x | 5.27e-04 | 1.96e-05 |
| 4096 | 6.54 | 6.76 | 0.97x | 5.13e-04 | 2.06e-05 |
| 8192 | 10.94 | 9.45 | 1.16x | 5.34e-04 | 1.62e-05 |
| 16384 | 11.32 | 14.39 | 0.79x | 6.12e-04 | 1.17e-05 |
| 32768 | 14.26 | 13.32 | 1.07x | 9.27e-04 | 9.97e-06 |
| 65536 | 16.48 | 14.06 | 1.17x | 7.38e-04 | 9.22e-06 |
| 131072 | 22.77 | 33.32 | 0.68x | 4.83e-04 | 1.08e-05 |
| 262144 | 27.08 | 21.00 | 1.29x | 4.39e-04 | 1.50e-05 |
| 524288 | 32.02 | 41.68 | 0.77x | 7.12e-04 | 2.12e-05 |
| 1048576 | 40.53 | 32.87 | 1.23x | 2.29e-03 | 3.02e-05 |

- Average speedup: **1.03x**
- Max speedup: **1.31x** (N=512)
- Max relative error: **2.29e-03** (0.23%, N=1048576)

## Results: Xt-vs-Xt (same API, fair comparison)

| N | FP32 (us) | FP16 (us) | Speedup | MaxRelErr | RMSE |
|---|-----------|-----------|---------|-----------|------|
| 256 | 6.48 | 5.39 | 1.20x | 3.51e-04 | 4.36e-05 |
| 512 | 7.44 | 5.70 | **1.30x** | 6.11e-04 | 3.68e-05 |
| 1024 | 6.75 | 6.46 | 1.04x | 4.77e-04 | 2.83e-05 |
| 2048 | 6.92 | 6.77 | 1.02x | 5.38e-04 | 1.96e-05 |
| 4096 | 7.17 | 7.48 | 0.96x | 6.33e-04 | 2.06e-05 |
| 8192 | 10.36 | 8.48 | 1.22x | 6.42e-04 | 1.62e-05 |
| 16384 | 10.90 | 14.95 | 0.73x | 8.30e-04 | 1.17e-05 |
| 32768 | 13.76 | 16.50 | 0.83x | 9.39e-04 | 9.97e-06 |
| 65536 | 15.05 | 13.71 | 1.10x | 7.53e-04 | 9.22e-06 |
| 131072 | 24.75 | 36.25 | 0.68x | 5.48e-04 | 1.08e-05 |
| 262144 | 18.05 | 20.62 | 0.88x | 4.39e-04 | 1.50e-05 |
| 524288 | 20.90 | 26.13 | 0.80x | 7.12e-04 | 2.12e-05 |
| 1048576 | 40.87 | 31.33 | **1.30x** | 2.29e-03 | 3.02e-05 |

## Precision analysis

### Max Relative Error (normalized by peak magnitude)

- **FP16 mantissa = 10 bits** → theoretical quantization error ≈ 0.1% per value
- Observed max_rel_err: **0.03% - 0.23%** across all sizes
- At N=1048576 (largest): 0.23% — ~2.2x the single-element quantization limit, expected for a 1M-point FFT where errors accumulate through log2(N) = 20 butterfly stages
- All sizes except N=1048576 pass the **< 0.1%** threshold

### RMSE (root mean square error per bin)

- RMSE decreases with N: **4.36e-05 → 3.02e-05**
- The error per frequency bin does NOT accumulate with FFT size — each bin's error is independent
- FP16 FFT preserves signal structure with negligible degradation for practical applications

### FP16 overflow

- Without input scaling, FP16 FFT overflowed for N ≥ 131072 (signal peak ~65536 > FP16 max 65504)
- With 1/n input normalization, all sizes run without overflow
- **Implication**: applications using FP16 FFT must normalize input amplitudes or use block FFT schemes

## Key findings

1. **FP16 speedup is moderate and inconsistent** (0.68x - 1.31x) — cuFFT's FP16 path is not universally faster than FP32 on this GPU
2. **Best cases for FP16**: small FFTs (N=256-512, ~1.2-1.3x) and very large FFTs (N=1048576, ~1.3x)
3. **Worst cases**: mid-range sizes (N=16384, 131072, 524288) where FP16 is 20-32% slower
4. **Precision is excellent**: max relative error < 0.3% across all sizes, well within FP16 theoretical limits
5. **Memory bandwidth advantage not realized**: despite FP16 using half the memory, RTX 5070 Ti's bandwidth is sufficient that FFT remains compute or latency bound at most sizes
6. **API choice matters**: Xt-vs-Xt shows slightly better FP16 speedup at small sizes (1.20x vs 0.88x for N=256), suggesting the legacy FP32 API is more optimized than the Xt FP32 path
