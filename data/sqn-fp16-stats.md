# FP16 FFT SQNR Statistics (mean ㊣ std)

**Date**: 2026-06-05
**GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU
**Reference**: FP64 torch.fft.fft (complex128)
**Trials**: 100 per (N, signal) combination
**Method**: Bergach 2026 ∫IV-B optimal complex scaling 汐

## Results

| Signal | N | SQNR (dB) |
|--------|---|-----------|
| uniform | 256 | 61.28 ㊣ 0.32 |
| uniform | 512 | 60.57 ㊣ 0.22 |
| uniform | 1024 | 59.91 ㊣ 0.16 |
| uniform | 2048 | 59.35 ㊣ 0.10 |
| uniform | 4096 | 56.47 ㊣ 0.08 |
| normal | 256 | 61.56 ㊣ 0.31 |
| normal | 512 | 60.78 ㊣ 0.22 |
| normal | 1024 | 60.11 ㊣ 0.16 |
| normal | 2048 | 59.52 ㊣ 0.11 |
| normal | 4096 | 56.53 ㊣ 0.07 |
| multitone | 256 | 61.60 ㊣ 0.81 |
| multitone | 512 | 60.84 ㊣ 0.71 |
| multitone | 1024 | 60.04 ㊣ 0.74 |
| multitone | 2048 | 59.48 ㊣ 0.64 |
| multitone | 4096 | 56.50 ㊣ 0.62 |
| impulse | 256 | 424.08 ㊣ 0.00 |
| impulse | 512 | 427.09 ㊣ 0.00 |
| impulse | 1024 | 430.10 ㊣ 0.00 |
| impulse | 2048 | 433.11 ㊣ 0.00 |
| impulse | 4096 | 436.12 ㊣ 0.00 |

## Summary by N

| N | Avg SQNR (dB) | Range |
|---|---------------|-------|
| 256 | 152.13 | 61.28 每 424.08 |
| 512 | 152.32 | 60.57 每 427.09 |
| 1024 | 152.54 | 59.91 每 430.10 |
| 2048 | 152.86 | 59.35 每 433.11 |
| 4096 | 151.41 | 56.47 每 436.12 |

## Summary by Signal Type

| Signal | Avg SQNR (dB) | Range |
|--------|---------------|-------|
| uniform | 59.51 | 56.47 每 61.28 |
| normal | 59.70 | 56.53 每 61.56 |
| multitone | 59.69 | 56.50 每 61.60 |
| impulse | 430.10 | 424.08 每 436.12 |