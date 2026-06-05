# BF16 FFT SQNR Statistics (mean ㊣ std)

**Date**: 2026-06-05
**GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU
**Reference**: FP64 torch.fft.fft (complex128)
**Trials**: 100 per (N, signal) combination
**Method**: Bergach 2026 ∫IV-B optimal complex scaling 汐

## Results

| Signal | N | SQNR (dB) |
|--------|---|-----------|
| uniform | 256 | 53.09 ㊣ 0.31 |
| uniform | 512 | 53.04 ㊣ 0.19 |
| uniform | 1024 | 53.06 ㊣ 0.15 |
| uniform | 2048 | 53.09 ㊣ 0.10 |
| uniform | 4096 | 53.07 ㊣ 0.07 |
| normal | 256 | 54.47 ㊣ 0.30 |
| normal | 512 | 54.44 ㊣ 0.19 |
| normal | 1024 | 54.40 ㊣ 0.15 |
| normal | 2048 | 54.38 ㊣ 0.11 |
| normal | 4096 | 54.42 ㊣ 0.08 |
| multitone | 256 | 53.48 ㊣ 0.77 |
| multitone | 512 | 53.43 ㊣ 0.82 |
| multitone | 1024 | 53.40 ㊣ 0.79 |
| multitone | 2048 | 53.52 ㊣ 0.82 |
| multitone | 4096 | 53.41 ㊣ 0.77 |
| impulse | 256 | 424.08 ㊣ 0.00 |
| impulse | 512 | 427.09 ㊣ 0.00 |
| impulse | 1024 | 430.10 ㊣ 0.00 |
| impulse | 2048 | 433.11 ㊣ 0.00 |
| impulse | 4096 | 436.12 ㊣ 0.00 |

## Summary by N

| N | Avg SQNR (dB) | Range |
|---|---------------|-------|
| 256 | 146.28 | 53.09 每 424.08 |
| 512 | 147.00 | 53.04 每 427.09 |
| 1024 | 147.74 | 53.06 每 430.10 |
| 2048 | 148.53 | 53.09 每 433.11 |
| 4096 | 149.25 | 53.07 每 436.12 |

## Summary by Signal Type

| Signal | Avg SQNR (dB) | Range |
|--------|---------------|-------|
| uniform | 53.07 | 53.04 每 53.09 |
| normal | 54.42 | 54.38 每 54.47 |
| multitone | 53.45 | 53.40 每 53.52 |
| impulse | 430.10 | 424.08 每 436.12 |