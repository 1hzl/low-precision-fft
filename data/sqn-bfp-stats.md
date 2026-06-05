# BFP FP8 FFT SQNR Statistics (mean ± std)

**Date**: 2026-06-05
**Platform**: CPU (Python/NumPy BFP prototype)
**Reference**: FP64 numpy.fft.fft (complex128)
**Trials**: 100 per (N, signal) combination
**Method**: Bergach 2026 §IV-B optimal complex scaling α
**Implementation**: BFPFP8 — block floating-point Radix-2 DIT, one shared exponent per stage

## Results

| Signal | N | SQNR (dB) |
|--------|---|-----------|
| uniform | 256 | 22.11 ± 0.31 |
| uniform | 512 | 21.61 ± 0.22 |
| uniform | 1024 | 21.17 ± 0.16 |
| uniform | 2048 | 20.81 ± 0.11 |
| uniform | 4096 | 20.44 ± 0.07 |
| normal | 256 | 22.46 ± 0.31 |
| normal | 512 | 21.95 ± 0.23 |
| normal | 1024 | 21.45 ± 0.14 |
| normal | 2048 | 21.05 ± 0.11 |
| normal | 4096 | 20.68 ± 0.08 |
| multitone | 256 | 23.37 ± 0.77 |
| multitone | 512 | 22.87 ± 0.62 |
| multitone | 1024 | 22.45 ± 0.57 |
| multitone | 2048 | 22.11 ± 0.49 |
| multitone | 4096 | 21.92 ± 0.38 |
| impulse | 256 | 424.08 ± 0.00 | *
| impulse | 512 | 427.09 ± 0.00 | *
| impulse | 1024 | 430.10 ± 0.00 | *
| impulse | 2048 | 433.11 ± 0.00 | *
| impulse | 4096 | 436.12 ± 0.00 | *

## Summary by N

| N | Avg SQNR (dB) | Range |
|---|---------------|-------|
| 256 | 123.01 | 22.11 – 424.08 |
| 512 | 123.38 | 21.61 – 427.09 |
| 1024 | 123.79 | 21.17 – 430.10 |
| 2048 | 124.27 | 20.81 – 433.11 |
| 4096 | 124.79 | 20.44 – 436.12 |

## Summary by Signal Type

| Signal | Avg SQNR (dB) | Range |
|--------|---------------|-------|
| uniform | 21.23 | 20.44 – 22.11 |
| normal | 21.52 | 20.68 – 22.46 |
| multitone | 22.54 | 21.92 – 23.37 |
| impulse | 430.10 | 424.08 – 436.12 | *

* Degenerate case — FFT of δ(t) is a constant vector. The transform is mathematically exact; SQNR reflects FP64 numerical noise, not actual FFT precision.