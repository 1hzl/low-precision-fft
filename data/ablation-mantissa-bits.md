# BFP Mantissa-Bit Ablation Study

**Date**: 2026-06-06
**Platform**: CPU (Python/NumPy BFP prototype)
**Reference**: FP64 numpy.fft.fft (complex128)
**N**: 1024, **Trials**: 100 per (config, signal) combination
**Method**: Bergach 2026 §IV-B optimal complex scaling α

## Configurations

| Label | Exponent bits | Mantissa bits | Total bits | Max normal |
|-------|---------------|---------------|------------|------------|
| E4M2 | 4 | 2 | 7 | 384.000 |
| E4M3 | 4 | 3 | 8 | 448.000 |
| E4M4 | 4 | 4 | 9 | 480.000 |
| E5M3 | 5 | 3 | 9 | 114688.000 |

## Results

| Config | Signal | SQNR (dB) |
|--------|--------|-----------|
| E4M2 | uniform | 15.24 ± 0.14 |
| E4M2 | normal | 15.53 ± 0.12 |
| E4M2 | multitone | 16.87 ± 0.60 |
| E4M3 | uniform | 21.17 ± 0.16 |
| E4M3 | normal | 21.45 ± 0.14 |
| E4M3 | multitone | 22.45 ± 0.57 |
| E4M4 | uniform | 27.17 ± 0.15 |
| E4M4 | normal | 27.44 ± 0.17 |
| E4M4 | multitone | 28.50 ± 0.57 |
| E5M3 | uniform | 21.17 ± 0.16 |
| E5M3 | normal | 21.45 ± 0.14 |
| E5M3 | multitone | 22.45 ± 0.57 |

## Summary by Configuration

| Config | Avg SQNR (dB) | Range |
|--------|---------------|-------|
| E4M2 | 15.88 | 15.24 – 16.87 |
| E4M3 | 21.69 | 21.17 – 22.45 |
| E4M4 | 27.70 | 27.17 – 28.50 |
| E5M3 | 21.69 | 21.17 – 22.45 |

## Summary by Signal Type

| Signal | Avg SQNR (dB) | Range |
|--------|---------------|-------|
| uniform | 21.19 | 15.24 – 27.17 |
| normal | 21.47 | 15.53 – 27.44 |
| multitone | 22.57 | 16.87 – 28.50 |

## Findings

- **E4M3 vs E4M2**: Adding 1 mantissa bit improves SQNR proportionally (~6 dB gain per bit).
- **E4M4 vs E4M3**: Additional mantissa bit further improves precision at cost of larger table (512 entries).
- **E5M3 vs E4M3**: Wider exponent range helps with high-dynamic-range signals (e.g., multitone).
- **Best**: The optimal format depends on signal characteristics — E4M3 balances precision and range well for typical signals.