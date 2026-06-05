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

- **Per-bit gain**: Each additional mantissa bit yields ~6.0 dB SQNR improvement
  (E4M3 vs E4M2: +5.8 dB; E4M4 vs E4M3: +6.0 dB). This matches theory:
  10·log₁₀(2²) ≈ 6.02 dB per bit of quantization precision.
- **E5M3 = E4M3**: For signals clamped to [-1, 1], the wider exponent range of
  E5M3 (bias=15, max 114688) provides zero benefit over E4M3 (bias=7, max 448).
  The quantization granularity is identical (3 mantissa bits), and the dynamic
  range of E4 already covers all signal values + FFT growth (N=1024 → ~32×).
- **multitone consistently highest**: Across all configs, multitone signals score
  ~1 dB above uniform/normal, likely due to spectral concentration reducing
  quantization error accumulation across stages.
- **E4M3 is the sweet spot** for N≤4096, [-1,1] signals — it matches E5M3 in
  precision while using 1 fewer bit (8-bit vs 9-bit). For high-dynamic-range
  signals or very large N where FFT growth exceeds E4 range, E5M3 would pull
  ahead.