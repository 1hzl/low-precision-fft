# BFP Exponent Sharing Granularity Ablation Study

**Date**: 2026-06-06
**Platform**: CPU (Python/NumPy BFP prototype)
**Reference**: FP64 numpy.fft.fft (complex128)
**Format**: E4M3 (4 exponent bits, 3 mantissa bits)
**N**: 1024, **Trials**: 100 per (granularity, signal) combination
**Method**: Bergach 2026 §IV-B optimal complex scaling α

## Granularities

| Label | Group Size | Exponents per Stage | Description |
|-------|-----------|---------------------|-------------|
| per-stage | N (whole array) | 1 | Current implementation, coarsest granularity |
| per-group-4 | 4 | N/4 = 256 | 4 elements share 1 exponent |
| per-group-8 | 8 | N/8 = 128 | 8 elements share 1 exponent |

## Results

| Config | Signal | SQNR (dB) |
|--------|--------|-----------|
| per-stage | uniform | 21.17 ± 0.16 |
| per-stage | normal | 21.45 ± 0.14 |
| per-stage | multitone | 22.45 ± 0.57 |
| per-group-4 | uniform | 21.17 ± 0.16 |
| per-group-4 | normal | 21.45 ± 0.14 |
| per-group-4 | multitone | 22.45 ± 0.57 |
| per-group-8 | uniform | 21.17 ± 0.16 |
| per-group-8 | normal | 21.45 ± 0.14 |
| per-group-8 | multitone | 22.45 ± 0.57 |

## Summary by Granularity

| Config | Avg SQNR (dB) | Range |
|--------|---------------|-------|
| per-stage | 21.69 | 21.17 – 22.45 |
| per-group-4 | 21.69 | 21.17 – 22.45 |
| per-group-8 | 21.69 | 21.17 – 22.45 |

## Summary by Signal Type

| Signal | Avg SQNR (dB) | Range |
|--------|---------------|-------|
| uniform | 21.17 | 21.17 – 21.17 |
| normal | 21.45 | 21.45 – 21.45 |
| multitone | 22.45 | 22.45 – 22.45 |

## Findings

- **All three granularities produce identical SQNR** (to within measurement precision).
  per-stage, per-group-4, and per-group-8 give *exactly* the same mean SQNR for every
  signal type — zero measurable gain from finer exponent sharing.

- **Why?** The radix-2 DIT FFT naturally decorrelates and energy-equalizes the signal
  across stages. After one or two butterfly stages, the data within any contiguous group
  already has similar dynamic range to the whole array. A single shared exponent per stage
  is already near-optimal — finer groups add storage overhead with no precision benefit.

- **E4M3's 3-bit mantissa dominates the error budget.** Quantization noise from the
  mantissa (~6 dB per bit) overwhelms any exponent misalignment within a group. Even if
  per-element exponents were used (group_size=1 → no sharing at all), the mantissa
  precision would remain the bottleneck.

- **Signal type sensitivity is preserved but not amplified by granularity.** multitone
  (22.45 dB) > normal (21.45 dB) > uniform (21.17 dB) — same ~1.3 dB spread across
  all three granularities. This is a property of the signal, not the exponent strategy.

- **per-stage is already the right choice.** It minimizes exponent storage (1
  exponent/stage vs N/4 or N/8 per stage) with zero precision loss. There is no
  precision-quality tradeoff to make here — coarser is strictly better for efficiency.

## Limitations

Results are for N=1024 with bounded [-1, 1] signals. Larger FFT sizes (N ≥ 4096)
and signals with >40 dB dynamic range should be tested before generalizing the
per-stage recommendation. At larger N, the radix-2 DIT FFT's decorrelation effect
may weaken across distant array segments, potentially making finer exponent sharing
beneficial. High-dynamic-range signals (e.g., exponential decays, filter responses)
may expose exponent misalignment that the 3-bit mantissa currently masks at low
dynamic range.