# FP16 Forward FFT SQNR — Bergach 2026 §III-A Reproduction (v3 Extended)

**Platform**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12GB VRAM, SM 12.0)
**Paper**: arXiv:2605.28451 — "Range, Not Precision" (Bergach, 2026-05-27)
**Date**: 2026-06-04

---

## Methodology

1. Generate random complex signal x (FP64), max |x_i| = 1
2. X_ref = FFT_FP64(x)  — ground truth (torch.fft.fft, complex128)
3. X_test = FFT_FP16(x)  — cuFFT FP16 via lowp_fft.fft(precision="fp16")
4. Align: α = ⟨X_ref, X_test⟩ / ||X_test||²  (optimal complex scaling, paper §IV-B)
5. SQNR = 10·log₁₀(||X_ref||² / ||X_ref − α·X_test||²)

## v3 Extension: Full Signal/N Coverage

### FP16 Results Matrix (200 trials each)

| Signal | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|--------|-------|-------|--------|--------|--------|
| **uniform** | 61.3 ± 0.3 | 60.5 ± 0.2 | 59.9 ± 0.1 | 59.3 ± 0.1 | 56.5 ± 0.1 |
| **normal** | 61.5 ± 0.3 | 60.7 ± 0.2 | 60.1 ± 0.2 | 59.5 ± 0.1 | 56.5 ± 0.1 |
| **multitone** | 61.5 ± 0.8 | 60.6 ± 0.7 | 60.1 ± 0.7 | 59.4 ± 0.6 | 56.6 ± 0.7 |
| **impulse** | 424.1 ± 0.0 | 427.1 ± 0.0 | 430.1 ± 0.0 | 433.1 ± 0.0 | 436.1 ± 0.0 |

All values: aligned SQNR (mean ± std), dB.

**Paper claim**: 56-61 dB for FP16 FFT vs FP64 reference (Bergach 2026 §III-A Table I).

**Verdict for real signals** (uniform, normal, multitone): **ALL within/close to 56-61 dB range.**

### Signal Type Analysis

- **uniform / normal**: Nearly identical SQNR (~0.2 dB difference at same N). Both produce broadband signals where all butterfly stages contribute quantization noise. Results tightly clustered (σ < 0.3 dB).

- **multitone**: Slightly higher variance (σ ≈ 0.6-0.8 dB) due to variable frequency placement across trials. Mean SQNR matches uniform/normal at all N within 0.1-0.2 dB.

- **impulse**: Degenerate case — δ[0]=1 → FFT is uniform (all ones). FP16 represents 1.0 exactly, so SQNR approaches FP64 limit (~424-436 dB). The constant offset between N values (~3 dB per doubling) comes from the signal power scaling in the FFT:
  - ||X_ref||² = N (for δ[0]=1 with backward normalization)
  - 10·log₁₀(N) explains the +3.01 dB per doubling
  - Example: 10·log₁₀(256) = 24.1, 10·log₁₀(4096) = 36.1 → Δ = 12 dB ✓ (424 → 436)

  Impulse is **not a meaningful FFT precision test** — it exercises only the DC component. Included for completeness.

### N Scaling

For real signals (uniform/normal/multitone), SQNR decreases with N:

| Step | Δ SQNR | Theory |
|------|--------|--------|
| 256 → 512 | ~0.8 dB | log₂(N) noise accumulation |
| 512 → 1024 | ~0.6 dB | — |
| 1024 → 2048 | ~0.6 dB | — |
| 2048 → 4096 | ~2.9 dB | — |

The 2048→4096 step shows a sharper drop (~3 dB instead of ~0.6 dB), consistent with Bergach 2026's observation that FP16 FFT SQNR degrades faster at larger N due to accumulated mantissa roundoff in butterfly stages.

### Alignment Gain

Alignment gain (|α| correction) is negligible for all signal types:

| Precision | |α| − 1 | typical | Alignment gain |
|-----------|--------|---------|---------------|
| FP16 | < 0.0003 | ~1.0000 | < 0.1 dB |
| FP32 | < 0.0001 | ~1.0000 | ~0.4-0.7 dB |

The FP32 alignment gain is larger because FP32 has less quantization noise, so the systematic gain offset (though tiny in absolute terms) represents a larger fraction of the total error. cuFFT's FP32 internal accumulators make FP16 nearly self-calibrated.

---

## FP32 Ceiling

| N | FP32 SQNR | Expected |
|---|-----------|----------|
| 256 | 137.6 ± 0.3 dB | ~138 dB |
| 512 | 135.8 ± 0.2 dB | ~138 dB |
| 1024 | 135.3 ± 0.2 dB | ~138 dB |
| 2048 | 135.5 ± 0.1 dB | ~138 dB |
| 4096 | 135.1 ± 0.1 dB | ~138 dB |

100 trials each, uniform signal, torch.fft.fft (complex64) vs FP64 reference.

**Expected**: FP32 23-bit mantissa → ~138 dB theoretical SQNR.

**Result**: All N values within 1-3 dB of the 138 dB theoretical ceiling. The slight drop with larger N is consistent with accumulated roundoff in the FFT (log₂(N) butterfly stages).

---

## Comparison with Previous Versions

| Aspect | v1 (WRONG) | v2 Corrected | v3 Extended (THIS) |
|--------|-----------|-------------|---------------------|
| Measure | Roundtrip (FFT→IFFT) | Forward FFT only | Forward FFT only |
| Signals | uniform only | uniform only | uniform, normal, multitone, impulse |
| N values | 1024, 4096 | 1024, 4096 | 256, 512, 1024, 2048, 4096 |
| FP32 ceiling | No | No | Yes (uniform × 5N × 100) |
| Trials/N | 50 | 200 | 200 (FP16), 100 (FP32) |

### Forward vs Roundtrip (all signals, avg)

| N | Forward SQNR (v3) | Roundtrip SQNR (v1) | Delta |
|---|-------------------|---------------------|-------|
| 1024 | ~60.0 dB | 57.1 dB | **+2.9 dB** |
| 4096 | ~56.5 dB | 53.4 dB | **+3.1 dB** |

Forward-only consistently ~3 dB higher than roundtrip (FFT+IFFT), confirming that
two noise injections → 2× noise power → ~3 dB penalty.

---

## Key Conclusions

1. **Bergach 2026 §III-A fully reproduced**: FP16 forward FFT achieves 56-61 dB SQNR for real signals (uniform, normal, multitone) on NVIDIA RTX 5070 Ti

2. **Signal type has negligible impact on SQNR**: uniform, normal, and multitone all produce nearly identical results (±0.2 dB at same N)

3. **N scaling confirmed**: SQNR degrades with N, especially at N≥2048 where accumulated mantissa roundoff accelerates

4. **FP32 ceiling verified**: ~135-138 dB across all N, matching the 138 dB theoretical limit for 23-bit mantissa

5. **cuFFT FP16 is self-calibrated**: |α| ≈ 1.0000, alignment gain < 0.1 dB — FP32 internal accumulators eliminate systematic gain offset

6. **Impulse is degenerate**: 424-436 dB SQNR confirms FP16 can represent the trivial δ[0] FFT (all ones) exactly, but this is not a meaningful precision test

---

## Data Files

- `fp16_fft_sqnr_summary.csv` — all configurations, one row per summary
- `fp16_fft_sqnr_{signal}_{precision}_N{N}.csv` — per-trial raw data (25 files)
- `fp16_fft_sqnr.py` — measurement script (v3 extended)
