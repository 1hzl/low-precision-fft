# Bergach 2026 Corrected Experiment 1 — FP16 FFT SQNR

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the previous roundtrip-based measurement and reproduce Bergach 2026 paper §III-A Table I: single-pass FP16 forward FFT SQNR vs FP64 reference, N=1024/4096, 200 trials.

**Architecture:** Pure Python/PyTorch script using existing `lowp_fft.fft(precision="fp16")` for FP16 FFT and `torch.fft.fft` with `torch.complex128` for FP64 reference. Signal generated once in FP64, then cast to FP32 for the FP16 FFT path (cuFFT requires FP32 input, internally converts to FP16 compute). Amplitude alignment from paper §IV-B applied before SQNR computation to separate systematic gain error from quantization noise.

**Key correction from previous experiment:** The old `fp16_bfp_sqnr.py` measured **roundtrip** SQNR (FFT→IFFT) with BFP trick. Paper §III-A Table I measures **single-pass forward FFT** SQNR: the FP16 FFT output is compared directly against an FP64 FFT reference, not a roundtrip reconstruction.

**Tech Stack:** Python 3.14 + PyTorch 2.9 + lowp_fft (pre-compiled cuFFT FP16 extension)

---

### Task 1: Write the corrected FP16 FFT SQNR experiment script

**Files:**
- Create: `experiments/bergach-repro/fp16_fft_sqnr.py`

- [ ] **Step 1: Write the complete experiment script**

```python
"""FP16 FFT SQNR -- Bergach 2026 Experiment 1 (Corrected).

CORRECTION from previous fp16_bfp_sqnr.py:
  - OLD (wrong): measured ROUNDTRIP SQNR (FFT + IFFT) with BFP trick
  - NEW (correct): measures SINGLE-PASS forward FFT SQNR vs FP64 reference
    This matches paper §III-A Table I, which reports FP16 Stockham FFT SQNR
    against double-precision reference, NOT roundtrip reconstruction.

Methodology (aligned with Bergach 2026 §III-A):
  1. Generate random complex signal x (FP64)
  2. Compute reference: X_ref = FFT_FP64(x)
  3. Compute test:     X_test = FFT_FP16(x_fp32)  -- cuFFT FP16 via lowp_fft
  4. Align amplitudes:  α = argmin ||X_ref - α·X_test||²  (paper §IV-B)
  5. SQNR = 10·log₁₀(||X_ref||² / ||X_ref - α·X_test||²)

N values: 1024, 4096 (paper Table I)
Trials: 200 random complex signals each
"""

import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lowp_fft import fft as fft_lowp


def sqnr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Signal-to-Quantization-Noise Ratio in dB.

    SQNR = 10 * log10(||ref||² / ||ref - test||²)

    Args:
        ref: Reference tensor (FP64 complex).
        test: Test tensor (cast to FP64 for subtraction).
    """
    signal_power = ref.abs().pow(2).sum().item()
    error = (ref - test.to(torch.complex128)).abs().pow(2).sum().item()
    return float(10.0 * np.log10(signal_power / max(error, 1e-40)))


def align_optimal_scale(ref: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
    """Find optimal complex scalar α minimizing ||ref - α·test||².

    Closed-form solution: α = ⟨ref, test⟩ / ||test||²
    where ⟨a,b⟩ = sum(a · conj(b)).

    This separates systematic gain/phase offset from random quantization noise,
    as described in Bergach 2026 §IV-B ("align amplitudes before computing SQNR").

    Args:
        ref: Reference tensor (FP64 complex128).
        test: Test tensor (any complex dtype, cast to complex128 internally).

    Returns:
        Aligned test tensor: α · test (complex128).
    """
    test_c128 = test.to(torch.complex128)
    # α = sum(ref * conj(test)) / sum(|test|²)
    inner = (ref * test_c128.conj()).sum()
    norm_test = test_c128.abs().pow(2).sum()
    alpha = inner / norm_test.clamp(min=1e-40)
    return alpha * test_c128


def run_trials(N: int, n_trials: int = 200, device: str = "cuda") -> dict:
    """Run SQNR measurement for a single N value.

    For each trial:
      1. Generate random complex signal x (|x_i| <= 1, uniform in [-1,1]^2)
      2. X_ref = FP64 FFT(x_fp64) using torch.fft.fft
      3. X_fp16 = FP16 FFT(x_fp32) using lowp_fft.fft(precision="fp16")
      4. Compute raw SQNR (no alignment) and aligned SQNR (with α scaling)

    Returns dict with per-trial SQNR lists.
    """
    torch.manual_seed(42)
    results = {
        "N": N,
        "raw_sqnr": [],
        "aligned_sqnr": [],
        "alpha_magnitudes": [],
    }

    for trial in range(n_trials):
        # Generate signal in FP64 (double precision) for reference
        g = torch.Generator(device="cpu")  # CPU generator for reproducibility
        g.manual_seed(trial * 10007 + N)
        real = torch.rand(N, generator=g, dtype=torch.float64) * 2.0 - 1.0
        imag = torch.rand(N, generator=g, dtype=torch.float64) * 2.0 - 1.0
        x_fp64 = torch.complex(real, imag)
        # Normalize max |element| to 1.0
        x_fp64 = x_fp64 / x_fp64.abs().max().clamp(min=1e-40)

        with torch.no_grad():
            # Reference: FP64 FFT (ground truth)
            x_fp64_gpu = x_fp64.to(device=device, dtype=torch.complex128)
            X_ref = torch.fft.fft(x_fp64_gpu, norm="backward")

            # Test: FP16 FFT via cuFFT extension
            # cuFFT FP16 path requires FP32 input (converts internally to FP16 compute)
            x_fp32 = x_fp64.to(dtype=torch.complex64).to(device)
            X_fp16 = fft_lowp(x_fp32, precision="fp16")

            # Compute SQNR (raw and aligned)
            raw_snr = sqnr_db(X_ref, X_fp16)

            X_aligned = align_optimal_scale(X_ref, X_fp16)
            aligned_snr = sqnr_db(X_ref, X_aligned)
            alpha_mag = float((X_aligned.abs().sum() / X_fp16.abs().sum().clamp(min=1e-40)).cpu())

        results["raw_sqnr"].append(raw_snr)
        results["aligned_sqnr"].append(aligned_snr)
        results["alpha_magnitudes"].append(alpha_mag)

        if (trial + 1) % 50 == 0:
            print(f"  {trial + 1}/{n_trials} trials done  "
                  f"[raw={raw_snr:.1f}, aligned={aligned_snr:.1f} dB]")

    return results


def summarize(results: dict) -> dict:
    """Compute summary statistics across trials."""
    raw = np.array(results["raw_sqnr"])
    aligned = np.array(results["aligned_sqnr"])
    alphas = np.array(results["alpha_magnitudes"])
    return {
        "N": results["N"],
        "raw_sqnr_mean": float(np.mean(raw)),
        "raw_sqnr_std": float(np.std(raw)),
        "raw_sqnr_min": float(np.min(raw)),
        "raw_sqnr_max": float(np.max(raw)),
        "aligned_sqnr_mean": float(np.mean(aligned)),
        "aligned_sqnr_std": float(np.std(aligned)),
        "aligned_sqnr_min": float(np.min(aligned)),
        "aligned_sqnr_max": float(np.max(aligned)),
        "alignment_gain_db": float(np.mean(aligned) - np.mean(raw)),
        "alpha_mean": float(np.mean(alphas)),
        "alpha_std": float(np.std(alphas)),
        "n_trials": len(raw),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("ERROR: CUDA required for cuFFT FP16 FFT. Exiting.")
        sys.exit(1)

    print("=" * 72)
    print("Bergach 2026 Experiment 1 (Corrected): FP16 FFT SQNR")
    print("=" * 72)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Paper claim (§III-A Table I): FP16 FFT mantissa-limited at")
    print(f"  56-61 dB SQNR (N=1024, 4096, vs double-precision reference)")
    print(f"Previous measurement was WRONG: measured roundtrip, not forward FFT")
    print()

    N_values = [1024, 4096]
    n_trials = 200
    all_summaries = []

    for N in N_values:
        print(f"--- N={N} ({n_trials} trials) ---")
        t0 = time.perf_counter()
        results = run_trials(N, n_trials, device)
        t1 = time.perf_counter()

        summary = summarize(results)
        all_summaries.append(summary)

        print(f"  Raw SQNR:          {summary['raw_sqnr_mean']:.1f}"
              f" +/- {summary['raw_sqnr_std']:.1f} dB")
        print(f"  Aligned SQNR:      {summary['aligned_sqnr_mean']:.1f}"
              f" +/- {summary['aligned_sqnr_std']:.1f} dB")
        print(f"  Alignment gain:    {summary['alignment_gain_db']:+.1f} dB")
        print(f"  Alpha (opt scale): {summary['alpha_mean']:.6f}"
              f" +/- {summary['alpha_std']:.6f}")
        print(f"  Time: {t1 - t0:.1f}s")
        print()

        # Save per-trial raw data
        out_dir = "experiments/bergach-repro"
        os.makedirs(out_dir, exist_ok=True)
        csv_path = f"{out_dir}/fp16_fft_sqnr_N{N}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial", "raw_sqnr_db", "aligned_sqnr_db", "alpha_mag"])
            for i in range(n_trials):
                w.writerow([
                    i,
                    results["raw_sqnr"][i],
                    results["aligned_sqnr"][i],
                    results["alpha_magnitudes"][i],
                ])
        print(f"  Raw data -> {csv_path}\n")

    # Summary CSV
    summary_path = "experiments/bergach-repro/fp16_fft_sqnr_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        keys = list(all_summaries[0].keys())
        w.writerow(keys)
        for s in all_summaries:
            w.writerow([s[k] for k in keys])
    print(f"Summary -> {summary_path}")

    # Comparison with paper
    print()
    print("=" * 72)
    print("Comparison: cuFFT FP16 Forward FFT SQNR vs Bergach 2026 §III-A")
    print("=" * 72)
    header = (f"{'N':>6s}  {'Raw SQNR':>14s}  {'Aligned SQNR':>14s}  "
              f"{'Paper §III-A':>14s}  {'Verdict':>20s}")
    print(header)
    print("-" * 72)
    for s in all_summaries:
        # Use aligned SQNR as primary metric (matches paper methodology)
        snr = s["aligned_sqnr_mean"]
        if 56 <= snr <= 61:
            verdict = "MATCHES paper"
        elif snr >= 53:
            verdict = f"CLOSE (within 3 dB)"
        else:
            verdict = "BELOW paper"
        print(f"{s['N']:6d}  {s['raw_sqnr_mean']:13.1f} dB  "
              f"{s['aligned_sqnr_mean']:13.1f} dB  "
              f"{'56-61 dB':>14s}  {verdict:>20s}")
    print("-" * 72)

    # Verdict
    print()
    aligned_snrs = [s["aligned_sqnr_mean"] for s in all_summaries]
    if all(56 <= snr <= 61 for snr in aligned_snrs):
        print("VERDICT: MATCHES paper (56-61 dB) on all sizes")
    elif all(snr >= 56 for snr in aligned_snrs):
        print("VERDICT: EXCEEDS paper lower bound (>= 56 dB)")
    elif all(snr >= 53 for snr in aligned_snrs):
        print("VERDICT: CLOSE to paper — within ~3 dB")
        print("  Difference likely due to: cuFFT (Cooley-Tukey) vs paper (Stockham)")
        print("  and NVIDIA FP16 rounding vs Apple M1 FP16 rounding")
    else:
        print("VERDICT: DOES NOT MATCH paper — significantly lower SQNR")

    # Key insight
    print()
    print("=" * 72)
    print("Key Difference from Previous (Incorrect) Experiment")
    print("=" * 72)
    print(f"  Previous: ROUNDTRIP SQNR = {53.4-57.1:.0f} dB  (FFT + IFFT, double penalty)")
    print(f"  Corrected: FORWARD SQNR = {aligned_snrs[0]:.0f}-{aligned_snrs[-1]:.0f} dB")
    print(f"  (FFT only vs FP64 reference)")
    print()
    print("The roundtrip includes both FFT and IFFT error, roughly doubling")
    print("the noise power and reducing SQNR by ~3 dB compared to forward-only.")
    print("Paper §III-A measures forward FFT only, matching this corrected approach.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses without syntax errors**

Run: `python -c "import ast; ast.parse(open('experiments/bergach-repro/fp16_fft_sqnr.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Dry-run with 2 trials to validate methodology**

Run: `python -c "
import sys, os
sys.path.insert(0, '.')
# Quick validation: N=1024, 2 trials
exec(open('experiments/bergach-repro/fp16_fft_sqnr.py').read().replace('n_trials = 200', 'n_trials = 2').replace('N_values = [1024, 4096]', 'N_values = [1024]'))
"`
Expected: SQNR ~55-61 dB for N=1024, script runs without errors.

---

### Task 2: Run full experiment

- [ ] **Step 1: Execute the full 200-trial experiment**

Run: `python experiments/bergach-repro/fp16_fft_sqnr.py`
Expected: Raw and aligned SQNR for N=1024 and N=4096, 200 trials each. Data saved to CSV files.

- [ ] **Step 2: Verify output files exist**

Run: `ls experiments/bergach-repro/fp16_fft_sqnr_*.csv`
Expected: Four CSV files:
- `fp16_fft_sqnr_N1024.csv` (200 rows)
- `fp16_fft_sqnr_N4096.csv` (200 rows)
- `fp16_fft_sqnr_summary.csv`

---

### Task 3: Write experiment report

**Files:**
- Create: `experiments/bergach-repro/fp16-fft-sqnr.md`

- [ ] **Step 1: Write the report**

```markdown
# FP16 FFT SQNR — Bergach 2026 §III-A Reproduction (Corrected)

**Platform**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12GB VRAM, SM 12.0)
**Paper**: arXiv:2605.28451 — "Range, Not Precision" (Bergach, 2026-05-27)
**Date**: 2026-06-04

---

## Correction from Previous Experiment

| Aspect | Previous (WRONG) | Corrected (THIS) |
|--------|-----------------|------------------|
| Measure | Roundtrip SQNR (FFT→IFFT) | Forward FFT SQNR only |
| Reference | FP32 roundtrip | FP64 (double) FFT |
| BFP trick | Applied (but not relevant) | Not applied |
| Alignment | None | Optimal complex scaling (paper §IV-B) |
| Matches paper § | §IV (SAR pipeline) | **§III-A (Table I)** |

The previous `fp16_bfp_sqnr.py` measured roundtrip reconstruction error,
which combines both FFT and IFFT quantization noise. Paper §III-A Table I
measures **single-pass forward FFT** SQNR against double-precision reference.

## Method

1. Generate random complex signal x ∈ [-1,1]² (FP64)
2. X_ref = FFT_FP64(x)  — ground truth
3. X_test = FFT_FP16(x)  — cuFFT FP16 via lowp_fft.fft(precision="fp16")
4. Align: α = ⟨X_ref, X_test⟩ / ||X_test||²  (optimal complex scaling)
5. SQNR = 10·log₁₀(||X_ref||² / ||X_ref − α·X_test||²)

**Signal**: Random complex uniform in [-1,1]², 200 trials
**Reference**: torch.fft.fft with torch.complex128 (FP64)
**Test**: lowp_fft.fft(precision="fp16") via cuFFT Xt API with CUDA_C_16F

## Results

| N | Raw SQNR | Aligned SQNR | Paper §III-A | Verdict |
|---|---------|-------------|-------------|---------|
| 1024 | [TBD] dB | [TBD] dB | 56-61 dB | [TBD] |
| 4096 | [TBD] dB | [TBD] dB | 56-61 dB | [TBD] |

## Analysis

### 1. Raw vs Aligned SQNR

The alignment gain ([TBD] dB) represents the systematic amplitude/phase offset
between cuFFT FP16 and FP64 FFT. This is expected because:
- cuFFT uses FP32 internal accumulators, producing slightly different scaling
- FP16 twiddle factor quantization shifts effective gain

### 2. Comparison with Previous Roundtrip Measurement

| N | Forward SQNR (new) | Roundtrip SQNR (old) | Delta |
|---|-------------------|---------------------|-------|
| 1024 | [TBD] dB | 57.1 dB | [TBD] dB |
| 4096 | [TBD] dB | 53.4 dB | [TBD] dB |

The forward SQNR should be ~3 dB higher than roundtrip SQNR (half the noise power).

### 3. Comparison with Bergach 2026 (Apple M1)

[TBD — fill after results]

## Key Conclusions

[TBD — fill after results]

## Data Files

- Per-trial: `experiments/bergach-repro/fp16_fft_sqnr_N1024.csv`
- Per-trial: `experiments/bergach-repro/fp16_fft_sqnr_N4096.csv`
- Summary: `experiments/bergach-repro/fp16_fft_sqnr_summary.csv`
```

- [ ] **Step 2: Fill in the [TBD] values after running the experiment**

After Task 2 completes, update the report with actual measured values.

---

### Task 4: Update LAPTOP-CHANGES.md

**Files:**
- Modify: `LAPTOP-CHANGES.md`

- [ ] **Step 1: Prepend new entry at top of LAPTOP-CHANGES.md**

Insert after the `## 2026-06-04: Bergach 2026 Reproduction` heading line, before the "### Experiment 1" subsection:

```markdown

### Experiment 1 (CORRECTED) — FP16 Forward FFT SQNR ✅

- [x] **CORRECTION**: Previous Exp 1 measured ROUNDTRIP (FFT+IFFT) — WRONG
- [x] Paper §III-A Table I measures SINGLE-PASS forward FFT SQNR vs FP64 reference
- [x] Implemented amplitude alignment (paper §IV-B): optimal complex scaling α
- [x] Measured SQNR for N=1024, 4096 (200 trials each) via cuFFT FP16 extension
- [x] Results:
  - N=1024: [TBD] dB (raw) / [TBD] dB (aligned) — paper: 56-61 dB
  - N=4096: [TBD] dB (raw) / [TBD] dB (aligned) — paper: 56-61 dB
- [x] Script: `experiments/bergach-repro/fp16_fft_sqnr.py`
- [x] Report: `experiments/bergach-repro/fp16-fft-sqnr.md`
- [x] Data: `experiments/bergach-repro/fp16_fft_sqnr_N{1024,4096}.csv`
```

---

### Task 5: Commit and push

- [ ] **Step 1: Stage all new and modified files**

```bash
git add experiments/bergach-repro/fp16_fft_sqnr.py
git add experiments/bergach-repro/fp16-fft-sqnr.md
git add experiments/bergach-repro/fp16_fft_sqnr_N1024.csv
git add experiments/bergach-repro/fp16_fft_sqnr_N4096.csv
git add experiments/bergach-repro/fp16_fft_sqnr_summary.csv
git add LAPTOP-CHANGES.md
git add docs/superpowers/plans/2026-06-04-bergach-corrected-exp1.md
```

- [ ] **Step 2: Commit with descriptive message**

```bash
git commit -m "fix(low-precision-fft): correct Bergach Exp 1 — forward FFT SQNR not roundtrip

Previous experiment incorrectly measured roundtrip (FFT+IFFT) SQNR with
BFP trick. Paper §III-A Table I measures single-pass forward FFT SQNR vs
FP64 reference. This commit adds the corrected measurement with amplitude
alignment from paper §IV-B."
```

- [ ] **Step 3: Push**

```bash
git push origin master
```

---

## Self-Review

### 1. Spec coverage
- [x] FP16 FFT SQNR measurement (single-pass forward, not roundtrip) — Task 1
- [x] FP64 (double) reference — Task 1 (torch.complex128)
- [x] cuFFT FP16 extension usage (`lowp_fft.fft(precision="fp16")`) — Task 1
- [x] SQNR formula: 10·log₁₀(||X_ref||² / ||X_fp16 − X_ref||²) — Task 1
- [x] Amplitude alignment (paper §IV-B) — Task 1 (align_optimal_scale)
- [x] N=1024, 4096 — Task 1
- [x] 200 trials — Task 1
- [x] Data saved to experiments/bergach-repro/ — Task 1
- [x] Report — Task 3
- [x] LAPTOP-CHANGES.md update — Task 4

### 2. Placeholder scan
- No TBD/TODO/implement later in code steps — Tasks 1-2 have complete code
- Report (Task 3) has [TBD] markers that get filled after experiment runs
- No "add appropriate error handling" — code has explicit error handling
- No "similar to Task N" — each task is self-contained

### 3. Type consistency
- `sqnr_db(ref: torch.Tensor, test: torch.Tensor) -> float` — used consistently
- `align_optimal_scale(ref, test) -> torch.Tensor` — returns complex128
- `run_trials(N, n_trials, device) -> dict` — returns dict with raw_sqnr, aligned_sqnr, alpha_magnitudes
- `summarize(results) -> dict` — consumes run_trials output format
- All CSV keys match across write/read
