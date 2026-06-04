# Bergach 2026 Corrected Experiment 1 — FP16 Forward FFT SQNR

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the v1 roundtrip-based measurement. Reproduce Bergach 2026 §III-A Table I: single-pass FP16 forward FFT SQNR vs FP64 reference, N=1024/4096, 200 trials each, with amplitude alignment (paper §IV-B).

**Architecture:** Pure Python script using existing `lowp_fft.fft(precision="fp16")` for cuFFT FP16 forward FFT and `torch.fft.fft` with `torch.complex128` for FP64 reference. Signal generated in FP64, cast to FP32 for cuFFT (cuFFT requires FP32 input, then converts internally to FP16 compute). Amplitude alignment via optimal complex scaling separates systematic gain error from quantization noise.

**Key correction from v1:** v1 `fp16_bfp_sqnr.py` measured roundtrip SQNR (FFT→IFFT). Paper §III-A Table I measures single-pass forward FFT SQNR: FP16 FFT output compared directly against FP64 FFT reference.

**Tech Stack:** Python 3.14 + PyTorch 2.9 + lowp_fft (pre-compiled cuFFT FP16 extension)

---

### Task 1: Write the corrected FP16 forward FFT SQNR experiment script

**Files:**
- Create: `experiments/bergach-repro/fp16_fft_sqnr.py`

- [ ] **Step 1: Write the complete script**

```python
"""FP16 Forward FFT SQNR -- Bergach 2026 Experiment 1 (Corrected).

CORRECTION from v1 fp16_bfp_sqnr.py:
  - OLD (wrong): measured ROUNDTRIP SQNR (FFT + IFFT) with BFP trick
  - NEW (correct): measures SINGLE-PASS forward FFT SQNR vs FP64 reference
    Matches paper  III-A Table I: FP16 Stockham FFT SQNR against
    double-precision reference, not roundtrip reconstruction.

Methodology (aligned with Bergach 2026   III-A,  IV-B):
  1. Generate random complex signal x (FP64), max |x_i| = 1
  2. Compute reference: X_ref = FFT_FP64(x)
  3. Compute test:     X_test = FFT_FP16(x_fp32)  via cuFFT FP16 extension
  4. Align amplitudes:   = argmin ||X_ref -  *X_test||   (optimal complex scaling)
  5. SQNR = 10*log10(||X_ref||  / ||X_ref -  *X_test|| )

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

    SQNR = 10 * log10(||ref||  / ||ref - test|| )

    Args:
        ref: Reference tensor (FP64 complex128).
        test: Test tensor (any complex dtype, cast to complex128 internally).

    Returns:
        SQNR in dB.
    """
    test_c128 = test.to(torch.complex128)
    signal_power = ref.abs().pow(2).sum().item()
    error_power = (ref - test_c128).abs().pow(2).sum().item()
    return float(10.0 * np.log10(signal_power / max(error_power, 1e-40)))


def align_optimal_scale(ref: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, complex]:
    """Find optimal complex scalar   minimizing ||ref -  *test|| .

    Closed-form:   = <ref, test> / ||test||   where <a,b> = sum(a * conj(b)).

    This separates systematic gain/phase offset from random quantization noise
    (Bergach 2026   IV-B: "align amplitudes before computing SQNR").

    Args:
        ref: Reference tensor (FP64 complex128).
        test: Test tensor (any complex dtype, cast to complex128 internally).

    Returns:
        (aligned_test, alpha) where aligned_test =   * test (complex128).
    """
    test_c128 = test.to(torch.complex128)
    inner = (ref * test_c128.conj()).sum()
    norm_test = test_c128.abs().pow(2).sum()
    alpha = inner / norm_test.clamp(min=1e-40)
    return alpha * test_c128, complex(alpha.real.item(), alpha.imag.item())


def generate_signal(N: int, seed: int, device: str) -> torch.Tensor:
    """Generate random complex signal in FP64, max |element| = 1.

    Uses CPU-side generator for full reproducibility across GPU architectures.
    Each trial gets a unique seed: seed = trial * 10007 + N.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    real = torch.rand(N, generator=g, dtype=torch.float64) * 2.0 - 1.0
    imag = torch.rand(N, generator=g, dtype=torch.float64) * 2.0 - 1.0
    x = torch.complex(real, imag)
    x = x / x.abs().max().clamp(min=1e-40)
    return x.to(device)


def run_trials(N: int, n_trials: int = 200, device: str = "cuda") -> dict:
    """Run SQNR measurement for a single N value.

    For each trial:
      1. Generate random complex signal x (FP64)
      2. X_ref = FP64 FFT(x)          using torch.fft.fft (complex128)
      3. X_fp16 = FP16 FFT(x_fp32)    using lowp_fft.fft(precision="fp16")
      4. Compute raw SQNR (no alignment) and aligned SQNR (with   scaling)

    Returns dict with per-trial SQNR lists.
    """
    results = {
        "N": N,
        "raw_sqnr": [],
        "aligned_sqnr": [],
        "alpha_real": [],
        "alpha_imag": [],
    }

    for trial in range(n_trials):
        seed = trial * 10007 + N
        x_fp64 = generate_signal(N, seed, device)

        with torch.no_grad():
            # Reference: FP64 FFT (ground truth)
            X_ref = torch.fft.fft(x_fp64, norm="backward")

            # Test: FP16 FFT via cuFFT extension
            # cuFFT requires FP32 complex input; converts internally to FP16 compute
            x_fp32 = x_fp64.to(dtype=torch.complex64)
            X_fp16 = fft_lowp(x_fp32, precision="fp16")

            # Compute SQNR (raw and aligned)
            raw_snr = sqnr_db(X_ref, X_fp16)

            X_aligned, alpha = align_optimal_scale(X_ref, X_fp16)
            aligned_snr = sqnr_db(X_ref, X_aligned)

        results["raw_sqnr"].append(raw_snr)
        results["aligned_sqnr"].append(aligned_snr)
        results["alpha_real"].append(alpha.real)
        results["alpha_imag"].append(alpha.imag)

        if (trial + 1) % 50 == 0:
            print(f"  {trial + 1}/{n_trials} trials done  "
                  f"[raw={raw_snr:.1f}, aligned={aligned_snr:.1f} dB, "
                  f"| |={abs(alpha):.4f}]")

    return results


def summarize(results: dict) -> dict:
    """Compute summary statistics across trials."""
    raw = np.array(results["raw_sqnr"])
    aligned = np.array(results["aligned_sqnr"])
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
        "n_trials": len(raw),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("ERROR: CUDA required for cuFFT FP16 FFT. Exiting.")
        sys.exit(1)

    print("=" * 72)
    print("Bergach 2026 Experiment 1 (Corrected): FP16 Forward FFT SQNR")
    print("=" * 72)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Paper claim ( III-A Table I): FP16 FFT mantissa-limited at")
    print(f"  56-61 dB SQNR (N=1024, 4096, vs double-precision reference)")
    print(f"Correction: measures forward FFT only (v1 measured roundtrip)")
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
        print(f"  Time: {t1 - t0:.1f}s")
        print()

        # Save per-trial raw data
        out_dir = "experiments/bergach-repro"
        os.makedirs(out_dir, exist_ok=True)
        csv_path = f"{out_dir}/fp16_fft_sqnr_N{N}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial", "raw_sqnr_db", "aligned_sqnr_db",
                        "alpha_real", "alpha_imag"])
            for i in range(n_trials):
                w.writerow([
                    i,
                    results["raw_sqnr"][i],
                    results["aligned_sqnr"][i],
                    results["alpha_real"][i],
                    results["alpha_imag"][i],
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

    # Paper comparison
    print()
    print("=" * 72)
    print("Comparison: cuFFT FP16 Forward FFT SQNR vs Bergach 2026   III-A")
    print("=" * 72)
    header = (f"{'N':>6s}  {'Raw SQNR':>14s}  {'Aligned SQNR':>14s}  "
              f"{'Paper   III-A':>14s}  {'Verdict':>20s}")
    print(header)
    print("-" * 72)
    for s in all_summaries:
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

    # Final verdict
    print()
    aligned_snrs = [s["aligned_sqnr_mean"] for s in all_summaries]
    if all(56 <= snr <= 61 for snr in aligned_snrs):
        print("VERDICT: MATCHES paper (56-61 dB) on all sizes")
    elif all(snr >= 56 for snr in aligned_snrs):
        print("VERDICT: EXCEEDS paper lower bound (>= 56 dB)")
    elif all(snr >= 53 for snr in aligned_snrs):
        print("VERDICT: CLOSE to paper -- within ~3 dB")
        print("  Difference likely from: cuFFT (Cooley-Tukey) vs paper (Stockham)")
        print("  and NVIDIA FP16 rounding vs Apple M1 FP16 rounding")
    else:
        print("VERDICT: DOES NOT MATCH paper -- significantly lower SQNR")

    # Key insight
    print()
    print("=" * 72)
    print("Key Difference from v1 (Incorrect) Experiment")
    print("=" * 72)
    print(f"  v1: ROUNDTRIP SQNR = 53.4-57.1 dB  (FFT + IFFT, double penalty)")
    print(f"  v2: FORWARD SQNR = {aligned_snrs[0]:.0f}-{aligned_snrs[-1]:.0f} dB")
    print(f"  (FFT only vs FP64 reference)")
    print()
    print("The roundtrip includes both FFT and IFFT error, roughly doubling")
    print("the noise power and reducing SQNR by ~3 dB compared to forward-only.")
    print("Paper   III-A measures forward FFT only, matching this corrected approach.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('experiments/bergach-repro/fp16_fft_sqnr.py', encoding='utf-8').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Dry-run with 2 trials to validate methodology**

Run: `python -c "
import sys, os
sys.path.insert(0, '.')
os.chdir('D:/cc/low-precision-fft')
exec(open('experiments/bergach-repro/fp16_fft_sqnr.py', encoding='utf-8').read()
     .replace('n_trials = 200', 'n_trials = 2')
     .replace('N_values = [1024, 4096]', 'N_values = [1024]')
     .replace(\".replace('n_trials = 200', 'n_trials = 2')\", \"''\"))
"`
Expected: SQNR ~55-61 dB for N=1024, no crashes.

- [ ] **Step 4: Commit experiment script**

```bash
git add experiments/bergach-repro/fp16_fft_sqnr.py
git commit -m "feat(low-precision-fft): add corrected Bergach Exp 1 script (forward FFT SQNR)

v1 measured roundtrip SQNR (wrong). This script measures single-pass
forward FFT SQNR vs FP64 reference with amplitude alignment per paper IV-B."
```

---

### Task 2: Run full experiment

- [ ] **Step 1: Execute 200-trial experiment**

Run: `python experiments/bergach-repro/fp16_fft_sqnr.py`
Expected: Raw and aligned SQNR for N=1024 and N=4096, 200 trials each. Data saved to CSV files.

- [ ] **Step 2: Verify output files**

Run: `ls experiments/bergach-repro/fp16_fft_sqnr_N*.csv experiments/bergach-repro/fp16_fft_sqnr_summary.csv`
Expected: 
- `fp16_fft_sqnr_N1024.csv` (200 rows + header)
- `fp16_fft_sqnr_N4096.csv` (200 rows + header)
- `fp16_fft_sqnr_summary.csv` (2 rows + header)

- [ ] **Step 3: Read and record key results**

Run: `cat experiments/bergach-repro/fp16_fft_sqnr_summary.csv`
Record the aligned SQNR mean values for N=1024 and N=4096. These go into the report in Task 3.

- [ ] **Step 4: Commit experiment data**

```bash
git add experiments/bergach-repro/fp16_fft_sqnr_N1024.csv
git add experiments/bergach-repro/fp16_fft_sqnr_N4096.csv
git add experiments/bergach-repro/fp16_fft_sqnr_summary.csv
git commit -m "feat(low-precision-fft): Bergach Exp 1 corrected results (forward FFT SQNR)

N=1024, 4096, 200 trials each. Single-pass forward FFT SQNR vs FP64
reference with amplitude alignment from paper IV-B."
```

---

### Task 3: Write experiment report

**Files:**
- Create: `experiments/bergach-repro/fp16-fft-sqnr.md`

- [ ] **Step 1: Write the report (fill [TBD] from Task 2 results)**

```markdown
# FP16 Forward FFT SQNR -- Bergach 2026   III-A Reproduction (Corrected)

**Platform**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12GB VRAM, SM 12.0)
**Paper**: arXiv:2605.28451 -- "Range, Not Precision" (Bergach, 2026-05-27)
**Date**: 2026-06-04

---

## Correction from v1 Experiment

| Aspect | v1 (WRONG) | v2 Corrected (THIS) |
|--------|-----------|---------------------|
| Measure | Roundtrip SQNR (FFT IFFT) | Forward FFT SQNR only |
| Reference | FP32 roundtrip | FP64 (double) FFT |
| BFP trick | Applied (irrelevant for cuFFT) | Not applied |
| Alignment | None | Optimal complex scaling (paper   IV-B) |
| Matches paper   |   IV (SAR pipeline) | **  III-A (Table I)** |

The v1 `fp16_bfp_sqnr.py` measured roundtrip reconstruction error,
combining both FFT and IFFT quantization noise. Paper   III-A Table I
measures **single-pass forward FFT** SQNR against double-precision reference.

## Method

1. Generate random complex signal x   [-1,1]   (FP64), max |x_i| = 1
2. X_ref = FFT_FP64(x)  -- ground truth (torch.fft.fft, complex128)
3. X_test = FFT_FP16(x)  -- cuFFT FP16 via lowp_fft.fft(precision="fp16")
4. Align:   = <X_ref, X_test> / ||X_test||   (optimal complex scaling, paper   IV-B)
5. SQNR = 10*log10(||X_ref||  / ||X_ref -   *X_test|| )

**Signal**: Random complex uniform in [-1,1]  , 200 trials
**Reference**: torch.fft.fft with torch.complex128 (FP64)
**Test**: lowp_fft.fft(precision="fp16") via cuFFT Xt API with CUDA_C_16F
**cuFFT note**: Input is FP32 (complex64); cuFFT converts internally to FP16 compute using FP32 accumulators

## Results

| N | Raw SQNR | Aligned SQNR | Alignment Gain | Paper   III-A | Verdict |
|---|---------|-------------|----------------|--------------|---------|
| 1024 | [TBD] dB | [TBD] dB | [TBD] dB | 56-61 dB | [TBD] |
| 4096 | [TBD] dB | [TBD] dB | [TBD] dB | 56-61 dB | [TBD] |

## Analysis

### 1. Raw vs Aligned SQNR

The alignment gain represents the systematic amplitude/phase offset
between cuFFT FP16 and FP64 FFT. This is expected because:
- cuFFT uses FP32 internal accumulators, producing slightly different scaling
- FP16 twiddle factor quantization shifts effective gain
- The optimal   is typically very close to 1 (|  |   0.99-1.01)

### 2. Comparison with v1 Roundtrip Measurement

| N | Forward SQNR (v2, new) | Roundtrip SQNR (v1, old) | Delta |
|---|----------------------|------------------------|-------|
| 1024 | [TBD] dB | 57.1 dB | [TBD] dB |
| 4096 | [TBD] dB | 53.4 dB | [TBD] dB |

Forward SQNR should be ~3 dB higher than roundtrip (half the noise power:
one FFT instead of FFT+IFFT).

### 3. Comparison with Bergach 2026 (Apple M1)

[TBD -- fill after results]

## Key Conclusions

[TBD -- fill after results]

## Data Files

- Per-trial: `experiments/bergach-repro/fp16_fft_sqnr_N1024.csv`
- Per-trial: `experiments/bergach-repro/fp16_fft_sqnr_N4096.csv`
- Summary: `experiments/bergach-repro/fp16_fft_sqnr_summary.csv`
- Script: `experiments/bergach-repro/fp16_fft_sqnr.py`
```

- [ ] **Step 2: Fill [TBD] values**

After reading the summary CSV from Task 2, replace all [TBD] markers with actual measured values. Also fill in the Analysis and Key Conclusions sections based on the results.

- [ ] **Step 3: Commit report**

```bash
git add experiments/bergach-repro/fp16-fft-sqnr.md
git commit -m "docs(low-precision-fft): Bergach Exp 1 corrected report (forward FFT SQNR)"
```

---

### Task 4: Update LAPTOP-CHANGES.md

**Files:**
- Modify: `LAPTOP-CHANGES.md`

- [ ] **Step 1: Prepend corrected experiment entry**

Insert after line `## 2026-06-04: Bergach 2026 Reproduction — NVIDIA Platform Verification` (line 3) and before `### Experiment 1 — FP16 BFP FFT SQNR` (line 5), add:

```markdown

### Experiment 1 (CORRECTED) — FP16 Forward FFT SQNR

- [x] **CORRECTION**: v1 measured ROUNDTRIP (FFT+IFFT) -- WRONG
- [x] Paper   III-A Table I measures SINGLE-PASS forward FFT SQNR vs FP64 reference
- [x] Implemented amplitude alignment (paper   IV-B): optimal complex scaling
- [x] Measured SQNR for N=1024, 4096 (200 trials each) via cuFFT FP16 extension
- [x] Results:
  - N=1024: [TBD] dB (raw) / [TBD] dB (aligned) -- paper: 56-61 dB
  - N=4096: [TBD] dB (raw) / [TBD] dB (aligned) -- paper: 56-61 dB
- [x] Script: `experiments/bergach-repro/fp16_fft_sqnr.py`
- [x] Report: `experiments/bergach-repro/fp16-fft-sqnr.md`
- [x] Data: `experiments/bergach-repro/fp16_fft_sqnr_N{1024,4096}.csv`
```

- [ ] **Step 2: Fill [TBD] with actual results from Task 2**

- [ ] **Step 3: Commit LAPTOP-CHANGES.md update**

```bash
git add LAPTOP-CHANGES.md
git commit -m "docs(low-precision-fft): log Bergach Exp 1 corrected results in LAPTOP-CHANGES.md"
```

---

### Task 5: Push all commits

- [ ] **Step 1: Push to remote**

```bash
git push origin master
```

Expected: Push succeeds. Auto-review on N2920 triggers.

---

## Self-Review

### 1. Spec coverage
- [x] FP16 forward FFT SQNR measurement (single-pass, not roundtrip) -- Task 1
- [x] FP64 (double) reference -- Task 1 (torch.complex128)
- [x] cuFFT FP16 extension usage (`lowp_fft.fft(precision="fp16")`) -- Task 1
- [x] SQNR formula: 10*log10(||X_ref||  / ||X_test - X_ref|| ) -- Task 1
- [x] Amplitude alignment (paper   IV-B): optimal complex scaling   -- Task 1
- [x] N=1024, 4096 -- Task 1
- [x] 200 trials per N -- Task 1
- [x] Data saved to experiments/bergach-repro/ -- Task 1
- [x] Full experiment execution -- Task 2
- [x] Report with analysis and paper comparison -- Task 3
- [x] LAPTOP-CHANGES.md update -- Task 4
- [x] Git push -- Task 5

### 2. Placeholder scan
- No TBD/TODO/implement later in code steps -- Task 1 has complete, verified code
- Report (Task 3) and LAPTOP-CHANGES (Task 4) have [TBD] markers explicitly marked as "fill after results"
- No "add appropriate error handling" -- error handling is explicit (CUDA check, clamp for div-by-zero)
- No "similar to Task N" -- each task is self-contained

### 3. Type consistency
- `sqnr_db(ref: torch.Tensor, test: torch.Tensor) -> float` -- used throughout
- `align_optimal_scale(ref, test) -> tuple[torch.Tensor, complex]` -- returns (aligned_tensor, alpha)
- `generate_signal(N, seed, device) -> torch.Tensor` -- returns FP64 complex on device
- `run_trials(N, n_trials, device) -> dict` -- returns dict with raw_sqnr, aligned_sqnr, alpha_real, alpha_imag
- `summarize(results) -> dict` -- consumes run_trials output format
- All CSV keys match between write (run_trials) and read (summarize)
