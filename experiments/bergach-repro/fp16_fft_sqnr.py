"""FP16 Forward FFT SQNR -- Bergach 2026 Experiment 1 (Corrected).

CORRECTION from v1 fp16_bfp_sqnr.py:
  - OLD (wrong): measured ROUNDTRIP SQNR (FFT + IFFT) with BFP trick
  - NEW (correct): measures SINGLE-PASS forward FFT SQNR vs FP64 reference
    Matches paper §III-A Table I: FP16 Stockham FFT SQNR against
    double-precision reference, not roundtrip reconstruction.

Methodology (aligned with Bergach 2026 §III-A, §IV-B):
  1. Generate random complex signal x (FP64), max |x_i| = 1
  2. Compute reference: X_ref = FFT_FP64(x)
  3. Compute test:     X_test = FFT_FP16(x_fp32)  via cuFFT FP16 extension
  4. Align amplitudes:  α = argmin ||X_ref - α·X_test||²  (optimal complex scaling)
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
    """
    test_c128 = test.to(torch.complex128)
    signal_power = ref.abs().pow(2).sum().item()
    error_power = (ref - test_c128).abs().pow(2).sum().item()
    return float(10.0 * np.log10(signal_power / max(error_power, 1e-40)))


def align_optimal_scale(ref: torch.Tensor, test: torch.Tensor):
    """Find optimal complex scalar α minimizing ||ref - α·test||².

    Closed-form: α = ⟨ref, test⟩ / ||test||² where ⟨a,b⟩ = sum(a · conj(b)).

    Separates systematic gain/phase offset from random quantization noise
    (Bergach 2026 §IV-B: "align amplitudes before computing SQNR").

    Returns (aligned_test, alpha) where aligned_test = α · test (complex128).
    """
    test_c128 = test.to(torch.complex128)
    inner = (ref * test_c128.conj()).sum()
    norm_test = test_c128.abs().pow(2).sum()
    alpha = inner / norm_test.clamp(min=1e-40)
    return alpha * test_c128, complex(alpha.real.item(), alpha.imag.item())


def generate_signal(N: int, seed: int, device: str) -> torch.Tensor:
    """Generate random complex signal in FP64, max |element| = 1.

    Uses CPU-side generator for full reproducibility across GPU architectures.
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
      4. Compute raw SQNR and aligned SQNR (with optimal α scaling)
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
            X_ref = torch.fft.fft(x_fp64, norm="backward")

            # cuFFT requires FP32 complex input; converts internally to FP16 compute
            x_fp32 = x_fp64.to(dtype=torch.complex64)
            X_fp16 = fft_lowp(x_fp32, precision="fp16")

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
                  f"|α|={abs(alpha):.4f}]")

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
    print(f"Paper claim (§III-A Table I): FP16 FFT mantissa-limited at")
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
    print("Comparison: cuFFT FP16 Forward FFT SQNR vs Bergach 2026 §III-A")
    print("=" * 72)
    header = (f"{'N':>6s}  {'Raw SQNR':>14s}  {'Aligned SQNR':>14s}  "
              f"{'Paper §III-A':>14s}  {'Verdict':>20s}")
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
    print("Paper §III-A measures forward FFT only, matching this corrected approach.")


if __name__ == "__main__":
    main()
