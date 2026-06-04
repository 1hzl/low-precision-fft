"""BF16 Forward FFT SQNR vs FP64 -- Bergach 2026 methodology.

Measures BF16 FFT SQNR against FP64 (complex128) ground truth, same method
as the FP16 experiment so the two are directly comparable.

Methodology (Bergach 2026 §III-A, §IV-B):
  1. Generate random complex signal x (FP64)
  2. Compute reference: X_ref = FFT_FP64(x)
  3. Compute test:     X_test = FFT_BF16(x → complex64)
  4. Align: α = argmin ||X_ref - α·X_test||²
  5. SQNR = 10·log₁₀(||X_ref||² / ||X_ref - α·X_test||²)

Signal: uniform real,imag ~ U(-1,1)
N: [256, 512, 1024, 2048, 4096], 100 trials each
"""

import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lowp_fft import fft as fft_lowp


# ─── SQNR utilities ──────────────────────────────────────────────────────────

def sqnr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    test_c128 = test.to(torch.complex128)
    signal_power = ref.abs().pow(2).sum().item()
    error_power = (ref - test_c128).abs().pow(2).sum().item()
    return float(10.0 * np.log10(signal_power / max(error_power, 1e-40)))


def align_optimal_scale(ref: torch.Tensor, test: torch.Tensor):
    """Find optimal complex scalar α minimizing ||ref - α·test||² (Bergach §IV-B)."""
    test_c128 = test.to(torch.complex128)
    inner = (ref * test_c128.conj()).sum()
    norm_test = test_c128.abs().pow(2).sum()
    alpha = inner / norm_test.clamp(min=1e-40)
    return alpha * test_c128, complex(alpha.real.item(), alpha.imag.item())


# ─── Signal generators ───────────────────────────────────────────────────────

def generate_uniform(N: int, seed: int, device: str) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    real = torch.rand(N, generator=g, dtype=torch.float64) * 2.0 - 1.0
    imag = torch.rand(N, generator=g, dtype=torch.float64) * 2.0 - 1.0
    return torch.complex(real, imag).to(device)


# ─── Trial runner ────────────────────────────────────────────────────────────

def run_trials(N: int, n_trials: int = 100, device: str = "cuda") -> dict:
    results = {
        "N": N,
        "precision": "bf16",
        "signal_type": "uniform",
        "raw_sqnr": [],
        "aligned_sqnr": [],
        "alpha_real": [],
        "alpha_imag": [],
    }

    for trial in range(n_trials):
        seed = trial * 10007 + N + 400003  # offset to avoid overlap with FP16 experiment
        x_fp64 = generate_uniform(N, seed, device)

        with torch.no_grad():
            X_ref = torch.fft.fft(x_fp64, norm="backward")

            # BF16 path: complex128 → complex64 → lowp_fft.fft(precision="bf16")
            x_fp32 = x_fp64.to(dtype=torch.complex64)
            X_test = fft_lowp(x_fp32, precision="bf16")

            raw_snr = sqnr_db(X_ref, X_test)
            X_aligned, alpha = align_optimal_scale(X_ref, X_test)
            aligned_snr = sqnr_db(X_ref, X_aligned)

        results["raw_sqnr"].append(raw_snr)
        results["aligned_sqnr"].append(aligned_snr)
        results["alpha_real"].append(alpha.real)
        results["alpha_imag"].append(alpha.imag)

        if (trial + 1) % 50 == 0:
            print(f"  {trial + 1}/{n_trials} done  "
                  f"[raw={raw_snr:.1f}, aligned={aligned_snr:.1f} dB, |α|={abs(alpha):.4f}]")

    return results


# ─── Summary ─────────────────────────────────────────────────────────────────

def summarize(results: dict) -> dict:
    raw = np.array(results["raw_sqnr"])
    aligned = np.array(results["aligned_sqnr"])
    return {
        "precision": "bf16",
        "signal_type": results["signal_type"],
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


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("ERROR: CUDA required for BF16 cuFFT FFT. Exiting.")
        sys.exit(1)

    print("=" * 72)
    print("Bergach 2026 Methodology: BF16 FFT SQNR vs FP64 Reference")
    print("=" * 72)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Signal: uniform real,imag ~ U(-1,1)")
    print(f"Reference: torch.fft.fft(complex128)")
    print(f"Test:     lowp_fft.fft(complex64, precision='bf16')")
    print()

    N_all = [256, 512, 1024, 2048, 4096]
    n_trials = 100
    all_summaries = []

    for N in N_all:
        label = f"BF16 uniform N={N}"
        print(f"--- {label} ({n_trials} trials) ---")
        t0 = time.perf_counter()
        results = run_trials(N, n_trials, device)
        t1 = time.perf_counter()

        summary = summarize(results)
        all_summaries.append(summary)

        print(f"  Aligned SQNR: {summary['aligned_sqnr_mean']:.1f}"
              f" +/- {summary['aligned_sqnr_std']:.1f} dB")
        print(f"  Raw SQNR:     {summary['raw_sqnr_mean']:.1f}"
              f" +/- {summary['raw_sqnr_std']:.1f} dB")
        print(f"  Alignment gain: {summary['alignment_gain_db']:+.2f} dB")
        print(f"  Time: {t1 - t0:.1f}s\n")

        out_dir = "experiments/bergach-repro"
        os.makedirs(out_dir, exist_ok=True)
        csv_path = f"{out_dir}/bf16_fft_sqnr_uniform_N{N}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial", "signal_type", "precision", "raw_sqnr_db",
                        "aligned_sqnr_db", "alpha_real", "alpha_imag"])
            for i in range(n_trials):
                w.writerow([
                    i, "uniform", "bf16",
                    results["raw_sqnr"][i], results["aligned_sqnr"][i],
                    results["alpha_real"][i], results["alpha_imag"][i],
                ])
        print(f"  Raw data -> {csv_path}")

    # ── Summary CSV ─────────────────────────────────────────────────────────
    summary_path = "experiments/bergach-repro/bf16_fft_sqnr_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        keys = ["precision", "signal_type", "N", "raw_sqnr_mean", "raw_sqnr_std",
                "raw_sqnr_min", "raw_sqnr_max", "aligned_sqnr_mean", "aligned_sqnr_std",
                "aligned_sqnr_min", "aligned_sqnr_max", "alignment_gain_db", "n_trials"]
        w.writerow(keys)
        for s in all_summaries:
            w.writerow([s[k] for k in keys])
    print(f"Summary -> {summary_path}")

    # ── Results table ───────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("BF16 SQNR vs FP64 Reference (Bergach methodology)")
    print("=" * 72)
    header = f"{'N':>6s}  {'Aligned SQNR':>16s}  {'Raw SQNR':>16s}  {'Gain':>8s}"
    print(header)
    print("-" * 52)
    for s in all_summaries:
        print(f"{s['N']:6d}  {s['aligned_sqnr_mean']:10.1f} ± {s['aligned_sqnr_std']:.1f} dB"
              f"  {s['raw_sqnr_mean']:10.1f} ± {s['raw_sqnr_std']:.1f} dB"
              f"  {s['alignment_gain_db']:+.2f}")
    print("-" * 52)

    # ── Comparison with FP16 ────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("Comparison: BF16 vs FP16 SQNR (both vs FP64 reference)")
    print("=" * 72)
    print(f"{'N':>6s}  {'BF16 SQNR':>16s}  {'FP16 SQNR':>16s}  {'Δ (BF16-FP16)':>16s}")
    print("-" * 60)
    fp16_snrs = {256: 61.3, 512: 60.5, 1024: 59.9, 2048: 59.3, 4096: 56.5}
    for s in all_summaries:
        N = s["N"]
        fp16_snr = fp16_snrs.get(N, 0)
        delta = s["aligned_sqnr_mean"] - fp16_snr
        print(f"{N:6d}  {s['aligned_sqnr_mean']:10.1f} ± {s['aligned_sqnr_std']:.1f} dB"
              f"  {fp16_snr:10.1f} dB"
              f"  {delta:+10.1f} dB")
    print("-" * 60)


if __name__ == "__main__":
    main()
