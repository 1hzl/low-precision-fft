"""FP16 BFP FFT SQNR -- Bergach 2026 Experiment 1 Reproduction.

Implements the fixed-shift 1/N BFP scheme from arXiv:2605.28451:

  Standard roundtrip:  x -> FFT_FP16 -> IFFT_FP16 -> x_hat
  BFP roundtrip:       x -> FFT_FP16 -> conj*/N -> FFT_FP16 -> conj* -> x_hat

The BFP trick folds 1/N scaling into the conjugate step before the inverse FFT,
keeping intermediate butterfly values within FP16 representable range.

Measures SQNR vs FP32 reference for N=1024, 4096 (200 trials each).
"""

import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lowp_fft import fft as fft_lowp
from lowp_fft import ifft as ifft_lowp


def sqnr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Signal-to-Quantization-Noise Ratio in dB.

    SQNR = 10 * log10(||ref||^2 / ||ref - test||^2)
    """
    signal_power = ref.abs().pow(2).sum().item()
    error_power = (ref - test).abs().pow(2).sum().item()
    return float(10.0 * np.log10(signal_power / max(error_power, 1e-30)))


def fp32_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """FP32 reference: FFT -> IFFT roundtrip."""
    X = torch.fft.fft(x, norm="backward")
    return torch.fft.ifft(X, norm="backward")


def fp16_standard_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """FP16 standard roundtrip via cuFFT FP16 extension.

    Uses lowp_fft.fft (forward) then lowp_fft.ifft (inverse).
    cuFFT computes the IFFT in FP16 then divides by N (norm="backward").
    """
    X = fft_lowp(x, precision="fp16")
    return ifft_lowp(X, precision="fp16")


def fp16_bfp_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """FP16 BFP roundtrip with Bergach fixed-shift 1/N.

    The "2 lines of code" insight from the paper:
      X   = FFT_FP16(x)              # forward FFT in FP16
      Xc  = conj(X) * (1.0 / N)      # fold 1/N into conjugate
      xh  = conj(FFT_FP16(Xc))       # inverse via conjugate identity

    By folding 1/N before the second FFT, all intermediate butterfly
    values stay bounded, preventing FP16 overflow in large transforms.
    """
    N = x.size(-1)
    X = fft_lowp(x, precision="fp16")
    # BFP fixed-shift: conj + 1/N (the 2 critical lines)
    Xc = X.conj() * (1.0 / float(N))
    Xc_fp16 = Xc.to(torch.complex32)  # ensure FP16 input to next FFT
    Xc_fft = fft_lowp(Xc_fp16, precision="fp16")
    return Xc_fft.conj()


def run_trials(N: int, n_trials: int = 200, device: str = "cuda") -> dict:
    """Run SQNR measurement for a single N value.

    Returns dict with lists of per-trial SQNR values.
    """
    torch.manual_seed(42)
    results = {
        "N": N,
        "standard_sqnr": [],
        "bfp_sqnr": [],
    }

    for trial in range(n_trials):
        # Random complex signal, |x| <= 1
        g = torch.Generator(device=device)
        g.manual_seed(trial * 10007 + N)
        real = torch.rand(N, generator=g, device=device, dtype=torch.float32) * 2 - 1
        imag = torch.rand(N, generator=g, device=device, dtype=torch.float32) * 2 - 1
        x = torch.complex(real, imag)
        # Normalize so max |element| <= 1
        x = x / x.abs().max().clamp(min=1e-12)

        with torch.no_grad():
            # FP32 reference
            ref = fp32_roundtrip(x).to(torch.complex64)

            # FP16 standard
            x_c64 = x.to(torch.complex64)
            std = fp16_standard_roundtrip(x_c64).to(torch.complex64)

            # FP16 BFP
            bfp = fp16_bfp_roundtrip(x_c64).to(torch.complex64)

        results["standard_sqnr"].append(sqnr_db(ref, std))
        results["bfp_sqnr"].append(sqnr_db(ref, bfp))

        if (trial + 1) % 50 == 0:
            print(f"  {trial + 1}/{n_trials} trials done")

    return results


def summarize(results: dict) -> dict:
    """Compute summary statistics across trials."""
    std_arr = np.array(results["standard_sqnr"])
    bfp_arr = np.array(results["bfp_sqnr"])
    return {
        "N": results["N"],
        "standard_sqnr_mean": float(np.mean(std_arr)),
        "standard_sqnr_std": float(np.std(std_arr)),
        "standard_sqnr_min": float(np.min(std_arr)),
        "standard_sqnr_max": float(np.max(std_arr)),
        "bfp_sqnr_mean": float(np.mean(bfp_arr)),
        "bfp_sqnr_std": float(np.std(bfp_arr)),
        "bfp_sqnr_min": float(np.min(bfp_arr)),
        "bfp_sqnr_max": float(np.max(bfp_arr)),
        "bfp_improvement_db": float(np.mean(bfp_arr) - np.mean(std_arr)),
        "n_trials": len(std_arr),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("Bergach 2026 Experiment 1: FP16 BFP FFT SQNR")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print(f"Paper claim: FP16 BFP FFT -> 56-61 dB SQNR (Apple M1)")
    print()

    N_values = [1024, 4096]
    n_trials = 200
    all_summaries = []
    all_raw = {}

    for N in N_values:
        print(f"--- N={N} ({n_trials} trials) ---")
        t0 = time.perf_counter()
        results = run_trials(N, n_trials, device)
        t1 = time.perf_counter()

        summary = summarize(results)
        all_summaries.append(summary)
        all_raw[N] = results

        print(f"  Standard FP16 roundtrip: {summary['standard_sqnr_mean']:.1f}"
              f" +/- {summary['standard_sqnr_std']:.1f} dB")
        print(f"  BFP FP16 roundtrip:      {summary['bfp_sqnr_mean']:.1f}"
              f" +/- {summary['bfp_sqnr_std']:.1f} dB")
        print(f"  BFP improvement:          {summary['bfp_improvement_db']:+.1f} dB")
        print(f"  Time: {t1 - t0:.1f}s")
        print()

        # Save per-trial raw data
        csv_path = f"experiments/bergach-repro/fp16_bfp_N{N}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial", "standard_sqnr_db", "bfp_sqnr_db"])
            for i in range(n_trials):
                w.writerow([i, results["standard_sqnr"][i], results["bfp_sqnr"][i]])
        print(f"  Raw data -> {csv_path}\n")

    # Summary CSV
    summary_path = "experiments/bergach-repro/fp16_bfp_data.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        keys = list(all_summaries[0].keys())
        w.writerow(keys)
        for s in all_summaries:
            w.writerow([s[k] for k in keys])
    print(f"Summary -> {summary_path}")

    # Comparison table
    print()
    print("=" * 70)
    print("Comparison: cuFFT FP16 BFP vs Bergach 2026 (Apple M1)")
    print("=" * 70)
    print(f"{'N':>6s}  {'Std SQNR':>12s}  {'BFP SQNR':>12s}  {'BFP Gain':>10s}  {'Paper BFP':>12s}")
    print("-" * 70)
    for s in all_summaries:
        print(f"{s['N']:6d}  {s['standard_sqnr_mean']:11.1f} dB  "
              f"{s['bfp_sqnr_mean']:11.1f} dB  "
              f"{s['bfp_improvement_db']:+9.1f} dB  "
              f"{'56-61 dB':>12s}")
    print("-" * 70)

    # Verdict
    print()
    bfp_snrs = [s["bfp_sqnr_mean"] for s in all_summaries]
    if all(56 <= snr <= 61 for snr in bfp_snrs):
        print("Verdict: MATCHES paper (56-61 dB) on all sizes")
    elif all(snr >= 56 for snr in bfp_snrs):
        print("Verdict: EXCEEDS paper lower bound (>= 56 dB)")
    elif all(snr >= 50 for snr in bfp_snrs):
        print("Verdict: CLOSE to paper -- within ~6 dB")
    else:
        print("Verdict: DOES NOT MATCH paper -- significantly lower SQNR")


if __name__ == "__main__":
    main()
