"""BF16 FFT SQNR statistics — mean ± std over 100 trials per (N, signal) pair.

Measures forward FFT SQNR (dB) of BF16 cuFFT vs FP64 reference with optimal
complex scaling α (Bergach 2026 §IV-B).

Output: data/sqn-bf16-stats.csv + data/sqn-bf16-stats.md
"""

import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowp_fft import fft as fft_lowp

N_VALUES = [256, 512, 1024, 2048, 4096]
SIGNAL_TYPES = ["uniform", "normal", "multitone", "impulse"]
TRIALS = 100

_SIGNAL_SEED_OFFSET = {"uniform": 0, "normal": 1, "multitone": 2, "impulse": 3}


def sqnr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    test_c128 = test.to(torch.complex128)
    signal_power = ref.abs().pow(2).sum().item()
    error_power = (ref - test_c128).abs().pow(2).sum().item()
    return float(10.0 * np.log10(signal_power / max(error_power, 1e-40)))


def align_optimal_scale(ref: torch.Tensor, test: torch.Tensor):
    test_c128 = test.to(torch.complex128)
    inner = (ref * test_c128.conj()).sum()
    norm_test = test_c128.abs().pow(2).sum()
    alpha = inner / norm_test.clamp(min=1e-40)
    return alpha * test_c128


def gen_uniform(N: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    real = torch.rand(N, generator=g, dtype=torch.float64) * 2.0 - 1.0
    imag = torch.rand(N, generator=g, dtype=torch.float64) * 2.0 - 1.0
    return torch.complex(real, imag).to(device)


def gen_normal(N: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    real = torch.randn(N, generator=g, dtype=torch.float64).clamp(-1.0, 1.0)
    imag = torch.randn(N, generator=g, dtype=torch.float64).clamp(-1.0, 1.0)
    return torch.complex(real, imag).to(device)


def gen_multitone(N: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    n_freqs = 5
    freqs = torch.randint(0, N // 2, (n_freqs,), generator=g)
    phases = torch.rand(n_freqs, generator=g, dtype=torch.float64) * 2.0 * math.pi
    x = torch.zeros(N, dtype=torch.complex128)
    t = torch.arange(N, dtype=torch.float64)
    for f, phi in zip(freqs.tolist(), phases.tolist()):
        x += torch.exp(1j * (2.0 * math.pi * f * t / N + phi))
    x = x / x.abs().max().clamp(min=1e-40)
    return x.to(device)


def gen_impulse(N: int, seed: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros(N, dtype=torch.complex128)
    x[0] = 1.0
    return x.to(device)


GENERATORS = {
    "uniform": gen_uniform,
    "normal": gen_normal,
    "multitone": gen_multitone,
    "impulse": gen_impulse,
}


def run_trials(N: int, signal_type: str, n_trials: int, device: torch.device):
    sqnr_values = []
    for trial in range(n_trials):
        seed = trial * 10007 + N + _SIGNAL_SEED_OFFSET[signal_type] * 100003
        x_fp64 = GENERATORS[signal_type](N, seed, device)

        with torch.no_grad():
            X_ref = torch.fft.fft(x_fp64, norm="backward")
            x_fp32 = x_fp64.to(dtype=torch.complex64)
            X_test = fft_lowp(x_fp32, precision="bf16")
            X_aligned = align_optimal_scale(X_ref, X_test)
            snr = sqnr_db(X_ref, X_aligned)

        sqnr_values.append(snr)

    return sqnr_values


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "signal", "sqnr_mean", "sqnr_std", "trials"])
        for row in rows:
            w.writerow(row)


def save_md(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# BF16 FFT SQNR Statistics (mean ± std)",
        "",
        f"**Date**: 2026-06-05",
        f"**GPU**: {torch.cuda.get_device_name(0)}",
        f"**Reference**: FP64 torch.fft.fft (complex128)",
        f"**Trials**: {TRIALS} per (N, signal) combination",
        f"**Method**: Bergach 2026 §IV-B optimal complex scaling α",
        "",
        "## Results",
        "",
        "| Signal | N | SQNR (dB) |",
        "|--------|---|-----------|",
    ]
    for N, sig, mean_val, std_val, trials in rows:
        lines.append(f"| {sig} | {N} | {mean_val:.2f} ± {std_val:.2f} |")

    lines += [
        "",
        "## Summary by N",
        "",
        "| N | Avg SQNR (dB) | Range |",
        "|---|---------------|-------|",
    ]
    by_n = {}
    for N, sig, mean_val, std_val, trials in rows:
        by_n.setdefault(N, []).append(mean_val)
    for N in N_VALUES:
        vals = by_n[N]
        lines.append(f"| {N} | {np.mean(vals):.2f} | {min(vals):.2f} – {max(vals):.2f} |")

    lines += [
        "",
        "## Summary by Signal Type",
        "",
        "| Signal | Avg SQNR (dB) | Range |",
        "|--------|---------------|-------|",
    ]
    by_sig = {}
    for N, sig, mean_val, std_val, trials in rows:
        by_sig.setdefault(sig, []).append(mean_val)
    for sig in SIGNAL_TYPES:
        vals = by_sig[sig]
        lines.append(f"| {sig} | {np.mean(vals):.2f} | {min(vals):.2f} – {max(vals):.2f} |")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA required for cuFFT BF16 FFT.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"N values: {N_VALUES}")
    print(f"Signal types: {SIGNAL_TYPES}")
    print(f"Trials per combination: {TRIALS}")
    print()

    total = len(N_VALUES) * len(SIGNAL_TYPES)
    rows = []
    idx = 0

    for sig in SIGNAL_TYPES:
        for N in N_VALUES:
            idx += 1
            print(f"[{idx}/{total}] {sig:10s} N={N:5d} ... ", end="", flush=True)
            t0 = time.perf_counter()

            sqnr_vals = run_trials(N, sig, TRIALS, device)
            mean_val = float(np.mean(sqnr_vals))
            std_val = float(np.std(sqnr_vals))

            t1 = time.perf_counter()
            print(f"{mean_val:.2f} ± {std_val:.2f} dB  ({t1 - t0:.1f}s)")

            rows.append((N, sig, mean_val, std_val, TRIALS))

    csv_path = os.path.join("data", "sqn-bf16-stats.csv")
    md_path = os.path.join("data", "sqn-bf16-stats.md")
    save_csv(rows, csv_path)
    print(f"\nCSV saved to {csv_path}")
    save_md(rows, md_path)
    print(f"Report saved to {md_path}")

    # Quick summary
    print()
    all_means = [r[2] for r in rows]
    print(f"Overall SQNR range: {min(all_means):.2f} – {max(all_means):.2f} dB")
    print(f"Grand mean: {np.mean(all_means):.2f} dB")


if __name__ == "__main__":
    main()
