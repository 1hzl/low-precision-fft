"""BFP FP8 FFT SQNR statistics — mean ± std over 100 trials per (N, signal) pair.

Measures forward FFT SQNR (dB) of BFP FP8 FFT vs FP64 reference with optimal
complex scaling α (Bergach 2026 §IV-B).

Output: data/sqn-bfp-stats.csv + data/sqn-bfp-stats.md
"""

import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowp_fft.bfp_fft import BFPFFT

N_VALUES = [256, 512, 1024, 2048, 4096]
SIGNAL_TYPES = ["uniform", "normal", "multitone", "impulse"]
TRIALS = 100

_SIGNAL_SEED_OFFSET = {"uniform": 0, "normal": 1, "multitone": 2, "impulse": 3}


def sqnr_db(ref: np.ndarray, test: np.ndarray) -> float:
    signal_power = float(np.sum(np.abs(ref) ** 2))
    error_power = float(np.sum(np.abs(ref - test) ** 2))
    return float(10.0 * np.log10(signal_power / max(error_power, 1e-40)))


def align_optimal_scale(ref: np.ndarray, test: np.ndarray) -> np.ndarray:
    inner = np.sum(ref * np.conj(test))
    norm_test = np.sum(np.abs(test) ** 2)
    alpha = inner / max(float(norm_test), 1e-40)
    return alpha * test


def gen_uniform(N: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    real = rng.uniform(-1.0, 1.0, N)
    imag = rng.uniform(-1.0, 1.0, N)
    return real + 1j * imag


def gen_normal(N: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    real = np.clip(rng.normal(0.0, 1.0, N), -1.0, 1.0)
    imag = np.clip(rng.normal(0.0, 1.0, N), -1.0, 1.0)
    return real + 1j * imag


def gen_multitone(N: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    n_freqs = 5
    freqs = rng.integers(0, N // 2, n_freqs)
    phases = rng.uniform(0.0, 2.0 * math.pi, n_freqs)
    x = np.zeros(N, dtype=np.complex128)
    t = np.arange(N, dtype=np.float64)
    for f, phi in zip(freqs.tolist(), phases.tolist()):
        x += np.exp(1j * (2.0 * math.pi * f * t / N + phi))
    x = x / max(float(np.max(np.abs(x))), 1e-40)
    return x


def gen_impulse(N: int, seed: int) -> np.ndarray:
    x = np.zeros(N, dtype=np.complex128)
    x[0] = 1.0
    return x


GENERATORS = {
    "uniform": gen_uniform,
    "normal": gen_normal,
    "multitone": gen_multitone,
    "impulse": gen_impulse,
}


def run_trials(N: int, signal_type: str, n_trials: int):
    sqnr_values = []
    bfp = BFPFFT(N)
    for trial in range(n_trials):
        seed = trial * 10007 + N + _SIGNAL_SEED_OFFSET[signal_type] * 100003
        x_fp64 = GENERATORS[signal_type](N, seed)

        X_ref = np.fft.fft(x_fp64, norm="backward")
        X_test = bfp.forward(x_fp64)
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
        "# BFP FP8 FFT SQNR Statistics (mean ± std)",
        "",
        "**Date**: 2026-06-05",
        "**Platform**: CPU (Python/NumPy BFP prototype)",
        "**Reference**: FP64 numpy.fft.fft (complex128)",
        "**Trials**: 100 per (N, signal) combination",
        "**Method**: Bergach 2026 §IV-B optimal complex scaling α",
        "**Implementation**: BFPFP8 — block floating-point Radix-2 DIT, one shared exponent per stage",
        "",
        "## Results",
        "",
        "| Signal | N | SQNR (dB) |",
        "|--------|---|-----------|",
    ]
    for N, sig, mean_val, std_val, trials in rows:
        star = " *" if sig == "impulse" else ""
        lines.append(f"| {sig} | {N} | {mean_val:.2f} ± {std_val:.2f} |{star}")

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
        star = " *" if sig == "impulse" else ""
        lines.append(f"| {sig} | {np.mean(vals):.2f} | {min(vals):.2f} – {max(vals):.2f} |{star}")

    lines += [
        "",
        "* Degenerate case — FFT of δ(t) is a constant vector. The transform is mathematically exact; SQNR reflects FP64 numerical noise, not actual FFT precision.",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("BFP FP8 FFT SQNR Statistics (mean ± std)")
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

            sqnr_vals = run_trials(N, sig, TRIALS)
            mean_val = float(np.mean(sqnr_vals))
            std_val = float(np.std(sqnr_vals))

            t1 = time.perf_counter()
            print(f"{mean_val:.2f} ± {std_val:.2f} dB  ({t1 - t0:.1f}s)")

            rows.append((N, sig, mean_val, std_val, TRIALS))

    csv_path = os.path.join("data", "sqn-bfp-stats.csv")
    md_path = os.path.join("data", "sqn-bfp-stats.md")
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
