"""FP16/FP32 Forward FFT SQNR -- Bergach 2026 Experiment 1 (Extended).

v3 extension: full signal/N coverage + FP32 ceiling.

Signal types:
  - uniform:   real,imag ~ U(-1,1)
  - normal:    real,imag ~ N(0,1), clip ±1
  - multitone: 5 random frequencies, equal amplitude
  - impulse:   δ[0]=1, else 0

Precision paths:
  - fp16: lowp_fft.fft(precision="fp16") via cuFFT
  - fp32: torch.fft.fft (complex64) for ceiling reference

Methodology (Bergach 2026 §III-A, §IV-B):
  1. Generate random complex signal x (FP64)
  2. Compute reference: X_ref = FFT_FP64(x)
  3. Compute test:     X_test = FFT_target(x)
  4. Align: α = argmin ||X_ref - α·X_test||²
  5. SQNR = 10·log₁₀(||X_ref||² / ||X_ref - α·X_test||²)
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
    """Find optimal complex scalar α minimizing ||ref - α·test||²."""
    test_c128 = test.to(torch.complex128)
    inner = (ref * test_c128.conj()).sum()
    norm_test = test_c128.abs().pow(2).sum()
    alpha = inner / norm_test.clamp(min=1e-40)
    return alpha * test_c128, complex(alpha.real.item(), alpha.imag.item())


# ─── Signal generators ───────────────────────────────────────────────────────

def _gen_uniform(N: int, seed: int, device: str) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    real = torch.rand(N, generator=g, dtype=torch.float64) * 2.0 - 1.0
    imag = torch.rand(N, generator=g, dtype=torch.float64) * 2.0 - 1.0
    x = torch.complex(real, imag)
    return x.to(device)


def _gen_normal(N: int, seed: int, device: str) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    real = torch.randn(N, generator=g, dtype=torch.float64).clamp(-1.0, 1.0)
    imag = torch.randn(N, generator=g, dtype=torch.float64).clamp(-1.0, 1.0)
    return torch.complex(real, imag).to(device)


def _gen_multitone(N: int, seed: int, device: str) -> torch.Tensor:
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


def _gen_impulse(N: int, seed: int, device: str) -> torch.Tensor:
    x = torch.zeros(N, dtype=torch.complex128)
    x[0] = 1.0
    return x.to(device)


_SIGNAL_SEED_OFFSET = {"uniform": 0, "normal": 1, "multitone": 2, "impulse": 3}

SIGNAL_GENERATORS = {
    "uniform": _gen_uniform,
    "normal": _gen_normal,
    "multitone": _gen_multitone,
    "impulse": _gen_impulse,
}


def generate_signal(N: int, seed: int, device: str, signal_type: str = "uniform") -> torch.Tensor:
    return SIGNAL_GENERATORS[signal_type](N, seed, device)


# ─── Trial runner ────────────────────────────────────────────────────────────

def run_trials(N: int, n_trials: int = 200, device: str = "cuda",
               signal_type: str = "uniform", precision: str = "fp16") -> dict:
    results = {
        "N": N,
        "signal_type": signal_type,
        "precision": precision,
        "raw_sqnr": [],
        "aligned_sqnr": [],
        "alpha_real": [],
        "alpha_imag": [],
    }

    for trial in range(n_trials):
        seed = trial * 10007 + N + _SIGNAL_SEED_OFFSET[signal_type] * 100003
        x_fp64 = generate_signal(N, seed, device, signal_type)

        with torch.no_grad():
            X_ref = torch.fft.fft(x_fp64, norm="backward")

            if precision == "fp16":
                x_fp32 = x_fp64.to(dtype=torch.complex64)
                X_test = fft_lowp(x_fp32, precision="fp16")
            elif precision == "fp32":
                X_test = torch.fft.fft(x_fp64.to(torch.complex64), norm="backward")
            else:
                raise ValueError(f"Unknown precision: {precision}")

            raw_snr = sqnr_db(X_ref, X_test)
            X_aligned, alpha = align_optimal_scale(X_ref, X_test)
            aligned_snr = sqnr_db(X_ref, X_aligned)

        results["raw_sqnr"].append(raw_snr)
        results["aligned_sqnr"].append(aligned_snr)
        results["alpha_real"].append(alpha.real)
        results["alpha_imag"].append(alpha.imag)

        if (trial + 1) % 100 == 0:
            print(f"  {trial + 1}/{n_trials} done  "
                  f"[raw={raw_snr:.1f}, aligned={aligned_snr:.1f} dB, |α|={abs(alpha):.4f}]")

    return results


# ─── Summary ─────────────────────────────────────────────────────────────────

def summarize(results: dict) -> dict:
    raw = np.array(results["raw_sqnr"])
    aligned = np.array(results["aligned_sqnr"])
    return {
        "signal_type": results["signal_type"],
        "precision": results["precision"],
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
        print("ERROR: CUDA required for cuFFT FP16 FFT. Exiting.")
        sys.exit(1)

    print("=" * 72)
    print("Bergach 2026 Experiment 1 (v3 Extended): FP16/FP32 FFT SQNR")
    print("=" * 72)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Paper claim (§III-A Table I): FP16 FFT 56-61 dB SQNR")
    print()

    # ── Run matrix ────────────────────────────────────────────────────────
    N_all = [256, 512, 1024, 2048, 4096]
    signals_fp16 = ["uniform", "normal", "multitone", "impulse"]
    n_trials = 200

    # FP32 ceiling: only uniform, 100 trials (smaller since higher precision = lower variance)
    fp32_cfg = {"signal": "uniform", "N_values": N_all, "n_trials": 100}

    all_summaries = []

    # --- FP16 runs: all signals × all N × 200 trials ---
    for signal_type in signals_fp16:
        for N in N_all:
            label = f"FP16 {signal_type} N={N}"
            print(f"--- {label} ({n_trials} trials) ---")
            t0 = time.perf_counter()
            results = run_trials(N, n_trials, device,
                                signal_type=signal_type, precision="fp16")
            t1 = time.perf_counter()

            summary = summarize(results)
            all_summaries.append(summary)

            print(f"  Aligned SQNR: {summary['aligned_sqnr_mean']:.1f}"
                  f" +/- {summary['aligned_sqnr_std']:.1f} dB")
            print(f"  Alignment gain: {summary['alignment_gain_db']:+.1f} dB")
            print(f"  Time: {t1 - t0:.1f}s\n")

            # Save per-N per-config raw data
            out_dir = "experiments/bergach-repro"
            os.makedirs(out_dir, exist_ok=True)
            csv_path = f"{out_dir}/fp16_fft_sqnr_{signal_type}_fp16_N{N}.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["trial", "signal_type", "precision", "raw_sqnr_db",
                            "aligned_sqnr_db", "alpha_real", "alpha_imag"])
                for i in range(n_trials):
                    w.writerow([
                        i, signal_type, "fp16",
                        results["raw_sqnr"][i], results["aligned_sqnr"][i],
                        results["alpha_real"][i], results["alpha_imag"][i],
                    ])
            print(f"  Raw data -> {csv_path}")

    # --- FP32 ceiling: uniform × all N × 100 trials ---
    cf = fp32_cfg
    for N in cf["N_values"]:
        label = f"FP32 {cf['signal']} N={N}"
        print(f"--- {label} ({cf['n_trials']} trials) [FP32 ceiling] ---")
        t0 = time.perf_counter()
        results = run_trials(N, cf["n_trials"], device,
                            signal_type=cf["signal"], precision="fp32")
        t1 = time.perf_counter()

        summary = summarize(results)
        all_summaries.append(summary)

        print(f"  Aligned SQNR: {summary['aligned_sqnr_mean']:.1f}"
              f" +/- {summary['aligned_sqnr_std']:.1f} dB")
        print(f"  Time: {t1 - t0:.1f}s\n")

        out_dir = "experiments/bergach-repro"
        csv_path = f"{out_dir}/fp16_fft_sqnr_{cf['signal']}_fp32_N{N}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial", "signal_type", "precision", "raw_sqnr_db",
                        "aligned_sqnr_db", "alpha_real", "alpha_imag"])
            for i in range(cf["n_trials"]):
                w.writerow([
                    i, cf["signal"], "fp32",
                    results["raw_sqnr"][i], results["aligned_sqnr"][i],
                    results["alpha_real"][i], results["alpha_imag"][i],
                ])
        print(f"  Raw data -> {csv_path}")

    # ── Comprehensive summary CSV ─────────────────────────────────────────
    summary_path = "experiments/bergach-repro/fp16_fft_sqnr_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        keys = ["signal_type", "precision", "N", "raw_sqnr_mean", "raw_sqnr_std",
                "raw_sqnr_min", "raw_sqnr_max", "aligned_sqnr_mean", "aligned_sqnr_std",
                "aligned_sqnr_min", "aligned_sqnr_max", "alignment_gain_db", "n_trials"]
        w.writerow(keys)
        for s in all_summaries:
            w.writerow([s[k] for k in keys])
    print(f"Summary -> {summary_path}")

    # ── 4×5 Matrix table (FP16 only) ──────────────────────────────────────
    print()
    print("=" * 72)
    print("FP16 SQNR Matrix: 4 Signal Types × 5 N Values")
    print("=" * 72)
    summary_by_key = {(s["signal_type"], s["precision"], s["N"]): s for s in all_summaries}

    header = f"{'Signal':>10s}" + "".join(f"{n:>12d}" for n in N_all) + f"{'':>14s}"
    print(header)
    print(f"{'':>10s}" + "".join(f"{'mean ± std':>12s}" for _ in N_all))
    print("-" * (10 + 12 * len(N_all)))

    for signal_type in signals_fp16:
        row = f"{signal_type:>10s}"
        for N in N_all:
            s = summary_by_key.get((signal_type, "fp16", N))
            if s:
                row += f"  {s['aligned_sqnr_mean']:5.1f}±{s['aligned_sqnr_std']:.1f}"
            else:
                row += f"{'N/A':>12s}"
        print(row)
    print("-" * (10 + 12 * len(N_all)))

    # ── FP32 ceiling table ────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("FP32 Ceiling (uniform signal)")
    print("=" * 72)
    print(f"{'N':>6s}  {'Aligned SQNR':>16s}  {'Expected':>12s}")
    print("-" * 40)
    for N in N_all:
        s = summary_by_key.get(("uniform", "fp32", N))
        if s:
            print(f"{N:6d}  {s['aligned_sqnr_mean']:10.1f} ± {s['aligned_sqnr_std']:.1f} dB"
                  f"  {'~138 dB':>12s}")
    print("-" * 40)

    # ── Verdict ───────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("Verdict: FP16 FFT SQNR vs Bergach 2026 §III-A (56-61 dB)")
    print("=" * 72)
    fp16_snrs = [s for s in all_summaries if s["precision"] == "fp16"]
    all_in_range = all(56 <= s["aligned_sqnr_mean"] <= 61 for s in fp16_snrs)
    min_snr = min(s["aligned_sqnr_mean"] for s in fp16_snrs)
    max_snr = max(s["aligned_sqnr_mean"] for s in fp16_snrs)

    print(f"  FP16 SQNR range across all signals/N: {min_snr:.1f} - {max_snr:.1f} dB")
    if all_in_range:
        print("  VERDICT: ALL configurations within 56-61 dB range -- MATCHES paper")
    else:
        borderline = [s for s in fp16_snrs
                      if not (56 <= s["aligned_sqnr_mean"] <= 61)]
        for s in borderline:
            print(f"  BORDERLINE: {s['signal_type']} N={s['N']}: {s['aligned_sqnr_mean']:.1f} dB")
        if all(s["aligned_sqnr_mean"] >= 53 for s in borderline):
            print("  VERDICT: CLOSE to paper -- all within ~3 dB of 56 dB lower bound")
        else:
            print("  VERDICT: SOME configurations significantly below paper range")


if __name__ == "__main__":
    main()
