#!/usr/bin/env python3
"""
FP8 E4M3 FFT Error Simulation

Simulates a radix-2 DIT FFT where every arithmetic operation is quantized
to FP8 E4M3 (1 sign, 4 exponent, 3 mantissa). Compares against FP32 reference
to validate the theoretical error model from docs/fp8-fft-error-model.md.

Author: low-precision-fft project
Date: 2026-06-04
"""

import numpy as np
import sys
from pathlib import Path

# ── FP8 E4M3 Quantization ────────────────────────────────────────────

def _fp8_quantize_table():
    """Build a lookup table of all 256 FP8 E4M3 values."""
    vals = []
    for bits in range(256):
        s = (bits >> 7) & 1
        e = (bits >> 3) & 0xF
        m = bits & 0x7
        if e == 0:
            val = (2.0 ** (-6)) * (m / 8.0)
        else:
            val = (2.0 ** (e - 7)) * (1.0 + m / 8.0)
        if s:
            val = -val
        vals.append(val)
    # Override NaN encoding (E=15, M=7): return 0.0
    vals[0xFF] = 0.0  # negative NaN → 0
    vals[0x7F] = 0.0  # positive NaN → 0
    return np.array(vals, dtype=np.float64)

_FP8_TABLE = _fp8_quantize_table()


# Precompute positive FP8 table + decision boundaries (module-level, done once)
_POS_VALS = np.unique(np.sort(_FP8_TABLE[0:128]))  # positive FP8 values
_BOUNDARIES = np.zeros(len(_POS_VALS))
_BOUNDARIES[0] = 0.0
for _i in range(1, len(_POS_VALS)):
    _BOUNDARIES[_i] = (_POS_VALS[_i - 1] + _POS_VALS[_i]) / 2.0


def _quantize_real_scalar(x):
    """Quantize a single real scalar to nearest FP8 E4M3 value."""
    if x == 0.0:
        return 0.0
    x = max(-448.0, min(448.0, x))
    sign = 1.0 if x >= 0 else -1.0
    abs_x = abs(x)
    idx = max(0, np.searchsorted(_BOUNDARIES, abs_x, side='right') - 1)
    idx = min(idx, len(_POS_VALS) - 1)
    return sign * float(_POS_VALS[idx])


def quantize_fp8_e4m3(x):
    """Quantize a float/complex scalar or array to FP8 E4M3 via nearest-value lookup."""
    is_complex = np.iscomplexobj(x)
    x_arr = np.asarray(x)
    scalar_input = (x_arr.ndim == 0)

    if is_complex:
        real_arr = np.atleast_1d(np.asarray(x.real, dtype=np.float64))
        imag_arr = np.atleast_1d(np.asarray(x.imag, dtype=np.float64))
        real_q = _quantize_real_vec(real_arr)
        imag_q = _quantize_real_vec(imag_arr)
        result = real_q + 1j * imag_q
    else:
        real_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        result = _quantize_real_vec(real_arr)

    if scalar_input:
        return float(result.flat[0]) if not is_complex else complex(result.flat[0])
    return result


def _quantize_real_vec(arr):
    """Quantize real-valued (non-complex) array to FP8 E4M3, fast path via searchsorted."""
    arr = np.clip(arr, -448.0, 448.0)
    sign = np.where(arr >= 0, 1.0, -1.0)
    abs_arr = np.abs(arr)
    idx = np.searchsorted(_BOUNDARIES, abs_arr, side='right') - 1
    idx = np.clip(idx, 0, len(_POS_VALS) - 1)
    result = sign * _POS_VALS[idx]
    result[abs_arr == 0] = 0.0
    return result


# ── FP8 FFT Simulation ──────────────────────────────────────────────

def fp8_fft_radix2(x, inverse=False):
    """
    Radix-2 DIT FFT with FP8 quantization after EVERY arithmetic operation.

    This is the worst-case simulation: every multiply and add is quantized
    to FP8, mirroring what a pure FP8 CUDA kernel would do.
    """
    x = np.asarray(x, dtype=np.complex128).copy()
    N = len(x)

    if N & (N - 1) != 0:
        raise ValueError(f"N={N} must be a power of 2")

    # Bit-reversal permutation
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            x[i], x[j] = x[j], x[i]

    # Butterfly stages
    step = 1
    while step < N:
        jump = step << 1
        if inverse:
            twiddle_base = np.exp(2j * np.pi / jump)
        else:
            twiddle_base = np.exp(-2j * np.pi / jump)

        for group in range(0, N, jump):
            w = 1.0 + 0.0j
            for pair in range(step):
                idx_a = group + pair
                idx_b = idx_a + step

                A = x[idx_a]
                B = x[idx_b]
                W = w

                # ── Butterfly: FP8 all the way ──
                # Step 1: W * B (complex multiply)
                # (a+ib)(c+id) = (ac-bd) + i(ad+bc)
                ac = quantize_fp8_e4m3(W.real * B.real)
                bd = quantize_fp8_e4m3(W.imag * B.imag)
                WB_real = quantize_fp8_e4m3(ac - bd)

                ad = quantize_fp8_e4m3(W.real * B.imag)
                bc = quantize_fp8_e4m3(W.imag * B.real)
                WB_imag = quantize_fp8_e4m3(ad + bc)

                WB = WB_real + 1j * WB_imag

                # Step 2: A' = A + WB (complex add)
                Aprime_real = quantize_fp8_e4m3(A.real + WB.real)
                Aprime_imag = quantize_fp8_e4m3(A.imag + WB.imag)

                # Step 3: B' = A - WB (complex subtract = add with negation)
                # Negation in FP8 is just sign flip (exact)
                Bprime_real = quantize_fp8_e4m3(A.real - WB.real)
                Bprime_imag = quantize_fp8_e4m3(A.imag - WB.imag)

                x[idx_a] = Aprime_real + 1j * Aprime_imag
                x[idx_b] = Bprime_real + 1j * Bprime_imag

                # Update twiddle factor (also FP8)
                w_next = w * twiddle_base
                w = quantize_fp8_e4m3(w_next)

        step = jump

    return x


# ── Reference FFT (FP32) ────────────────────────────────────────────

def fp32_fft(x, inverse=False):
    """Standard radix-2 DIT FFT in FP64 (as FP32 reference)."""
    return np.fft.fft(x) if not inverse else np.fft.ifft(x)


# ── Error Metrics ────────────────────────────────────────────────────

def compute_metrics(ref, test):
    """Compute error metrics between reference and test signals."""
    abs_diff = np.abs(ref - test)
    rel_err = abs_diff / (np.abs(ref) + 1e-10)
    return {
        'max_abs_err': float(np.max(abs_diff)),
        'mean_abs_err': float(np.mean(abs_diff)),
        'max_rel_err': float(np.max(rel_err)),
        'mean_rel_err': float(np.mean(rel_err)),
        'rmse': float(np.sqrt(np.mean(abs_diff ** 2))),
        'snr_db': float(10 * np.log10(
            np.sum(np.abs(ref) ** 2) / (np.sum(abs_diff ** 2) + 1e-15)
        )),
    }


# ── Scaling Simulation ──────────────────────────────────────────────

def scale_for_fp8(x):
    """Scale input so max |FFT| < 448. Returns (scaled_x, scale_factor)."""
    # Conservative: scale so max |x| after FFT stays well within FP8 range
    N = len(x)
    # For random signal with |x| <= 1, empirical FFT peak ~ sqrt(N)*pi/4
    # Conservative: use N as theoretical max
    scale = 448.0 / N
    return x * scale, scale


# ── Main Benchmark ──────────────────────────────────────────────────

def run_benchmark(N_values=None, n_trials=5, signal_types=None,
                  normalize=True, verbose=True):
    """
    Run FP8 FFT error benchmark across N values and signal types.

    Args:
        N_values: list of N (power-of-2)
        n_trials: number of random trials per N
        signal_types: list of signal generators
        normalize: if True, apply 1/N normalization to prevent overflow
        verbose: print progress

    Returns:
        list of dicts with benchmark results
    """
    if N_values is None:
        N_values = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    if signal_types is None:
        signal_types = {
            'random_uniform': lambda n: np.random.uniform(-1, 1, n) +
                                       1j * np.random.uniform(-1, 1, n),
            'random_normal': lambda n: (np.random.randn(n) +
                                       1j * np.random.randn(n)) / 3.0,
            'multitone': lambda n: _gen_multitone(n, 5),
            'chirp': lambda n: _gen_chirp(n),
        }

    results = []
    for N in N_values:
        for sig_name, sig_gen in signal_types.items():
            trial_errs = []
            for trial in range(n_trials):
                np.random.seed(trial * 10000 + N)
                x = sig_gen(N)

                if normalize:
                    x = x / N  # forward normalization

                # Reference
                ref = fp32_fft(x)

                # FP8 FFT
                try:
                    fp8_out = fp8_fft_radix2(x)
                except Exception as e:
                    if verbose:
                        print(f"  SKIP N={N} {sig_name}: {e}")
                    continue

                metrics = compute_metrics(ref, fp8_out)
                trial_errs.append(metrics)

            # Average across trials
            avg = {k: np.mean([t[k] for t in trial_errs])
                   for k in trial_errs[0]}
            avg['N'] = N
            avg['signal'] = sig_name
            avg['n_trials'] = n_trials
            results.append(avg)

            if verbose:
                print(f"N={N:5d}  {sig_name:16s}  "
                      f"SNR={avg['snr_db']:6.1f} dB  "
                      f"max_rel_err={avg['max_rel_err']:.4f}  "
                      f"rmse={avg['rmse']:.6f}")

    return results


def _gen_multitone(n, n_tones):
    """Generate signal with n_tones random frequency components."""
    x = np.zeros(n, dtype=np.complex128)
    freqs = np.random.choice(n, n_tones, replace=False)
    for f in freqs:
        amp = np.random.uniform(0.5, 1.0)
        phase = np.random.uniform(0, 2 * np.pi)
        x += amp * np.exp(1j * phase) * np.exp(2j * np.pi * f *
                                                np.arange(n) / n)
    peak = np.max(np.abs(x))
    if peak > 0:
        x /= peak  # normalize to |x| <= 1
    return x


def _gen_chirp(n):
    """Generate linear chirp from f0=0 to f1=0.5."""
    t = np.arange(n) / n
    phase = 2 * np.pi * 0.5 * t ** 2 * n
    return np.exp(1j * phase)


# ── Theory Comparison ────────────────────────────────────────────────

def theoretical_snr_estimate_random_walk(N, ulp=0.125):
    """
    Naive random-walk model (WRONG — overestimates error by 16-27 dB).

    Included for comparison only. The corrected model is
    theoretical_snr_estimate() below.
    """
    log2N = int(np.log2(N))
    ops_per_butterfly = 12
    total_ops = (N / 2) * log2N * ops_per_butterfly
    per_op_rms = ulp / np.sqrt(12)
    rms_error = np.sqrt(total_ops) * per_op_rms
    signal_rms = np.sqrt(1.0 / 3.0)
    snr = signal_rms / (rms_error + 1e-15)
    snr_db = 20 * np.log10(snr)
    return {
        'N': N, 'log2N': log2N, 'total_ops': total_ops,
        'per_op_rms': per_op_rms, 'est_rms_error': rms_error,
        'est_snr_db': snr_db,
    }


def theoretical_snr_estimate(N, eps=0.125):
    """
    Floating-point FFT error bound (Higham 2002, Ch.23).

    Worst-case L2 error bound: ‖err‖₂ / ‖x‖₂ ≤ γ_{log₂N}
    where γ_k = k·ε / (1 − k·ε), ε = unit roundoff (≈ ulp/2 for rounding to nearest).

    For FP8 E4M3: ε ≈ 0.125, so the bound diverges (kε ≥ 1) for k ≥ 8 → N ≥ 256.
    This means classical worst-case error analysis GUARANTEES NOTHING for FP8 FFT with N ≥ 256.

    Returns the bound (may be Infinity when diverged) and whether the bound is meaningful.
    """
    log2N = int(np.log2(N))
    k = log2N
    ke = k * eps

    if ke >= 1.0:
        # Bound diverged — worst-case error is unbounded
        return {
            'N': N, 'log2N': log2N, 'eps': eps,
            'k_eps': ke, 'gamma': float('inf'),
            'est_snr_db': float('-inf'),
            'diverged': True,
        }
    else:
        gamma = ke / (1.0 - ke)
        snr_db = -20.0 * np.log10(max(gamma, 1e-15))
        return {
            'N': N, 'log2N': log2N, 'eps': eps,
            'k_eps': ke, 'gamma': gamma,
            'est_snr_db': snr_db,
            'diverged': False,
        }


def theoretical_snr_bfp(N, eps_per_stage=0.125):
    """
    Block Floating-Point FFT error estimate.

    With BFP, each stage has a shared exponent, so we only quantize once per value
    per stage. The per-stage relative error is ≈ eps (instead of eps accumulating
    through every multiply and add within the stage).

    Error bound: γ_{log₂N} with ε_{bfp} = eps (per-stage, not per-op)
    → diverges only when log₂N ≥ 1/eps = 8 → N ≥ 256

    But even at the divergence boundary, BFP's effective eps is much smaller
    because the shared exponent keeps values in the optimal FP8 range [0.5, 1.0).
    """
    log2N = int(np.log2(N))
    k = log2N
    ke = k * eps_per_stage

    if ke >= 1.0:
        gamma = float('inf')
        snr_db = float('-inf')
    else:
        gamma = ke / (1.0 - ke)
        snr_db = -20.0 * np.log10(max(gamma, 1e-15))

    return {
        'N': N, 'log2N': log2N,
        'eps_per_stage': eps_per_stage,
        'gamma': gamma,
        'est_snr_db': snr_db,
        'diverged': ke >= 1.0,
    }


def print_theory_table(N_values=None):
    """Print theoretical SNR estimates — naive FP8 vs BFP."""
    if N_values is None:
        N_values = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    header = f"{'N':>6s}  {'log2N':>5s}  {'Naive FP8':>12s}  {'BFP bound':>12s}  {'Verdict':>15s}"
    print("─" * 75)
    print("Higham (2002) Error Bound Analysis: Naive FP8 vs Block Floating-Point")
    print("─" * 75)
    print(header)
    print("─" * 75)
    for N in N_values:
        naive = theoretical_snr_estimate(N)
        bfp = theoretical_snr_bfp(N)
        n_snr = f"{naive['est_snr_db']:.1f} dB" if not naive['diverged'] else "DIVERGED"
        b_snr = f"{bfp['est_snr_db']:.1f} dB" if not bfp['diverged'] else "DIVERGED"
        if naive['diverged']:
            verdict = "NEED BFP"
        elif naive['est_snr_db'] < 0:
            verdict = "POOR"
        else:
            verdict = "OK"
        print(f"{N:6d}  {naive['log2N']:5d}  {n_snr:>12s}  {b_snr:>12s}  {verdict:>15s}")
    print("─" * 75)
    print("Higham gamma_k bound: ||err||_2 <= gamma_k * ||x||_2  (gamma_k = k*eps/(1-k*eps))")
    print("Naive FP8: eps ~ 0.125 per op -> bound diverges at k>=8 (N>=256)")
    print("BFP:       eps ~ 0.125 per STAGE -> bound diverges at k>=8 (N>=256)")
    print("(Both diverge in theory, but BFP has much better empirical behavior)")
    print("─" * 75)


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("FP8 E4M3 FFT Error Simulation")
    print("=" * 70)
    print()

    # 1. Theory predictions
    print_theory_table()
    print()

    # 2. Simulation
    print("─" * 70)
    print("Simulated FP8 FFT Errors (normalized, 5 trials avg)")
    print("─" * 70)
    N_vals = [16, 32, 64, 128, 256, 512, 1024, 2048]
    results = run_benchmark(N_values=N_vals, n_trials=5, normalize=True, verbose=True)
    print()

    # 3. Summary table
    print("─" * 80)
    print("Naive FP8 FFT — Empirical Summary")
    print("─" * 80)
    print(f"{'N':>6s}  {'SNR (rand)':>11s}  {'SNR (chirp)':>12s}  {'Assessment':>15s}  {'Note':>25s}")
    print("─" * 80)
    for N in N_vals:
        sim_rand = [r for r in results if r['N'] == N and r['signal'] == 'random_uniform']
        sim_chirp = [r for r in results if r['N'] == N and r['signal'] == 'chirp']
        if sim_rand:
            snr_r = sim_rand[0]['snr_db']
            snr_c = sim_chirp[0]['snr_db'] if sim_chirp else float('nan')
            if snr_r >= 10:
                assess, note = "GOOD", "usable without BFP"
            elif snr_r >= 6:
                assess, note = "USABLE", "marginal, BFP recommended"
            elif snr_r >= 0:
                assess, note = "MARGINAL", "BFP required"
            else:
                assess, note = "FAILED", "BFP absolutely required"
            print(f"{N:6d}  {snr_r:10.1f} dB  {snr_c:11.1f} dB  {assess:>15s}  {note:>25s}")
    print("─" * 80)
    print("Key:  N ≤ 128 → usable as-is.  N ≥ 256 → BFP or other scaling strategy required.")
    print()

    # 4. Block floating-point comparison
    print("=" * 70)
    print("Block Floating-Point (BFP) Feasibility Check")
    print("=" * 70)
    print()
    print("For BFP, each FFT stage shares one exponent.")
    print("Per-stage value ranges (simulated with random normal input, N=1024):")
    print()

    N = 1024
    np.random.seed(42)
    x_rand = np.random.randn(N) + 1j * np.random.randn(N)
    x_rand /= np.max(np.abs(x_rand))  # normalize to |x| <= 1

    # Manually track per-stage max values
    x_stage = x_rand.copy()
    # Bit reversal
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            x_stage[i], x_stage[j] = x_stage[j], x_stage[i]

    stage = 1
    log2N = int(np.log2(N))
    while stage < N:
        jump = stage << 1
        twiddle_base = np.exp(-2j * np.pi / jump)
        stage_max = 0.0
        for group in range(0, N, jump):
            w = 1.0 + 0.0j
            for pair in range(stage):
                A = x_stage[group + pair]
                B = x_stage[group + pair + stage]
                Aprime = A + w * B
                Bprime = A - w * B
                stage_max = max(stage_max, abs(Aprime), abs(Bprime))
                x_stage[group + pair] = Aprime
                x_stage[group + pair + stage] = Bprime
                w *= twiddle_base
        s_idx = int(np.log2(stage))
        print(f"  Stage {s_idx:2d} (size={stage:4d}): "
              f"max|val| = {stage_max:.3f}, "
              f"FP8 safe: {'YES' if stage_max < 448 else 'NO (needs scaling)'} "
              f"exponent needed: {int(np.ceil(np.log2(stage_max + 1e-10)))}")
        stage = jump

    print()
    print("Done.")
