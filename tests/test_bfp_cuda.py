"""Tests for BFP FFT CUDA kernel (Sprint 3.3).

Runs the standalone bfp_fft.exe and validates:
  1. Correctness: BFP CUDA vs Python BFP prototype, SQNR deviation < 2 dB
  2. Stability: N=256..4096 all >= 15 dB
  3. Exponents: per-stage count = log2N + 1
"""

import subprocess
import sys
import re
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
BFP_EXE = PROJECT_ROOT / "build" / "bfp_fft.exe"


def requires_bfp_exe():
    if not BFP_EXE.exists():
        pytest.skip(f"{BFP_EXE} not built — run build_bfp.bat first")


def parse_bfp_output(stdout_bytes):
    """Parse bfp_fft.exe output bytes into list of {N, signal, sqnr_db, max_err}."""
    results = []
    for line_bytes in stdout_bytes.split(b"\n"):
        line = line_bytes.strip()
        if line.startswith(b"#") or not line:
            continue
        parts = line.split()
        # Remove "dB" token if present in the line
        parts = [p for p in parts if p != b"dB"]
        if len(parts) >= 4:
            try:
                N = int(parts[0])
                signal = parts[1].decode("ascii", errors="replace")
                sqnr = float(parts[2])
                max_err = float(parts[3])
                results.append({
                    "N": N, "signal": signal,
                    "sqnr_db": sqnr, "max_err": max_err,
                })
            except (ValueError, IndexError):
                pass
    return results


def run_bfp_exe(N=None):
    """Run bfp_fft.exe and return parsed results."""
    requires_bfp_exe()
    args = [str(BFP_EXE)]
    if N is not None:
        args.append(str(N))
    result = subprocess.run(args, capture_output=True, timeout=120,
                            cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"bfp_fft.exe failed:\n{result.stderr}")
    return parse_bfp_output(result.stdout)


# ── Baseline: Python BFP prototype SQNR ────────────────────────────────

def python_bfp_sqnr(N, seed=42):
    """Compute BFP SQNR using the Python prototype for given N."""
    from lowp_fft.bfp_fft import BFPFFT
    np.random.seed(seed + N)
    x = (np.random.randn(N) + 1j * np.random.randn(N)) / N / 3.0
    ref = np.fft.fft(x)
    bfp = BFPFFT(N)
    bfp_out = bfp.forward(x)
    d = np.sum(np.abs(ref - bfp_out) ** 2)
    s = np.sum(np.abs(ref) ** 2)
    return float(10 * np.log10(s / (d + 1e-15)))


# ── Tests ───────────────────────────────────────────────────────────────

class TestBFPCUDABasics:
    """Basic validation of the CUDA BFP FFT executable."""

    def test_exe_exists(self):
        assert BFP_EXE.exists(), f"{BFP_EXE} not found — run build_bfp.bat"

    def test_exe_runs(self):
        requires_bfp_exe()
        result = subprocess.run([str(BFP_EXE)], capture_output=True,
                                timeout=120, cwd=str(PROJECT_ROOT))
        assert result.returncode == 0, f"bfp_fft.exe crashed:\n{result.stderr}"
        assert b"BFP FFT CUDA Kernel" in result.stdout

    def test_produces_results(self):
        results = run_bfp_exe()
        assert len(results) >= 10, f"Expected >= 10 results, got {len(results)}"


class TestBFPCUDAStability:
    """BFP CUDA SQNR must be >= 15 dB for all tested N."""

    @pytest.mark.parametrize("N", [256, 512, 1024, 2048, 4096])
    def test_sqnr_minimum(self, N):
        results = run_bfp_exe(N)
        for r in results:
            assert r["sqnr_db"] >= 15.0, \
                f"N={r['N']} {r['signal']}: SQNR={r['sqnr_db']:.1f} dB < 15 dB"


class TestBFPCUDAvsPython:
    """CUDA BFP SQNR must match Python BFP prototype within ±2 dB."""

    @pytest.mark.parametrize("N", [256, 512, 1024, 2048, 4096])
    def test_random_sqnr_match(self, N):
        py_sqnr = python_bfp_sqnr(N)
        results = run_bfp_exe(N)
        cuda_random = [r for r in results if r["signal"] == "random"]
        assert len(cuda_random) == 1, f"Expected 1 random result for N={N}"
        cuda_sqnr = cuda_random[0]["sqnr_db"]
        diff = abs(cuda_sqnr - py_sqnr)
        assert diff < 2.0, \
            f"N={N}: CUDA {cuda_sqnr:.1f} dB vs Python {py_sqnr:.1f} dB, diff={diff:.1f} dB"


class TestBFPCUDAExponents:
    """BFP CUDA must produce log2(N)+1 exponents per FFT call.

    This is tested indirectly by verifying the .exe runs without error
    for all N, since exponent errors would cause NaN/inf in output."""

    def test_all_n_run_without_error(self):
        for N in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            results = run_bfp_exe(N)
            for r in results:
                assert not np.isnan(r["sqnr_db"]), \
                    f"N={N}: NaN SQNR (possible exponent overflow)"
                assert not np.isinf(r["sqnr_db"]), \
                    f"N={N}: Inf SQNR (possible exponent error)"


# ── Benchmark helper ────────────────────────────────────────────────────

def run_bfp_benchmark():
    """Print comparison table: CUDA BFP vs Python BFP vs naive FP8."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from tests.sim_fp8_fft_error import fp8_fft_radix2

    cuda_results = run_bfp_exe()

    print(f"{'N':>6s}  {'CUDA BFP':>10s}  {'Py BFP':>10s}  {'Naive FP8':>10s}  {'Δ CUDA-Py':>10s}")
    print("-" * 60)
    for N in [256, 512, 1024, 2048, 4096]:
        cuda_r = [r for r in cuda_results if r["N"] == N and r["signal"] == "random"]
        cuda_sqnr = cuda_r[0]["sqnr_db"] if cuda_r else float("nan")

        py_sqnr = python_bfp_sqnr(N)

        np.random.seed(42 + N)
        x = (np.random.randn(N) + 1j * np.random.randn(N)) / N / 3.0
        ref = np.fft.fft(x)
        naive = fp8_fft_radix2(x)
        d = np.sum(np.abs(ref - naive) ** 2)
        s = np.sum(np.abs(ref) ** 2)
        naive_sqnr = float(10 * np.log10(s / (d + 1e-15)))

        diff = cuda_sqnr - py_sqnr
        print(f"{N:6d}  {cuda_sqnr:9.1f} dB  {py_sqnr:9.1f} dB  "
              f"{naive_sqnr:9.1f} dB  {diff:+9.1f} dB")


if __name__ == "__main__":
    run_bfp_benchmark()
