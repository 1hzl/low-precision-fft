"""Sprint 3.4 — BFP FP8 CUDA vs cuFFT FP16 vs cuFFT FP32 throughput benchmark.

Measures GPU kernel execution time for all three methods across
N = [256, 512, 1024, 2048, 4096] and batch = [1, 16, 64, 256].

BFP: measured via standanole bfp_fft.exe --bench mode (CUDA events on GPU).
cuFFT: measured via torch.cuda.Event for FP16 (lowp_fft extension) and FP32.
"""

import csv
import math
import os
import re
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowp_fft import _cufft_ext as _ext
from lowp_fft import fft as fft_lowp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BFP_EXE = os.path.join(PROJECT_ROOT, "build", "bfp_fft.exe")

FFT_SIZES = [256, 512, 1024, 2048, 4096]
BATCH_SIZES = [1, 16, 64, 256]
WARMUP = 100
REPS = 1000


def run_bfp_bench(N, warmup, reps):
    """Run BFP benchmark via bfp_fft.exe --bench mode.
    Returns median per-FFT GPU time in microseconds.
    BFP has no native batching, so per-FFT time is constant regardless of batch.
    """
    if not os.path.exists(BFP_EXE):
        return None

    args = [BFP_EXE, "--bench", str(N), str(warmup), str(reps), "1"]
    try:
        result = subprocess.run(args, capture_output=True, timeout=300,
                                cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"  BFP EXE error (N={N}): {result.stderr.decode()}")
            return None
        stdout = result.stdout.decode()
        m = re.search(r"per_fft_us=(\d+\.?\d*)", stdout)
        if m:
            return float(m.group(1))
    except Exception as e:
        print(f"  BFP EXE exception (N={N}): {e}")
    return None


def bench_cufft(device, n, batch, warmup, reps, precision="fp16"):
    """Benchmark cuFFT using torch.cuda.Event for GPU timing."""
    sync = torch.cuda.synchronize
    stream = torch.cuda.current_stream()

    x = torch.randn(batch, n, 2, dtype=torch.float32, device=device)
    x_c64 = torch.view_as_complex(x).contiguous()

    if precision == "fp32":
        x_in = x_c64
    else:
        x_in = x_c64.to(torch.complex32).contiguous()

    # Warmup
    for _ in range(warmup):
        if precision == "fp32":
            _ = torch.fft.fft(x_in, dim=-1)
        else:
            _ = _ext.fft_fp16_forward(x_in)
    sync()

    # Timed with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream)
    for _ in range(reps):
        if precision == "fp32":
            _ = torch.fft.fft(x_in, dim=-1)
        else:
            _ = _ext.fft_fp16_forward(x_in)
    end.record(stream)
    sync()

    total_ms = start.elapsed_time(end)
    per_fft_us = (total_ms * 1000) / (reps * batch)

    # GFLOPS: 5N log2(N) operations per FFT
    ops_per_fft = 5 * n * math.log2(n)
    gflops = ops_per_fft / (per_fft_us * 1e-6) / 1e9

    return per_fft_us, gflops


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"# Sprint 3.4 — BFP FP8 vs cuFFT Throughput Benchmark")
    print(f"# Device: {props.name} ({props.total_memory // 1024**2} MiB VRAM)")
    print(f"# Warmup: {WARMUP}, Reps: {REPS}")
    print()

    # Collect BFP baselines (per-FFT time, independent of batch)
    bfp_times = {}
    print("Measuring BFP FP8 GPU times...")
    for n in FFT_SIZES:
        us = run_bfp_bench(n, min(50, WARMUP), min(200, REPS // 5))
        if us is not None:
            bfp_times[n] = us
            ops_per_fft = 5 * n * math.log2(n)
            gflops = ops_per_fft / (us * 1e-6) / 1e9
            print(f"  N={n:6d}: {us:8.1f} us/FFT, {gflops:6.2f} GFLOPS")
        else:
            bfp_times[n] = None
            print(f"  N={n:6d}: FAILED")

    print()
    print(f"{'N':>6s} {'Batch':>6s} {'BFP FP8':>10s} {'cuFFT FP16':>11s} {'cuFFT FP32':>11s} {'BFP vs FP16':>12s} {'BFP GFLOPS':>11s} {'FP16 GFLOPS':>11s} {'FP32 GFLOPS':>11s}")
    print("-" * 115)

    results = []
    for n in FFT_SIZES:
        for batch in BATCH_SIZES:
            vram_est = batch * n * 8 * 3 / 1e9
            if vram_est > 10:
                print(f"  Skipping N={n} batch={batch} (VRAM est {vram_est:.1f}GB > 10GB)")
                continue

            # cuFFT FP16
            fp16_us, fp16_gflops = bench_cufft(device, n, batch, WARMUP, REPS, "fp16")

            # cuFFT FP32
            fp32_us, fp32_gflops = bench_cufft(device, n, batch, WARMUP, REPS, "fp32")

            # BFP FP8: per-FFT time × batch (no native batching)
            bfp_us = bfp_times.get(n)
            bfp_total_us = bfp_us * batch if bfp_us else None
            if bfp_us:
                ops_per_fft = 5 * n * math.log2(n)
                bfp_gflops = ops_per_fft / (bfp_us * 1e-6) / 1e9
            else:
                bfp_gflops = 0

            speedup = fp16_us / bfp_us if bfp_us else 0

            bfp_str = f"{bfp_us:.1f} us" if bfp_us else "N/A"
            print(f"{n:6d} {batch:6d} {bfp_str:>10s} {fp16_us:10.1f} us {fp32_us:10.1f} us {speedup:11.2f}x {bfp_gflops:10.2f} {fp16_gflops:10.2f} {fp32_gflops:10.2f}")

            results.append({
                "N": n,
                "batch": batch,
                "bfp_us": bfp_us if bfp_us else -1,
                "fp16_us": fp16_us,
                "fp32_us": fp32_us,
                "bfp_vs_fp16": speedup,
                "bfp_gflops": bfp_gflops,
                "fp16_gflops": fp16_gflops,
                "fp32_gflops": fp32_gflops,
            })

    # Save CSV
    csv_path = os.path.join("data", "sprint-3.4-throughput.csv")
    os.makedirs("data", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "batch", "bfp_us", "fp16_us", "fp32_us",
                     "bfp_vs_fp16", "bfp_gflops", "fp16_gflops", "fp32_gflops"])
        for r in results:
            w.writerow([r["N"], r["batch"],
                        f"{r['bfp_us']:.2f}" if r['bfp_us'] >= 0 else "N/A",
                        f"{r['fp16_us']:.2f}",
                        f"{r['fp32_us']:.2f}",
                        f"{r['bfp_vs_fp16']:.4f}" if r['bfp_us'] >= 0 else "N/A",
                        f"{r['bfp_gflops']:.2f}",
                        f"{r['fp16_gflops']:.2f}",
                        f"{r['fp32_gflops']:.2f}"])

    print(f"\nResults saved to {csv_path}")

    # Summary
    if results:
        valid = [r for r in results if r["bfp_us"] > 0]
        if valid:
            speeds = [r["bfp_vs_fp16"] for r in valid]
            print(f"BFP vs FP16 speedup range: {min(speeds):.2f}x – {max(speeds):.2f}x, "
                  f"avg {sum(speeds)/len(speeds):.2f}x")
            print(f"(BFP has no native batching, so larger batch helps cuFFT, not BFP)")


if __name__ == "__main__":
    main()
