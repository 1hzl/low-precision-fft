"""Sprint 4.2 — BFP Memory Bandwidth Benchmark.

Measures effective memory bandwidth (GB/s) in memory-bandwidth-bound scenarios:
large batch (256--1024) × large N (4096--32768).

Compares:
  - BFP FP8   (2 bytes/element: 1 byte per __nv_fp8_e4m3 real + imag)
  - FP16       (4 bytes/element: complex32)
  - FP32       (8 bytes/element: complex64)

BFP throughput measured via bfp_fft.exe --bench mode.
cuFFT FP16/FP32 measured via torch.cuda.Event.
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BFP_EXE = os.path.join(PROJECT_ROOT, "build", "bfp_fft.exe")

# Configuration: memory-bandwidth-bound region — large N × large batch
FFT_SIZES = [4096, 8192, 16384, 32768]
BATCH_SIZES = [256, 512, 1024]
WARMUP = 20
REPS = 200
MAX_BATCH_TIME_S = 30.0  # cap per-config wall time


def vram_estimate(n, batch):
    """Estimate worst-case VRAM for FP32 cuFFT."""
    return batch * n * 8 * 3 / 1e9


def run_bfp_bench(N, warmup, reps):
    """Run BFP benchmark via bfp_fft.exe --bench mode.
    Returns median per-FFT GPU time in microseconds.
    """
    if not os.path.exists(BFP_EXE):
        return None

    args = [BFP_EXE, "--bench", str(N), str(warmup), str(reps), "1"]
    try:
        result = subprocess.run(args, capture_output=True, timeout=300,
                                cwd=PROJECT_ROOT)
        if result.returncode != 0:
            return None
        stdout = result.stdout.decode()
        m = re.search(r"per_fft_us=(\d+\.?\d*)", stdout)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def bench_cufft(device, n, batch, warmup, reps, precision="fp16"):
    """Benchmark cuFFT using torch.cuda.Event for GPU timing.
    Returns (per_fft_us, total_time_s).
    """
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

    t0 = time.perf_counter()
    start.record(stream)
    for _ in range(reps):
        if precision == "fp32":
            _ = torch.fft.fft(x_in, dim=-1)
        else:
            _ = _ext.fft_fp16_forward(x_in)
    end.record(stream)
    sync()
    wall_s = time.perf_counter() - t0

    total_ms = start.elapsed_time(end)
    per_fft_us = (total_ms * 1000) / (reps * batch)

    return per_fft_us, wall_s


def effective_bandwidth(n, per_fft_us, bytes_per_element):
    """Compute effective memory bandwidth in GB/s.

    For an in-place FFT, each element is read once + written once per pass.
    We use 2 × bytes_per_element as a conservative estimate of memory moved
    per FFT (input read + output write). Real FFT implementations may
    move more due to scratch buffers; this is a lower-bound estimate.
    """
    total_bytes = n * bytes_per_element * 2  # read + write
    time_s = per_fft_us * 1e-6
    if time_s == 0:
        return 0.0
    return total_bytes / time_s / 1e9  # GB/s


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    vram_gb = props.total_memory / 1024**3
    bw_theoretical = props.memoryClockRate * props.memoryBusWidth / 8.0 / 1e6 * 2  # GB/s (DDR)

    print(f"# Sprint 4.2 — BFP Memory Bandwidth Benchmark")
    print(f"# Device: {props.name} ({vram_gb:.1f} GiB VRAM)")
    print(f"# Theoretical BW: {bw_theoretical:.0f} GB/s (GDDR)")
    print(f"# Warmup: {WARMUP}, Reps: {REPS}")
    print()

    # Collect BFP per-FFT times (independent of batch since no native batching)
    bfp_times = {}
    print("Measuring BFP FP8 GPU per-FFT times...")
    for n in FFT_SIZES:
        us = run_bfp_bench(n, min(WARMUP, 10), min(REPS // 4, 50))
        if us is not None:
            bfp_times[n] = us
            bw = effective_bandwidth(n, us, 2)  # 2 bytes/element (FP8)
            print(f"  N={n:6d}: {us:8.1f} us/FFT, {bw:6.2f} GB/s effective BW")
        else:
            bfp_times[n] = None
            print(f"  N={n:6d}: FAILED")

    print()
    header = (f"{'N':>6s} {'Batch':>6s} "
              f"{'BFP-FP8':>10s} {'FP16':>10s} {'FP32':>10s} "
              f"{'BFP8 BW':>10s} {'FP16 BW':>10s} {'FP32 BW':>10s} "
              f"{'BFP/FP16':>10s} {'BFP/FP32':>10s}")
    print(header)
    print("-" * len(header))

    results = []
    for n in FFT_SIZES:
        for batch in BATCH_SIZES:
            est = vram_estimate(n, batch)
            if est > vram_gb * 0.85:
                print(f"  Skipping N={n} batch={batch} (VRAM est {est:.1f}GiB > 85%)")
                continue

            # cuFFT FP16
            fp16_us, wall16 = bench_cufft(device, n, batch, WARMUP, REPS, "fp16")

            # cuFFT FP32
            fp32_us, wall32 = bench_cufft(device, n, batch, WARMUP, REPS, "fp32")

            # BFP FP8: per-FFT time × batch (sequential, no native batching)
            bfp_us = bfp_times.get(n)
            bfp_total_us = bfp_us * batch if bfp_us else None

            if bfp_us:
                bfp_bw = effective_bandwidth(n, bfp_us, 2)
            else:
                bfp_bw = 0
            fp16_bw = effective_bandwidth(n, fp16_us, 4)
            fp32_bw = effective_bandwidth(n, fp32_us, 8)

            bfp_vs_fp16 = fp16_us / bfp_us if bfp_us else 0
            bfp_vs_fp32 = fp32_us / bfp_us if bfp_us else 0

            bfp_str = f"{bfp_us:.1f} us" if bfp_us else "N/A"
            print(f"{n:6d} {batch:6d} "
                  f"{bfp_str:>10s} {fp16_us:9.1f} us {fp32_us:9.1f} us "
                  f"{bfp_bw:9.2f} GB/s {fp16_bw:9.2f} GB/s {fp32_bw:9.2f} GB/s "
                  f"{bfp_vs_fp16:9.2f}x {bfp_vs_fp32:9.2f}x")

            results.append({
                "N": n,
                "batch": batch,
                "bfp_us": bfp_us if bfp_us else -1,
                "fp16_us": fp16_us,
                "fp32_us": fp32_us,
                "bfp_bw_gbs": bfp_bw,
                "fp16_bw_gbs": fp16_bw,
                "fp32_bw_gbs": fp32_bw,
                "bfp_vs_fp16_speedup": bfp_vs_fp16,
                "bfp_vs_fp32_speedup": bfp_vs_fp32,
            })

    # Save CSV
    csv_path = os.path.join("data", "sprint-4.2-memory-bw.csv")
    os.makedirs("data", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "batch", "bfp_us", "fp16_us", "fp32_us",
                     "bfp_bw_gbs", "fp16_bw_gbs", "fp32_bw_gbs",
                     "bfp_vs_fp16_speedup", "bfp_vs_fp32_speedup"])
        for r in results:
            w.writerow([r["N"], r["batch"],
                        f"{r['bfp_us']:.2f}" if r['bfp_us'] >= 0 else "N/A",
                        f"{r['fp16_us']:.2f}",
                        f"{r['fp32_us']:.2f}",
                        f"{r['bfp_bw_gbs']:.2f}",
                        f"{r['fp16_bw_gbs']:.2f}",
                        f"{r['fp32_bw_gbs']:.2f}",
                        f"{r['bfp_vs_fp16_speedup']:.4f}" if r['bfp_us'] >= 0 else "N/A",
                        f"{r['bfp_vs_fp32_speedup']:.4f}" if r['bfp_us'] >= 0 else "N/A"])

    print(f"\nResults saved to {csv_path}")

    # Summary
    if results:
        valid = [r for r in results if r["bfp_us"] > 0]
        if valid:
            # Memory bandwidth comparison
            f16_speeds = [r["bfp_vs_fp16_speedup"] for r in valid]
            f32_speeds = [r["bfp_vs_fp32_speedup"] for r in valid]
            print(f"\nBFP FP8 (2 bytes/elem) vs FP16 (4 bytes/elem):")
            print(f"  Speedup range: {min(f16_speeds):.2f}x – {max(f16_speeds):.2f}x, "
                  f"mean {sum(f16_speeds)/len(f16_speeds):.2f}x")
            print(f"  BW ratio (BFP/FP16): "
                  f"{sum(r['bfp_bw_gbs'] for r in valid)/len(valid):.2f} / "
                  f"{sum(r['fp16_bw_gbs'] for r in valid)/len(valid):.2f} GB/s")
            print(f"BFP FP8 (2 bytes/elem) vs FP32 (8 bytes/elem):")
            print(f"  Speedup range: {min(f32_speeds):.2f}x – {max(f32_speeds):.2f}x, "
                  f"mean {sum(f32_speeds)/len(f32_speeds):.2f}x")
            print(f"\nNote: BFP has no native batching — per-FFT kernel is launched "
                  f"sequentially. Effective BW reflects per-element storage advantage.")


if __name__ == "__main__":
    main()
