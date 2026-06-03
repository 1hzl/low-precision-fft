"""Sprint 2.4 — FP16 vs FP32 FFT throughput benchmark.

Measures raw cuFFT performance (batched), Python wrapper overhead, and
throughput in GB/s and GFLOPS.

Key insight: Python wrapper overhead (~15us/call) dominates for small FFTs.
Batching amortizes this overhead.
"""

import csv
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowp_fft import _cufft_ext as _ext
from lowp_fft import fft as fft_lowp


FFT_SIZES = [
    256, 512, 1024, 2048, 4096, 8192, 16384,
    32768, 65536, 131072, 262144, 524288, 1048576,
]

BATCH_SIZES = [1, 4, 16, 64, 256]


def bench_raw_batched(device, n, batch, warmup=10, reps=100):
    """Benchmark raw cuFFT FP16 vs torch.fft FP32, batched."""
    sync = torch.cuda.synchronize

    # Create batched input: (batch, n) complex
    x = torch.randn(batch, n, 2, dtype=torch.float32, device=device)
    x_c64 = torch.view_as_complex(x).contiguous()
    x_c32 = x_c64.to(torch.complex32).contiguous()

    # FP32 via torch.fft
    for _ in range(warmup):
        _ = torch.fft.fft(x_c64, dim=-1)
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = torch.fft.fft(x_c64, dim=-1)
    sync()
    t32 = (time.perf_counter() - t0) / reps * 1e6

    # FP16 via raw cuFFT (bypass Python wrapper)
    for _ in range(warmup):
        _ = _ext.fft_fp16_forward(x_c32)
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = _ext.fft_fp16_forward(x_c32)
    sync()
    t16 = (time.perf_counter() - t0) / reps * 1e6

    # Throughput: (read + write) bytes / time
    bytes_fp32 = batch * n * 8 * 2  # complex64 = 8B/elem, read+write
    bytes_fp16 = batch * n * 4 * 2  # complex32 = 4B/elem, read+write
    gbps32 = bytes_fp32 / (t32 * 1e-6) / 1e9
    gbps16 = bytes_fp16 / (t16 * 1e-6) / 1e9

    # GFLOPS: 5N log2(N) operations per FFT
    ops_per_fft = 5 * n * math.log2(n)
    gflops32 = batch * ops_per_fft / (t32 * 1e-6) / 1e9
    gflops16 = batch * ops_per_fft / (t16 * 1e-6) / 1e9

    return {
        "size": n, "batch": batch,
        "fp32_us": t32, "fp16_us": t16,
        "speedup": t32 / t16,
        "fp32_gbps": gbps32, "fp16_gbps": gbps16,
        "fp32_gflops": gflops32, "fp16_gflops": gflops16,
    }


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    peak_bw = props.memory_clock_rate * props.memory_bus_width / 8 * 1e-3  # GB/s approx
    print(f"Device: {props.name} ({props.total_memory // 1024**2} MiB VRAM)")
    print(f"Peak memory BW (approx): {peak_bw:.0f} GB/s")
    print()

    results = []
    print(f"{'N':>8s} {'batch':>6s} {'FP32us':>8s} {'FP16us':>8s} {'speedup':>8s} {'FP32GB/s':>9s} {'FP16GB/s':>9s} {'FP32GFLOPS':>11s} {'FP16GFLOPS':>11s}")
    print("-" * 95)

    for n in FFT_SIZES:
        for batch in BATCH_SIZES:
            # Skip if VRAM estimate exceeds 10GB
            vram_est = batch * n * 8 * 3 / 1e9  # 3 tensors: input, output_fp32, output_fp16
            if vram_est > 10:
                print(f"  Skipping N={n} batch={batch} (VRAM est {vram_est:.1f}GB > 10GB)")
                continue

            r = bench_raw_batched(device, n, batch)
            results.append(r)
            print(f"{n:8d} {batch:6d} {r['fp32_us']:8.1f} {r['fp16_us']:8.1f} {r['speedup']:7.2f}x {r['fp32_gbps']:9.2f} {r['fp16_gbps']:9.2f} {r['fp32_gflops']:11.2f} {r['fp16_gflops']:11.2f}")

    # Save CSV
    csv_path = os.path.join("data", "sprint-2.4-throughput.csv")
    os.makedirs("data", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "batch", "fp32_us", "fp16_us", "speedup",
                     "fp32_gbps", "fp16_gbps", "fp32_gflops", "fp16_gflops"])
        for r in results:
            w.writerow([r["size"], r["batch"], f"{r['fp32_us']:.2f}", f"{r['fp16_us']:.2f}",
                        f"{r['speedup']:.4f}", f"{r['fp32_gbps']:.2f}", f"{r['fp16_gbps']:.2f}",
                        f"{r['fp32_gflops']:.2f}", f"{r['fp16_gflops']:.2f}"])

    # Summary
    speedups = [r["speedup"] for r in results]
    print(f"\nSummary: speedup range {min(speedups):.2f}x – {max(speedups):.2f}x, avg {sum(speedups)/len(speedups):.2f}x")
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
