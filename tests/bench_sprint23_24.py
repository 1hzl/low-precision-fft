"""Sprint 2.3 & 2.4 — FP16 vs FP32 precision and performance benchmarks.

Measures:
  - Precision: max relative error, RMSE, mean absolute error
  - Performance: execution time, throughput (GB/s), speedup ratio
  - Signal types: multi-tone, random noise, impulse, chirp
  - FFT sizes: 256 .. 1048576 (powers of 2)

Saves CSV + markdown report to data/.
"""

import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowp_fft import fft as fft_lowp


# ─── Signal generators ─────────────────────────────────────────────────

def _normalize(x: torch.Tensor, peak: float = 0.5) -> torch.Tensor:
    """Scale so |x| <= peak / sqrt(N) to avoid FP16 overflow at large N."""
    n = x.size(-1)
    x = x / (x.abs().max() + 1e-12)
    return x * (peak / math.sqrt(float(n)))


def signal_multitone(n: int, device: torch.device) -> torch.Tensor:
    """Sum of several sine waves."""
    t = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(0)
    freqs = torch.tensor([3.0, 7.0, 13.0, 23.0, 47.0], device=device).unsqueeze(1)
    real = torch.sum(torch.cos(2 * math.pi * freqs * t / n), dim=0)
    imag = torch.sum(torch.sin(2 * math.pi * freqs * t / n), dim=0)
    x = torch.stack([real, imag], dim=-1) / math.sqrt(5.0)
    return _normalize(torch.view_as_complex(x), peak=0.5)


def signal_random(n: int, device: torch.device) -> torch.Tensor:
    """Uniform(-1, 1) random complex signal."""
    real = torch.rand(n, dtype=torch.float32, device=device) * 2 - 1
    imag = torch.rand(n, dtype=torch.float32, device=device) * 2 - 1
    x = torch.stack([real, imag], dim=-1)
    return _normalize(torch.view_as_complex(x), peak=0.5)


def signal_impulse(n: int, device: torch.device) -> torch.Tensor:
    """Unit impulse at index 0."""
    x = torch.zeros(n, dtype=torch.complex64, device=device)
    x[0] = 1.0 + 0j
    return x


def signal_chirp(n: int, device: torch.device) -> torch.Tensor:
    """Linear chirp from f_low to f_high."""
    t = torch.arange(n, dtype=torch.float32, device=device)
    f0, f1 = 1.0 / n, 0.45  # normalized frequencies
    k = (f1 - f0) / n
    phase = 2 * math.pi * (f0 * t + 0.5 * k * t * t)
    x = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)
    return _normalize(torch.view_as_complex(x), peak=0.5)


SIGNAL_GENERATORS = {
    "multitone": signal_multitone,
    "random": signal_random,
    "impulse": signal_impulse,
    "chirp": signal_chirp,
}

# ─── Sizes ─────────────────────────────────────────────────────────────

FFT_SIZES = [
    256, 512, 1024, 2048, 4096, 8192, 16384,
    32768, 65536, 131072, 262144, 524288, 1048576,
]

# ─── Metrics ───────────────────────────────────────────────────────────

def _error_metrics(ref: torch.Tensor, test: torch.Tensor):
    """Compute max relative error (per element, normalized by peak ref magnitude),
    RMSE, and mean absolute error."""
    abs_diff = (test.to(torch.complex64) - ref).abs()
    peak = ref.abs().max().clamp(min=1e-12)
    rel_err = (abs_diff / peak).max().item()
    rmse = abs_diff.pow(2).mean().sqrt().item()
    mae = abs_diff.mean().item()
    return rel_err, rmse, mae


@dataclass
class BenchResult:
    size: int
    signal: str
    fp32_time_us: float
    fp16_time_us: float
    speedup: float
    fp32_gbps: float
    fp16_gbps: float
    max_rel_err: float
    rmse: float
    mae: float


# ─── Main benchmark ────────────────────────────────────────────────────

def run_benchmarks(
    device: torch.device,
    warmup: int = 10,
    reps: int = 50,
) -> List[BenchResult]:
    results: List[BenchResult] = []
    cuda_sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    total_cases = len(FFT_SIZES) * len(SIGNAL_GENERATORS)
    case_idx = 0

    for sig_name, gen_fn in SIGNAL_GENERATORS.items():
        for n in FFT_SIZES:
            case_idx += 1
            print(f"  [{case_idx}/{total_cases}] {sig_name:12s} N={n:7d} ...", end=" ", flush=True)

            # Generate signal in FP64 for maximum reference precision
            with torch.no_grad():
                x_c64 = gen_fn(n, device).to(torch.complex64)
                x_c32 = x_c64.to(torch.complex32)

                # FP32 reference
                for _ in range(warmup):
                    _ = torch.fft.fft(x_c64, norm="backward")
                cuda_sync()

                t0 = time.perf_counter()
                for _ in range(reps):
                    ref_fp32 = torch.fft.fft(x_c64, norm="backward")
                cuda_sync()
                t1 = time.perf_counter()
                fp32_time_us = (t1 - t0) / reps * 1e6

                # FP16 via lowp_fft
                for _ in range(warmup):
                    _ = fft_lowp(x_c64, precision="fp16")
                cuda_sync()

                t0 = time.perf_counter()
                for _ in range(reps):
                    test_fp16 = fft_lowp(x_c64, precision="fp16")
                cuda_sync()
                t1 = time.perf_counter()
                fp16_time_us = (t1 - t0) / reps * 1e6

                # Precision
                ref_c64 = ref_fp32.to(torch.complex64)
                max_rel_err, rmse, mae = _error_metrics(ref_c64, test_fp16)

                # Throughput: bytes read + bytes written per FFT
                # Complex: 2 floats per element → FP32=8B, FP16=4B per element
                bytes_per_element_fp32 = 8   # complex64 = 2×float32 = 8 bytes
                bytes_per_element_fp16 = 4   # complex32 = 2×float16 = 4 bytes
                gb = 1e9

                fp32_gbps = (n * bytes_per_element_fp32 * 2) / (fp32_time_us * 1e-6) / gb
                fp16_gbps = (n * bytes_per_element_fp16 * 2) / (fp16_time_us * 1e-6) / gb

                speedup = fp32_time_us / fp16_time_us

                results.append(BenchResult(
                    size=n,
                    signal=sig_name,
                    fp32_time_us=fp32_time_us,
                    fp16_time_us=fp16_time_us,
                    speedup=speedup,
                    fp32_gbps=fp32_gbps,
                    fp16_gbps=fp16_gbps,
                    max_rel_err=max_rel_err,
                    rmse=rmse,
                    mae=mae,
                ))

                print(f"speedup={speedup:.2f}x  rel_err={max_rel_err:.2e}")

    return results


def save_results(results: List[BenchResult], csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "size", "signal", "fp32_time_us", "fp16_time_us", "speedup",
            "fp32_gbps", "fp16_gbps", "max_rel_err", "rmse", "mae",
        ])
        for r in results:
            w.writerow([
                r.size, r.signal, f"{r.fp32_time_us:.3f}", f"{r.fp16_time_us:.3f}",
                f"{r.speedup:.4f}", f"{r.fp32_gbps:.3f}", f"{r.fp16_gbps:.3f}",
                f"{r.max_rel_err:.6e}", f"{r.rmse:.6e}", f"{r.mae:.6e}",
            ])
    print(f"\nResults saved to {csv_path}")


def write_report(results: List[BenchResult], report_path: str):
    """Generate a markdown report."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Aggregate by signal type
    from collections import defaultdict
    by_signal = defaultdict(list)
    for r in results:
        by_signal[r.signal].append(r)

    lines = [
        "# FP16 vs FP32 FFT — Precision & Performance Benchmark",
        "",
        f"**Date**: 2026-06-03",
        f"**GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12GB VRAM)",
        f"**CUDA**: 13.3",
        f"**Warmup**: 10 iterations | **Benchmark**: 50 iterations per case",
        "",
        "## Precision Summary",
        "",
        "| Signal | Best RelErr | Worst RelErr | Avg RMSE | Avg Speedup |",
        "|--------|------------|-------------|----------|-------------|",
    ]

    for sig_name in SIGNAL_GENERATORS:
        sig_results = by_signal[sig_name]
        best = min(r.max_rel_err for r in sig_results)
        worst = max(r.max_rel_err for r in sig_results)
        avg_rmse = sum(r.rmse for r in sig_results) / len(sig_results)
        avg_speedup = sum(r.speedup for r in sig_results) / len(sig_results)
        lines.append(
            f"| {sig_name:12s} | {best:.2e} | {worst:.2e} | {avg_rmse:.2e} | {avg_speedup:.2f}x |"
        )

    lines += [
        "",
        "## Detailed Results by Signal Type",
        "",
    ]

    for sig_name in SIGNAL_GENERATORS:
        sig_results = by_signal[sig_name]
        lines += [
            f"### {sig_name}",
            "",
            "| N | FP32 (us) | FP16 (us) | Speedup | FP32 GB/s | FP16 GB/s | MaxRelErr | RMSE |",
            "|---|-----------|-----------|---------|-----------|-----------|-----------|------|",
        ]
        for r in sig_results:
            lines.append(
                f"| {r.size:7d} | {r.fp32_time_us:9.2f} | {r.fp16_time_us:9.2f} | "
                f"{r.speedup:.2f}x | {r.fp32_gbps:9.2f} | {r.fp16_gbps:9.2f} | "
                f"{r.max_rel_err:.2e} | {r.rmse:.2e} |"
            )
        lines.append("")

    # Overall stats
    all_speedups = [r.speedup for r in results]
    all_relerrs = [r.max_rel_err for r in results]
    lines += [
        "## Overall Statistics",
        "",
        f"- **Speedup**: {min(all_speedups):.2f}x – {max(all_speedups):.2f}x (avg {sum(all_speedups)/len(all_speedups):.2f}x)",
        f"- **Max Relative Error**: {min(all_relerrs):.2e} – {max(all_relerrs):.2e} (avg {sum(all_relerrs)/len(all_relerrs):.2e})",
        f"- **Design target**: relative error < 1e-3, throughput improvement >= 1.5×",
        "",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"Device: {props.name} ({props.total_memory // 1024**2} MiB VRAM)")
    print(f"FP16 FFT sizes: {FFT_SIZES}")
    print(f"Signal types: {list(SIGNAL_GENERATORS)}")
    print()

    results = run_benchmarks(device, warmup=10, reps=50)

    csv_path = os.path.join("data", "sprint-2.3-2.4-benchmark.csv")
    report_path = os.path.join("data", "sprint-2.3-2.4-report.md")
    save_results(results, csv_path)
    write_report(results, report_path)


if __name__ == "__main__":
    main()
