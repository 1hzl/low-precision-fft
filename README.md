# Low Precision FFT for PyTorch

Low-precision FFT research project: FP32, FP16, BF16, and Block Floating-Point (BFP) FP8
FFT implementations for PyTorch, targeting LLM fine-tuning with circulant adapters.

**GPU**: NVIDIA RTX 5070 Ti (SM_120, 12 GB VRAM, CUDA 13.3)
**Reference**: arXiv 2505.00582 — Block Circulant Adapter for LLMs

## Supported Precisions

| Precision | Backend          | Bytes/elem | FFT  | IFFT | SQNR (typical, N=1024) |
|-----------|------------------|------------|------|------|------------------------|
| FP32      | torch.fft (cuFFT)| 8          | yes  | yes  | reference              |
| FP16      | cuFFT Xt         | 4          | yes  | yes  | 56–61 dB               |
| BF16      | cuFFT Xt         | 4          | yes  | yes  | 53.1 dB                |
| BFP FP8   | Custom CUDA      | 2          | yes  | yes  | 20–22 dB               |

- **FP16/BF16** use native cuFFT Xt (SM_80+) via a PyTorch C++ extension with autograd support.
- **BFP FP8** uses a custom block floating-point Radix-2 DIT FFT kernel where each stage shares
  one integer exponent. Butterflies run in float32, outputs are requantized to FP8 mantissas.
  A pure-Python CPU prototype is also available for experimentation.

*SQNR measured against FP64 reference FFT on RTX 5070 Ti (SM_120, CUDA 13.3).*  
*Sources: `docs/sprint-3.4-final-report.md` (FP16, BFP FP8), `LAPTOP-CHANGES.md` (BF16).*

## Installation

```bash
pip install -e .
```

Requires:
- Python >= 3.10
- PyTorch >= 2.0 with CUDA
- CUDA Toolkit 13.3 + MSVC Build Tools (for cuFFT extension)
- SM_80+ GPU (for FP16/BF16 cuFFT Xt fast path)

Build the standalone BFP CUDA kernel (optional, for benchmarking):

```bash
build_bfp.bat   # requires VS Developer Command Prompt + CUDA 13.3
```

## Usage

```python
import torch
import lowp_fft

x = torch.randn(256, dtype=torch.complex64, device="cuda")

# FP32 (default) — uses torch.fft
y_fp32 = lowp_fft.fft(x)

# FP16 — uses cuFFT Xt native half-precision
y_fp16 = lowp_fft.fft(x, precision="fp16")

# BF16 — uses cuFFT Xt with bfloat16 interleaved format
y_bf16 = lowp_fft.fft(x, precision="bf16")

# BFP FP8 — uses custom block floating-point kernel (CPU prototype by default)
y_fp8 = lowp_fft.fft(x, precision="fp8")

# Inverse FFT works the same way
x_hat = lowp_fft.ifft(y_fp16, precision="fp16")
```

## Project Structure

```
low-precision-fft/
├── lowp_fft/                    # Python package
│   ├── __init__.py              # Public API: fft(), ifft()
│   ├── _autograd.py             # Autograd for FP16/BF16 cuFFT ops
│   ├── bfp_fft.py               # BFP FP8 Python/NumPy prototype
│   └── csrc/
│       └── cufft_fp16.cu        # cuFFT Xt PyTorch C++ extension
├── src/cuda/                    # Standalone CUDA kernels
│   ├── bfp_fft.cu               # BFP FP8 Radix-2 DIT FFT + IFFT
│   └── bfp_fft.h                # Public C API
├── tests/                       # Tests and benchmarks
│   ├── test_bfp_fft.py          # BFP unit + boundary tests
│   ├── test_autograd.py         # FP16/BF16 autograd tests
│   ├── bench_bfp_throughput.py  # Throughput: BFP vs cuFFT FP16/FP32
│   └── bench_bfp_memory.py      # Memory bandwidth benchmark
├── data/                        # Benchmark results (CSV)
├── docs/                        # Design docs, error analysis
├── setup.py                     # Package install
├── build_bfp.bat                # Build standalone BFP CUDA kernel
└── TODO.md                      # Task tracking
```

## References

- [Block Circulant Adapter for LLMs](https://arxiv.org/abs/2505.00582) (arXiv 2505.00582)
- PyTorch `CuFFTPlanCache.h:308-311` — `kHalf → CUDA_C_16F` type mapping
