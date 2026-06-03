"""Low-precision FFT for PyTorch.

Provides a unified API for FP32, FP16, BF16, and (future) FP8 FFT operations.
Uses a custom cuFFT Xt extension for native FP16 when available, with fallback
to PyTorch's built-in torch.fft.
"""

import math
import os
import sys
import torch
from typing import Optional

# Ensure CUDA DLLs are findable on Windows
_cuda_dll_dir = os.path.join(
    os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3"),
    "bin/x64",
)
if os.path.isdir(_cuda_dll_dir) and sys.platform == "win32":
    os.add_dll_directory(_cuda_dll_dir)

# Try loading the custom cuFFT FP16 extension
_cufft_ext = None
try:
    import lowp_fft._cufft_ext as _cufft_ext  # noqa: F811
except ImportError:
    pass


def _maybe_complex(tensor: torch.Tensor) -> torch.Tensor:
    """Convert real tensor [..., 2] to complex if needed."""
    if tensor.is_complex():
        return tensor
    if tensor.size(-1) == 2:
        return torch.view_as_complex(tensor)
    raise ValueError(
        f"Cannot interpret tensor of shape {tensor.shape} as complex. "
        f"Expected complex dtype or last dim == 2."
    )


def _ensure_dtype(
    tensor: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    """Convert tensor to target complex dtype, handling real→complex."""
    if target_dtype == torch.complex64:
        return tensor.to(torch.complex64)
    if target_dtype == torch.complex32:
        if tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
            return torch.view_as_complex(tensor)
        if tensor.dtype == torch.float32 or tensor.dtype == torch.complex64:
            return tensor.to(torch.complex32)
        return tensor.to(torch.complex32)
    return tensor


def fft(
    input: torch.Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: str = "backward",
    *,
    precision: Optional[str] = None,
) -> torch.Tensor:
    """1D FFT with optional low-precision execution.

    Args:
        input: Input tensor (real or complex).
        n: FFT size (pads/truncates if != input size along dim).
        dim: Dimension along which to compute FFT.
        norm: Normalization mode ("forward", "backward", "ortho").
        precision: "fp32", "fp16", or "bf16". None defaults to input dtype.

    Returns:
        Complex tensor in the requested precision.
    """
    input_complex = _maybe_complex(input)

    if precision is None or precision == "fp32":
        if input_complex.dtype not in (torch.complex64, torch.complex128):
            input_complex = input_complex.to(torch.complex64)
        return torch.fft.fft(input_complex, n=n, dim=dim, norm=norm)

    if precision == "fp16":
        input_half = input_complex.to(torch.complex32)
        # Use custom cuFFT Xt extension for native FP16 execution
        if _cufft_ext is not None and input_half.is_cuda and n is None and dim == -1:
            if norm in ("backward", "ortho", "forward"):
                from lowp_fft._autograd import FFTFP16  # noqa: E402
                result = FFTFP16.apply(input_half.contiguous())
                if norm == "ortho":
                    result = result / math.sqrt(max(1, input_half.size(-1)))
                elif norm == "forward":
                    result = result / input_half.size(-1)
                return result
        return torch.fft.fft(input_half, n=n, dim=dim, norm=norm)

    if precision == "bf16":
        # BF16: PyTorch may not have native bf16 FFT; fall back to FP32 then cast
        try:
            input_bf16 = input_complex.to(
                torch.complex64
            )  # no complex32 for bf16
            result = torch.fft.fft(input_bf16, n=n, dim=dim, norm=norm)
            # Cast back: complex64 → real view → bf16 → complex view
            real_view = torch.view_as_real(result)
            bf16_view = real_view.to(torch.bfloat16)
            return torch.view_as_complex(bf16_view)
        except Exception:
            raise RuntimeError(
                "BF16 FFT not supported on this platform. "
                "Requires SM_80+ GPU with CUDA 11+."
            )

    if precision == "fp8":
        raise NotImplementedError("FP8 FFT is planned for Phase 3.")

    raise ValueError(
        f"Unknown precision '{precision}'. Choose 'fp32', 'fp16', 'bf16', or 'fp8'."
    )


def ifft(
    input: torch.Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: str = "backward",
    *,
    precision: Optional[str] = None,
) -> torch.Tensor:
    """1D inverse FFT. See :func:`fft` for parameter details."""
    input_complex = _maybe_complex(input)

    if precision is None or precision == "fp32":
        if input_complex.dtype not in (torch.complex64, torch.complex128):
            input_complex = input_complex.to(torch.complex64)
        return torch.fft.ifft(input_complex, n=n, dim=dim, norm=norm)

    if precision == "fp16":
        input_half = input_complex.to(torch.complex32)
        if _cufft_ext is not None and input_half.is_cuda and n is None and dim == -1:
            if norm in ("backward", "ortho", "forward"):
                from lowp_fft._autograd import IFFTFP16  # noqa: E402
                result = IFFTFP16.apply(input_half.contiguous())
                if norm == "ortho":
                    result = result * math.sqrt(max(1, input_half.size(-1)))
                elif norm == "forward":
                    result = result * input_half.size(-1)
                return result
        return torch.fft.ifft(input_half, n=n, dim=dim, norm=norm)

    if precision == "bf16":
        try:
            input_bf16 = input_complex.to(torch.complex64)
            result = torch.fft.ifft(input_bf16, n=n, dim=dim, norm=norm)
            real_view = torch.view_as_real(result)
            bf16_view = real_view.to(torch.bfloat16)
            return torch.view_as_complex(bf16_view)
        except Exception:
            raise RuntimeError(
                "BF16 IFFT not supported on this platform."
            )

    if precision == "fp8":
        raise NotImplementedError("FP8 IFFT is planned for Phase 3.")

    raise ValueError(f"Unknown precision '{precision}'.")


__all__ = ["fft", "ifft"]
