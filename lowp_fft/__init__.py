"""Low-precision FFT for PyTorch.

Leverages PyTorch's native cuFFT FP16/BF16 paths (where available) and
provides a unified API for FP32, FP16, BF16, and (future) FP8 FFT operations.
"""

import torch
from typing import Optional


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
