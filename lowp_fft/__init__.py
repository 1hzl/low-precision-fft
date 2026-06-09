"""Low-precision FFT for PyTorch.

Provides a unified API for FP32, FP16, BF16, and (future) FP8 FFT operations.
Uses a custom cuFFT Xt extension for native FP16 when available, with fallback
to PyTorch's built-in torch.fft.
"""

import math
import os
import sys
import warnings
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


def _fp16_fft_impl(
    input_complex: torch.Tensor,
    n: Optional[int],
    dim: int,
    norm: str,
    direction: str,
) -> torch.Tensor:
    """Shared FP16 FFT/IFFT implementation used by fft() and ifft().

    Uses cuFFT Xt extension for native FP16 when available (n=None, dim=-1),
    with torch.fft fallback otherwise.
    """
    if input_complex.dtype == torch.complex32:
        input_half = input_complex
    else:
        input_half = input_complex.to(torch.complex32)

    fast_path = (
        _cufft_ext is not None
        and input_half.is_cuda
        and n is None
        and dim == -1
    )

    if fast_path and norm in ("backward", "ortho", "forward"):
        contig = input_half.contiguous()
        if torch.is_grad_enabled():
            from lowp_fft._autograd import FFTFP16, IFFTFP16  # noqa: E402
            cls = FFTFP16 if direction == "forward" else IFFTFP16
            result = cls.apply(contig)
        else:
            fn = _cufft_ext.fft_fp16_forward if direction == "forward" else _cufft_ext.ifft_fp16_forward
            result = fn(contig)

        n_dim = float(input_half.size(dim))
        if direction == "inverse":
            # cuFFT Xt returns unnormalised IFFT; match torch "backward" /N default
            result = result / n_dim
            if norm == "ortho":
                result = result * math.sqrt(max(1, n_dim))
            elif norm == "forward":
                result = result * n_dim
        else:
            # cuFFT Xt returns unnormalised FFT; match torch norm conventions
            if norm == "ortho":
                result = result / math.sqrt(max(1, n_dim))
            elif norm == "forward":
                result = result / n_dim
        return result

    else:
        reasons = []
        if _cufft_ext is None:
            reasons.append("cuFFT FP16 extension not loaded")
        if not input_half.is_cuda:
            reasons.append("input is not CUDA")
        if n is not None:
            reasons.append(f"n={n} (only n=None supported)")
        if dim != -1:
            reasons.append(f"dim={dim} (only dim=-1 supported)")
        if not reasons:
            reasons.append(f"norm='{norm}' not in valid fast-path modes "
                           "(backward, ortho, forward)")
        warnings.warn(
            f"cuFFT FP16 fast path unavailable ({'; '.join(reasons)}); "
            f"falling back to torch.fft. Use n=None, dim=-1, CUDA tensor "
            f"for native FP16 acceleration.",
            UserWarning,
        )

    return torch.fft.fft(input_half, n=n, dim=dim, norm=norm) if direction == "forward" else torch.fft.ifft(input_half, n=n, dim=dim, norm=norm)


def _supports_bf16_cufft() -> bool:
    """Return True if GPU compute capability >= 8.0 (Ampere+)."""
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def _bf16_fft_impl(
    input_complex: torch.Tensor,
    n: Optional[int],
    dim: int,
    norm: str,
    direction: str,
) -> torch.Tensor:
    """Shared BF16 FFT/IFFT implementation used by fft() and ifft().

    Uses cuFFT Xt extension for native BF16 when available (n=None, dim=-1),
    with FP32 torch.fft fallback otherwise.

    Input: complex64 tensor [..., N]. Output: complex64 (BF16 precision).
    The extension works with bfloat16 real-interleaved [..., N, 2] internally.
    """
    if input_complex.dtype not in (torch.complex64, torch.complex128):
        input_complex = input_complex.to(torch.complex64)

    fast_path = (
        _cufft_ext is not None
        and hasattr(_cufft_ext, 'fft_bf16_forward')
        and input_complex.is_cuda
        and n is None
        and dim == -1
        and _supports_bf16_cufft()
    )

    if fast_path and norm in ("backward", "ortho", "forward"):
        real_view = torch.view_as_real(input_complex)
        input_bf16 = real_view.contiguous().to(torch.bfloat16)

        if torch.is_grad_enabled():
            from lowp_fft._autograd import FFTBF16, IFFTBF16  # noqa: E402
            cls = FFTBF16 if direction == "forward" else IFFTBF16
            result_bf16 = cls.apply(input_bf16)
        else:
            fn = _cufft_ext.fft_bf16_forward if direction == "forward" else _cufft_ext.ifft_bf16_forward
            result_bf16 = fn(input_bf16)

        result_float = result_bf16.to(torch.float32)
        result = torch.view_as_complex(result_float)

        n_dim = float(input_complex.size(dim))
        if direction == "inverse":
            result = result / n_dim
            if norm == "ortho":
                result = result * math.sqrt(max(1, n_dim))
            elif norm == "forward":
                result = result * n_dim
        else:
            if norm == "ortho":
                result = result / math.sqrt(max(1, n_dim))
            elif norm == "forward":
                result = result / n_dim
        return result

    else:
        reasons = []
        if _cufft_ext is None:
            reasons.append("cuFFT extension not loaded")
        elif not hasattr(_cufft_ext, 'fft_bf16_forward'):
            reasons.append("BF16 extension not built (rebuild with updated sources)")
        if not input_complex.is_cuda:
            reasons.append("input is not CUDA")
        if n is not None:
            reasons.append(f"n={n} (only n=None supported)")
        if dim != -1:
            reasons.append(f"dim={dim} (only dim=-1 supported)")
        if input_complex.is_cuda and not _supports_bf16_cufft():
            major, _ = torch.cuda.get_device_capability()
            reasons.append(
                f"GPU sm_{major}0 does not support BF16 cuFFT "
                f"(requires Ampere sm_80+)"
            )
        if not reasons:
            reasons.append(f"norm='{norm}' not in valid fast-path modes "
                           "(backward, ortho, forward)")
        warnings.warn(
            f"cuFFT BF16 fast path unavailable ({'; '.join(reasons)}); "
            f"falling back to torch.fft FP32 compute + BF16 truncate.",
            UserWarning,
        )

    # Fallback: FP32 compute + BF16 truncate
    try:
        input_fp32 = input_complex.to(torch.complex64)
        result = torch.fft.fft(input_fp32, n=n, dim=dim, norm=norm) if direction == "forward" else torch.fft.ifft(input_fp32, n=n, dim=dim, norm=norm)
        real_view = torch.view_as_real(result)
        bf16_trunc = real_view.to(torch.bfloat16).to(torch.float32)
        return torch.view_as_complex(bf16_trunc)
    except (TypeError, RuntimeError, ValueError) as e:
        raise RuntimeError(
            f"BF16 {'FFT' if direction == 'forward' else 'IFFT'} not supported on this platform. "
            f"Requires SM_80+ GPU with CUDA 11+. Original error: {e}"
        )


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
        precision: "fp32", "fp16", "bf16", or "fp8". None defaults to input dtype.

    Returns:
        Complex tensor in the requested precision.
    """
    input_complex = _maybe_complex(input)

    if precision is None or precision == "fp32":
        if input_complex.dtype not in (torch.complex64, torch.complex128):
            input_complex = input_complex.to(torch.complex64)
        return torch.fft.fft(input_complex, n=n, dim=dim, norm=norm)

    if precision == "fp16":
        return _fp16_fft_impl(input_complex, n, dim, norm, "forward")

    if precision == "bf16":
        return _bf16_fft_impl(input_complex, n, dim, norm, "forward")

    if precision == "fp8":
        from lowp_fft.bfp_fft import BFPFFT
        x_np = input_complex.cpu().numpy()
        bfp = BFPFFT(x_np.shape[-1])
        result_np = bfp.forward(x_np)
        result = torch.from_numpy(result_np).to(input_complex.device)
        if norm == "ortho":
            n_dim = float(input_complex.size(dim))
            result = result / math.sqrt(max(1, n_dim))
        elif norm == "forward":
            n_dim = float(input_complex.size(dim))
            result = result / n_dim
        return result

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
        return _fp16_fft_impl(input_complex, n, dim, norm, "inverse")

    if precision == "bf16":
        return _bf16_fft_impl(input_complex, n, dim, norm, "inverse")

    if precision == "fp8":
        from lowp_fft.bfp_fft import BFPFFT
        x_np = input_complex.cpu().numpy()
        bfp = BFPFFT(x_np.shape[-1])
        result_np = bfp.inverse(x_np)
        result = torch.from_numpy(result_np).to(input_complex.device)
        if norm == "ortho":
            n_dim = float(input_complex.size(dim))
            result = result * math.sqrt(max(1, n_dim))
        elif norm == "forward":
            n_dim = float(input_complex.size(dim))
            result = result * n_dim
        return result

    raise ValueError(f"Unknown precision '{precision}'.")


__all__ = ["fft", "ifft"]
