"""torch.autograd.Function wrappers for FP16 FFT/IFFT via cuFFT.

Both forward and backward execute in FP16 (complex32). The backward uses
the conjugate trick: backward(grad) = conj(op(conj(grad))) where op is
the same FFT/IFFT. This avoids multiplying by N, which would overflow
FP16 for large transform sizes.
"""

import os
import sys
import torch

# Ensure CUDA DLLs are findable (Windows)
_cuda_dll_dir = os.path.join(
    os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3"),
    "bin/x64",
)
if os.path.isdir(_cuda_dll_dir) and sys.platform == "win32":
    os.add_dll_directory(_cuda_dll_dir)

from lowp_fft import _cufft_ext  # noqa: E402


def _conj_contig(t: torch.Tensor) -> torch.Tensor:
    """Conjugate and ensure a true copy (not just a view). cuFFT can fail
    with conjugated views even when torch reports them as contiguous."""
    return t.conj().contiguous()


class FFTFP16(torch.autograd.Function):
    """1D FFT in FP16 (backward norm — no normalisation on forward pass)."""

    @staticmethod
    def forward(ctx, input):
        result = _cufft_ext.fft_fp16_forward(input.contiguous())
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad = _conj_contig(grad_output)
        grad = _cufft_ext.fft_fp16_forward(grad)
        return grad.conj()


class IFFTFP16(torch.autograd.Function):
    """1D IFFT in FP16 (backward norm — 1/N normalisation on forward pass).

    The cuFFT extension's inverse is *unnormalised* (no 1/N), so we divide
    by N here to match PyTorch's backward norm convention.
    """

    @staticmethod
    def forward(ctx, input):
        n = input.size(-1)
        result = _cufft_ext.ifft_fp16_forward(input.contiguous())
        return result.div_(n)

    @staticmethod
    def backward(ctx, grad_output):
        n = grad_output.size(-1)
        grad = _conj_contig(grad_output)
        grad = _cufft_ext.ifft_fp16_forward(grad).div_(n)
        return grad.conj()
