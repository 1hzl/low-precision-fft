"""torch.autograd.Function wrappers for FP16/BF16 FFT/IFFT via cuFFT.

Forward pass executes in low precision for speed. Backward pass uses PyTorch's
built-in FP32 FFT for gradient correctness — the cuFFT extension has
a CUDA stream compatibility issue when called from autograd backward()
(error code 6 / CUFFT_EXEC_FAILED). Once the extension is rebuilt with
cufftSetStream support, backward can switch to low precision.

The backward formula (verified against torch.fft):
    backward(FFT)(grad) = conj(FFT(conj(grad))) = N * IFFT(grad)
    backward(IFFT)(grad) = conj(FFT(conj(grad)))
"""

import os
import sys
import torch


class FFTFP16(torch.autograd.Function):
    """1D FFT in FP16 (backward norm — no normalisation on forward pass)."""

    @staticmethod
    def forward(ctx, input):
        # Lazy import to avoid circular dependency at module level
        from lowp_fft import _cufft_ext as _ext  # noqa: E402
        ctx._saved_n = input.size(-1)
        result = _ext.fft_fp16_forward(input.contiguous())
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # Use FP32 torch.fft for backward (cuFFT has stream issues in autograd)
        grad = grad_output.to(torch.complex64).conj()
        grad = torch.fft.fft(grad, n=ctx._saved_n, norm="backward")
        return grad.conj().to(grad_output.dtype)


class IFFTFP16(torch.autograd.Function):
    """1D IFFT in FP16 (backward norm — 1/N normalisation on forward pass)."""

    @staticmethod
    def forward(ctx, input):
        from lowp_fft import _cufft_ext as _ext  # noqa: E402
        n = input.size(-1)
        ctx._saved_n = n
        result = _ext.ifft_fp16_forward(input.contiguous())
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.to(torch.complex64).conj()
        grad = torch.fft.fft(grad, n=ctx._saved_n, norm="backward")
        return grad.conj().to(grad_output.dtype)


class FFTBF16(torch.autograd.Function):
    """1D FFT in BF16 (backward norm — no normalisation on forward pass).

    Input/output are bfloat16 tensors of shape [..., N, 2] (real/imag interleaved).
    """

    @staticmethod
    def forward(ctx, input):
        from lowp_fft import _cufft_ext as _ext  # noqa: E402
        ctx._saved_n = input.size(-2)
        result = _ext.fft_bf16_forward(input.contiguous())
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_complex = torch.view_as_complex(grad_output.to(torch.float32))
        grad = grad_complex.conj()
        grad = torch.fft.fft(grad, n=ctx._saved_n, norm="backward")
        return torch.view_as_real(grad.conj()).to(torch.bfloat16)


class IFFTBF16(torch.autograd.Function):
    """1D IFFT in BF16 (backward norm — 1/N normalisation on forward pass).

    Input/output are bfloat16 tensors of shape [..., N, 2] (real/imag interleaved).
    """

    @staticmethod
    def forward(ctx, input):
        from lowp_fft import _cufft_ext as _ext  # noqa: E402
        ctx._saved_n = input.size(-2)
        result = _ext.ifft_bf16_forward(input.contiguous())
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_complex = torch.view_as_complex(grad_output.to(torch.float32))
        grad = grad_complex.conj()
        grad = torch.fft.fft(grad, n=ctx._saved_n, norm="backward")
        return torch.view_as_real(grad.conj()).to(torch.bfloat16)
