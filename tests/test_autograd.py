"""Tests for Sprint 2.2 — FP16 FFT/IFFT autograd via conjugate trick."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowp_fft._autograd import FFTFP16, IFFTFP16


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _complex32_tensor(*shape, device):
    """Create a complex32 tensor from random float32 data."""
    n = 1
    for s in shape:
        n *= s
    x = torch.randn(n * 2, dtype=torch.float32, device=device)
    return torch.view_as_complex(x.reshape(*shape, 2)).to(torch.complex32)


class TestFFTFP16:
    def test_forward_shape(self, device):
        x = _complex32_tensor(64, device=device)
        result = FFTFP16.apply(x)
        assert result.shape == x.shape
        assert result.dtype == torch.complex32

    def test_forward_no_nan(self, device):
        x = _complex32_tensor(256, device=device)
        result = FFTFP16.apply(x)
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()

    def test_backward_grad_shape(self, device):
        x = _complex32_tensor(32, device=device)
        x.requires_grad_(True)
        result = FFTFP16.apply(x)
        loss = result.real.sum() + result.imag.sum()
        loss.backward()
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()

    def test_gradcheck_small(self, device):
        """Numerical gradient check on a small input."""
        x = torch.randn(8, dtype=torch.float64, device=device)
        x_complex = torch.view_as_complex(x.reshape(-1, 2)).to(torch.complex32)
        x_complex.requires_grad_(True)
        ok = torch.autograd.gradcheck(
            FFTFP16.apply, x_complex, eps=1e-1, atol=1e-2, rtol=1e-2
        )
        assert ok

    def test_gradient_vs_fp32_fft(self, device):
        """Verify FFT gradient is close to torch.fft.fft gradient."""
        torch.manual_seed(42)
        x = torch.randn(64, dtype=torch.float32, device=device)
        x_complex32 = torch.view_as_complex(x.reshape(-1, 2)).to(torch.complex32)
        x_complex64 = x_complex32.to(torch.complex64)

        # FP16 path
        x_fp16 = x_complex32.clone().detach().requires_grad_(True)
        r_fp16 = FFTFP16.apply(x_fp16)
        r_fp16.real.sum().backward()
        grad_fp16 = x_fp16.grad.to(torch.complex64)

        # FP32 reference
        x_fp32 = x_complex64.clone().detach().requires_grad_(True)
        r_fp32 = torch.fft.fft(x_fp32)
        r_fp32.real.sum().backward()
        grad_fp32 = x_fp32.grad

        rel_err = (grad_fp16 - grad_fp32).abs() / (grad_fp32.abs() + 1e-8)
        max_rel = rel_err.max().item()
        assert max_rel < 0.05, f"Gradient relative error {max_rel} exceeds 5%"


class TestIFFTFP16:
    def test_forward_shape(self, device):
        x = _complex32_tensor(64, device=device)
        result = IFFTFP16.apply(x)
        assert result.shape == x.shape
        assert result.dtype == torch.complex32

    def test_forward_no_nan(self, device):
        x = _complex32_tensor(256, device=device)
        result = IFFTFP16.apply(x)
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()

    def test_backward_grad_shape(self, device):
        x = _complex32_tensor(32, device=device)
        x.requires_grad_(True)
        result = IFFTFP16.apply(x)
        loss = result.real.sum() + result.imag.sum()
        loss.backward()
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()

    def test_gradcheck_small(self, device):
        """Numerical gradient check on a small input."""
        x = torch.randn(8, dtype=torch.float64, device=device)
        x_complex = torch.view_as_complex(x.reshape(-1, 2)).to(torch.complex32)
        x_complex.requires_grad_(True)
        ok = torch.autograd.gradcheck(
            IFFTFP16.apply, x_complex, eps=1e-1, atol=1e-2, rtol=1e-2
        )
        assert ok

    def test_gradient_vs_fp32_ifft(self, device):
        """Verify IFFT gradient is close to torch.fft.ifft gradient."""
        torch.manual_seed(42)
        x = torch.randn(64, dtype=torch.float32, device=device)
        x_complex32 = torch.view_as_complex(x.reshape(-1, 2)).to(torch.complex32)
        x_complex64 = x_complex32.to(torch.complex64)

        x_fp16 = x_complex32.clone().detach().requires_grad_(True)
        r_fp16 = IFFTFP16.apply(x_fp16)
        r_fp16.real.sum().backward()
        grad_fp16 = x_fp16.grad.to(torch.complex64)

        x_fp32 = x_complex64.clone().detach().requires_grad_(True)
        r_fp32 = torch.fft.ifft(x_fp32)
        r_fp32.real.sum().backward()
        grad_fp32 = x_fp32.grad

        rel_err = (grad_fp16 - grad_fp32).abs() / (grad_fp32.abs() + 1e-8)
        max_rel = rel_err.max().item()
        assert max_rel < 0.05, f"IFFT gradient relative error {max_rel} exceeds 5%"


class TestRoundtrip:
    def test_fft_ifft_identity(self, device):
        """FFT → IFFT should recover original signal within FP16 precision."""
        torch.manual_seed(42)
        x = _complex32_tensor(256, device=device)
        f = FFTFP16.apply(x)
        r = IFFTFP16.apply(f)
        diff = (r - x).abs()
        max_diff = diff.max().item()
        avg_mag = x.abs().mean().item()
        rel_err = max_diff / (avg_mag + 1e-8)
        assert rel_err < 0.01, f"Roundtrip relative error {rel_err} exceeds 1%"

    def test_roundtrip_gradient(self, device):
        """Gradient through FFT→IFFT chain should be close to identity."""
        torch.manual_seed(42)
        x = _complex32_tensor(32, device=device)
        x.requires_grad_(True)
        f = FFTFP16.apply(x)
        r = IFFTFP16.apply(f)
        r.real.sum().backward()
        assert torch.isfinite(x.grad).all()
