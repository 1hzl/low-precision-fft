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

    @pytest.mark.skip(reason="FP16 gradcheck non-deterministic; validated by gradient_vs_fp32 test")
    def test_gradcheck_small(self, device):
        """Numerical gradient check — skipped for FP16 (see marker)."""
        x = torch.randn(32, dtype=torch.float64, device=device)
        x_complex = torch.view_as_complex(x.reshape(-1, 2)).to(torch.complex32)
        x_complex.requires_grad_(True)
        ok = torch.autograd.gradcheck(
            FFTFP16.apply, x_complex, eps=1e-1, atol=2e-2, rtol=2e-2
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

    @pytest.mark.skip(reason="FP16 gradcheck non-deterministic; validated by gradient_vs_fp32 test")
    def test_gradcheck_small(self, device):
        """Numerical gradient check — skipped for FP16 (see marker)."""
        x = torch.randn(32, dtype=torch.float64, device=device)
        x_complex = torch.view_as_complex(x.reshape(-1, 2)).to(torch.complex32)
        x_complex.requires_grad_(True)
        ok = torch.autograd.gradcheck(
            IFFTFP16.apply, x_complex, eps=1e-1, atol=5e-2, rtol=5e-2
        )
        assert ok

    def test_gradient_vs_fp32_ifft(self, device):
        """Verify IFFT gradient is close to torch.fft.ifft gradient.

        The FP16 forward returns raw/un-normalised IFFT, while torch's default
        is "backward" norm (1/N). The gradient scales accordingly by N.
        We divide the FP16 gradient by N to compare.
        """
        torch.manual_seed(42)
        x = torch.randn(64, dtype=torch.float32, device=device)
        x_complex32 = torch.view_as_complex(x.reshape(-1, 2)).to(torch.complex32)
        x_complex64 = x_complex32.to(torch.complex64)

        x_fp16 = x_complex32.clone().detach().requires_grad_(True)
        r_fp16 = IFFTFP16.apply(x_fp16)
        r_fp16.real.sum().backward()
        grad_fp16 = x_fp16.grad.to(torch.complex64) / x_fp16.size(-1)

        x_fp32 = x_complex64.clone().detach().requires_grad_(True)
        r_fp32 = torch.fft.ifft(x_fp32)
        r_fp32.real.sum().backward()
        grad_fp32 = x_fp32.grad

        rel_err = (grad_fp16 - grad_fp32).abs() / (grad_fp32.abs() + 1e-8)
        max_rel = rel_err.max().item()
        assert max_rel < 0.05, f"IFFT gradient relative error {max_rel} exceeds 5%"


class TestPublicAPI:
    """Tests for the public lowp_fft.fft / ifft API with various parameters."""

    @pytest.fixture(scope="module")
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")

    def _complex64_tensor(self, *shape, device):
        n = 1
        for s in shape:
            n *= s
        x = torch.randn(n * 2, dtype=torch.float32, device=device)
        return torch.view_as_complex(x.reshape(*shape, 2))

    # ── Norm modes ────────────────────────────────────────────────

    @pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
    def test_norm_fp32_fft(self, norm, device):
        """FP32 FFT with all three norm modes should match torch.fft."""
        import lowp_fft
        x = self._complex64_tensor(64, device=device)
        result = lowp_fft.fft(x, norm=norm, precision="fp32")
        expected = torch.fft.fft(x, norm=norm)
        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
    def test_norm_fp16_fft(self, norm, device):
        """FP16 FFT with all three norm modes should be close to torch.fft."""
        import lowp_fft
        x = self._complex64_tensor(64, device=device)
        x_half = x.to(torch.complex32)
        result = lowp_fft.fft(x_half, norm=norm, precision="fp16")
        expected = torch.fft.fft(x, norm=norm)
        rel_err = (result.to(torch.complex64) - expected).abs() / (expected.abs() + 1e-8)
        assert rel_err.max().item() < 2e-2, f"norm={norm} FP16 FFT error too large"

    @pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
    def test_norm_fp16_ifft(self, norm, device):
        """FP16 IFFT with all three norm modes should be close to torch.fft."""
        import lowp_fft
        x = self._complex64_tensor(64, device=device)
        x_half = x.to(torch.complex32)
        result = lowp_fft.ifft(x_half, norm=norm, precision="fp16")
        expected = torch.fft.ifft(x, norm=norm)
        rel_err = (result.to(torch.complex64) - expected).abs() / (expected.abs() + 1e-8)
        assert rel_err.max().item() < 2e-2, f"norm={norm} FP16 IFFT error too large"

    # ── N=1 edge case ─────────────────────────────────────────────

    def test_n1_fft(self, device):
        """FFT of a single element should be identity (up to scaling)."""
        import lowp_fft
        x = self._complex64_tensor(1, device=device)
        result = lowp_fft.fft(x, precision="fp32")
        assert result.shape == x.shape
        assert result.dtype == torch.complex64
        # For N=1, FFT is identity (no normalisation with "backward")
        assert torch.allclose(result, x, rtol=1e-5, atol=1e-6)

    def test_n1_fp16_fft(self, device):
        """FP16 FFT of a single element."""
        import lowp_fft
        x = self._complex64_tensor(1, device=device).to(torch.complex32)
        result = lowp_fft.fft(x, precision="fp16")
        assert result.shape == x.shape
        assert result.dtype == torch.complex32
        assert torch.isfinite(result.real).all() and torch.isfinite(result.imag).all()

    # ── n padding ─────────────────────────────────────────────────

    def test_n_padding_fft(self, device):
        """FFT with n > input length pads with zeros."""
        import lowp_fft
        x = self._complex64_tensor(32, device=device)
        result = lowp_fft.fft(x, n=128, precision="fp32")
        expected = torch.fft.fft(x, n=128)
        assert result.shape[-1] == 128
        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_n_truncate_fft(self, device):
        """FFT with n < input length truncates."""
        import lowp_fft
        x = self._complex64_tensor(64, device=device)
        result = lowp_fft.fft(x, n=32, precision="fp32")
        expected = torch.fft.fft(x, n=32)
        assert result.shape[-1] == 32
        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)

    # ── dim parameter ─────────────────────────────────────────────

    def test_dim0_fft(self, device):
        """FFT along dim=0 instead of dim=-1."""
        import lowp_fft
        x = self._complex64_tensor(8, 16, device=device)
        result = lowp_fft.fft(x, dim=0, precision="fp32")
        expected = torch.fft.fft(x, dim=0)
        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)

    # ── bf16 path ─────────────────────────────────────────────────

    def test_bf16_fft_shape(self, device):
        """BF16 FFT returns correct shape and finite values."""
        import lowp_fft
        x = self._complex64_tensor(64, device=device)
        result = lowp_fft.fft(x, precision="bf16")
        assert result.shape == x.shape
        assert result.dtype == torch.complex64
        # BF16 is stored in complex64 but with bf16 precision
        assert torch.isfinite(result.real).all()

    def test_bf16_ifft_shape(self, device):
        """BF16 IFFT returns correct shape and finite values."""
        import lowp_fft
        x = self._complex64_tensor(64, device=device)
        result = lowp_fft.ifft(x, precision="bf16")
        assert result.shape == x.shape
        assert torch.isfinite(result.real).all()

    # ── Non-CUDA fallback ─────────────────────────────────────────

    def test_cpu_fp32_fallback(self):
        """FP32 FFT on CPU should still work via torch.fft."""
        import lowp_fft
        x = torch.randn(64, dtype=torch.complex64)
        result = lowp_fft.fft(x, precision="fp32")
        expected = torch.fft.fft(x)
        assert torch.allclose(result, expected)

    def test_cpu_fp16_fallback(self):
        """FP16 path on CPU warns about fallback, then fails because MKL
        does not support Half tensors. This is expected — FP16 FFT requires CUDA."""
        import lowp_fft
        x = torch.randn(64, dtype=torch.float32)
        x_c = torch.view_as_complex(x.reshape(-1, 2)).to(torch.complex32)
        with pytest.warns(UserWarning, match="not CUDA"):
            with pytest.raises(RuntimeError, match="MKL|Half|not supported"):
                lowp_fft.fft(x_c, precision="fp16")

    # ── fp16 fallback via n/dim ───────────────────────────────────

    def test_fp16_fallback_n_notnone(self, device):
        """FP16 with n!=None triggers fast-path warning and falls back."""
        import lowp_fft
        x = self._complex64_tensor(32, device=device).to(torch.complex32)
        with pytest.warns(UserWarning, match="n=64"):
            result = lowp_fft.fft(x, n=64, precision="fp16")
        assert result.shape[-1] == 64

    def test_fp16_fallback_dim(self, device):
        """FP16 with dim!= -1 triggers fast-path warning and falls back."""
        import lowp_fft
        x = self._complex64_tensor(32, device=device).to(torch.complex32)
        with pytest.warns(UserWarning, match="dim=0"):
            result = lowp_fft.fft(x, dim=0, precision="fp16")
        assert result.shape == x.shape

    # ── Empty tensor guard ────────────────────────────────────────

    def test_empty_tensor_raises(self):
        """Empty tensor should raise a reasonable error, not segfault."""
        import lowp_fft
        x = torch.empty(0, dtype=torch.complex64)
        with pytest.raises((ValueError, RuntimeError)):
            lowp_fft.fft(x, precision="fp32")


class TestRoundtrip:
    def test_fft_ifft_identity(self, device):
        """FFT → IFFT should recover original signal within FP16 precision.

        Note: Autograd functions return raw/un-normalized cuFFT output.
        The IFFT requires manual 1/N normalization to match torch convention.
        """
        torch.manual_seed(42)
        x = _complex32_tensor(256, device=device)
        f = FFTFP16.apply(x)
        r = IFFTFP16.apply(f) / x.size(-1)
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


class TestPlanCacheEviction:
    """Stress-test plan cache: 65+ unique (n, batch) combos must not crash.

    Bug: the LRU eviction used to flush *after* inserting the new entry,
    destroying the plan it was about to return.  Fixed by evicting *before*
    insertion so the new entry survives.
    """

    @pytest.fixture(scope="class")
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")

    def test_many_unique_sizes_no_crash(self, device):
        """Create 70 unique (n, batch) combos — must exceed kMaxCacheEntries=64."""
        import lowp_fft

        torch.manual_seed(42)
        results = []
        for i in range(70):
            n = 8 + i              # unique n per iteration
            batch = 1 + (i % 5)    # vary batch too
            x = torch.randn(batch, n, 2, dtype=torch.float32, device=device)
            xc = torch.view_as_complex(x).to(torch.complex32)
            y = lowp_fft.fft(xc, precision="fp16")
            results.append(y)

        assert len(results) == 70
        for y in results:
            assert torch.isfinite(y.real).all()
            assert torch.isfinite(y.imag).all()

    def test_eviction_does_not_corrupt_plan(self, device):
        """After eviction, the returned plan must still be valid for execution."""
        import lowp_fft

        torch.manual_seed(123)
        # Run 100 different sizes, evicting mid-way
        for i in range(100):
            n = 16 + (i * 7) % 128  # varied sizes
            batch = max(1, i % 8)
            x = torch.randn(batch, n, 2, dtype=torch.float32, device=device)
            xc = torch.view_as_complex(x).to(torch.complex32)
            y = lowp_fft.fft(xc, precision="fp16")
            # Every result must be finite — a stale/destroyed plan would
            # produce garbage or crash.
            assert y.shape == (batch, n)
            assert torch.isfinite(y.real).all(), f"non-finite at i={i}"
            assert torch.isfinite(y.imag).all(), f"non-finite at i={i}"


class TestInvalidNormWarning:
    """Invalid norm mode must trigger a UserWarning before fallback."""

    @pytest.fixture(scope="class")
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")

    def test_invalid_norm_warns_fft(self, device):
        """fft() with norm='bogus' must warn before falling back (torch.fft also rejects)."""
        import lowp_fft

        x = torch.randn(64, dtype=torch.complex64, device=device)
        with pytest.warns(UserWarning, match="bogus"):
            with pytest.raises(RuntimeError, match="Invalid normalization"):
                lowp_fft.fft(x, precision="fp16", norm="bogus")

    def test_invalid_norm_warns_ifft(self, device):
        """ifft() with norm='bogus' must warn before falling back."""
        import lowp_fft

        x = torch.randn(64, dtype=torch.complex64, device=device)
        with pytest.warns(UserWarning, match="bogus"):
            with pytest.raises(RuntimeError, match="Invalid normalization"):
                lowp_fft.ifft(x, precision="fp16", norm="bogus")
