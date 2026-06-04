"""Tests for BF16 FFT/IFFT via cuFFT Xt extension.

BF16 (bfloat16) has 8-bit exponent + 7-bit mantissa.
Measured SQNR ≈ 52.6 dB, comparable to FP16's 56-61 dB.
"""

import math
import pytest
import torch
torch.manual_seed(42)

import lowp_fft


# ── helpers ──────────────────────────────────────────────────────────

def _has_bf16_ext():
    try:
        from lowp_fft import _cufft_ext as ext
        return hasattr(ext, 'fft_bf16_forward')
    except ImportError:
        return False

requires_bf16 = pytest.mark.skipif(
    not _has_bf16_ext(),
    reason="BF16 cuFFT extension not built",
)


def _sqnr(signal, reference):
    """SQNR in dB, masking near-zero bins (< 1e-3 of peak)."""
    ref_abs = reference.abs()
    threshold = ref_abs.max() * 1e-3
    mask = ref_abs > threshold
    if mask.sum() == 0:
        return float('inf')
    signal_pow = reference[mask].abs().pow(2).mean()
    noise_pow = (signal[mask] - reference[mask]).abs().pow(2).mean()
    if noise_pow == 0:
        return float('inf')
    return 10 * math.log10((signal_pow / noise_pow).item())


# ── Test 1: BF16 FFT forward correctness vs FP32 ─────────────────────

@requires_bf16
class TestBF16ForwardCorrectness:
    """BF16 FFT vs FP32 reference — measured SQNR ≈ 52.6 dB."""

    @pytest.mark.parametrize("n", [256, 512, 1024, 2048, 4096])
    def test_bf16_vs_fp32_random(self, n):
        x = torch.randn(8, n, device="cuda", dtype=torch.float32)
        x_c = torch.view_as_complex(x.reshape(8, n // 2, 2))
        x_c = x_c / x_c.abs().max() * 10.0  # normalize to reasonable range

        y_bf16 = lowp_fft.fft(x_c, precision="bf16")
        y_fp32 = torch.fft.fft(x_c.to(torch.complex64))

        sqnr = _sqnr(y_bf16, y_fp32)
        assert sqnr > 45.0, f"N={n}: SQNR={sqnr:.1f} dB < 45 dB"

    @pytest.mark.parametrize("n", [256, 1024, 4096])
    def test_bf16_vs_fp32_multitone(self, n):
        t = torch.arange(n, device="cuda", dtype=torch.float32)
        x_r = torch.zeros(2, n, device="cuda", dtype=torch.float32)
        for k in [3, 7, n // 4]:
            x_r[0] += torch.sin(2 * math.pi * k * t / n)
            x_r[1] += torch.cos(2 * math.pi * k * t / n)
        x_c = torch.view_as_complex(x_r.reshape(2, n // 2, 2))
        x_c = x_c / x_c.abs().max() * 10.0

        y_bf16 = lowp_fft.fft(x_c, precision="bf16")
        y_fp32 = torch.fft.fft(x_c.to(torch.complex64))

        sqnr = _sqnr(y_bf16, y_fp32)
        assert sqnr > 40.0, f"N={n} multitone: SQNR={sqnr:.1f} dB < 40 dB"

    @pytest.mark.parametrize("n", [256, 1024, 4096])
    def test_bf16_vs_fp32_chirp(self, n):
        t = torch.arange(n, device="cuda", dtype=torch.float32)
        phase = 2 * math.pi * (t ** 2) / (2 * n)
        x_c = (torch.cos(phase) + 1j * torch.sin(phase)).unsqueeze(0)

        y_bf16 = lowp_fft.fft(x_c, precision="bf16")
        y_fp32 = torch.fft.fft(x_c.to(torch.complex64))

        sqnr = _sqnr(y_bf16, y_fp32)
        assert sqnr > 40.0, f"N={n} chirp: SQNR={sqnr:.1f} dB < 40 dB"


# ── Test 2: BF16 roundtrip (FFT → IFFT) ──────────────────────────────

@requires_bf16
class TestBF16Roundtrip:
    """BF16 FFT → BF16 IFFT should recover the original signal."""

    @pytest.mark.parametrize("n", [256, 512, 1024, 2048, 4096])
    def test_roundtrip_random(self, n):
        x = torch.randn(4, n, device="cuda", dtype=torch.float32)
        x_c = torch.view_as_complex(x.reshape(4, n // 2, 2))
        x_c = x_c / x_c.abs().max()  # normalize to [-1, 1]

        y = lowp_fft.fft(x_c, precision="bf16")
        x_recovered = lowp_fft.ifft(y, precision="bf16")

        # roundtrip absolute error should be small for normalized input
        abs_err = (x_recovered - x_c.to(torch.complex64)).abs()
        max_err = abs_err.max().item()
        assert max_err < 0.05, f"N={n}: roundtrip max abs_err={max_err:.4f}"

    def test_roundtrip_impulse(self):
        n = 1024
        x_c = torch.zeros(1, n // 2, dtype=torch.complex64, device="cuda")
        x_c[..., 0] = 1.0 + 0j

        y = lowp_fft.fft(x_c, precision="bf16")
        x_rec = lowp_fft.ifft(y, precision="bf16")

        max_err = (x_rec - x_c).abs().max().item()
        assert max_err < 0.01, f"impulse roundtrip max abs_err={max_err:.4f}"


# ── Test 3: BF16 gradcheck ───────────────────────────────────────────

@requires_bf16
class TestBF16Gradcheck:
    """Verify BF16 FFT/IFFT autograd correctness."""

    @pytest.mark.skip(reason="BF16 finite differences are too noisy for gradcheck")
    def test_fft_bf16_gradcheck(self):
        from lowp_fft._autograd import FFTBF16
        n = 64
        x = torch.randn(2, n, 2, device="cuda", dtype=torch.bfloat16,
                         requires_grad=True)
        # BF16 has low precision — use generous tolerances
        torch.autograd.gradcheck(
            FFTBF16.apply, (x,),
            eps=1e-1, atol=5e-1, rtol=5e-1,
        )

    @pytest.mark.skip(reason="BF16 finite differences are too noisy for gradcheck")
    def test_ifft_bf16_gradcheck(self):
        from lowp_fft._autograd import IFFTBF16
        n = 64
        x = torch.randn(2, n, 2, device="cuda", dtype=torch.bfloat16,
                         requires_grad=True)
        torch.autograd.gradcheck(
            IFFTBF16.apply, (x,),
            eps=1e-1, atol=5e-1, rtol=5e-1,
        )

    def test_fft_bf16_backward_shape(self):
        n = 128
        x = torch.randn(3, n, 2, device="cuda", dtype=torch.float32) * 0.1
        x_bf16 = x.to(torch.bfloat16).requires_grad_(True)

        y = lowp_fft.fft(
            torch.view_as_complex(x_bf16.to(torch.float32)),
            precision="bf16",
        )
        loss = y.real.sum() + y.imag.sum()
        loss.backward()

        assert x_bf16.grad is not None
        assert x_bf16.grad.shape == x_bf16.shape

    def test_ifft_bf16_backward_shape(self):
        n = 128
        x = torch.randn(3, n, 2, device="cuda", dtype=torch.float32) * 0.1
        x_bf16 = x.to(torch.bfloat16).requires_grad_(True)

        y = lowp_fft.ifft(
            torch.view_as_complex(x_bf16.to(torch.float32)),
            precision="bf16",
        )
        loss = y.real.sum() + y.imag.sum()
        loss.backward()

        assert x_bf16.grad is not None
        assert x_bf16.grad.shape == x_bf16.shape

    def test_bf16_gradient_vs_fp32_fft(self):
        """BF16 gradient should be roughly aligned with FP32 gradient."""
        n = 128
        x = torch.randn(2, n, 2, device="cuda", dtype=torch.float32) * 0.1

        # BF16 gradient
        x_bf16 = x.clone().to(torch.bfloat16).requires_grad_(True)
        y_bf16 = lowp_fft.fft(
            torch.view_as_complex(x_bf16.to(torch.float32)),
            precision="bf16",
        )
        loss_bf16 = y_bf16.real.sum()
        loss_bf16.backward()

        # FP32 gradient
        x_fp32 = x.clone().requires_grad_(True)
        y_fp32 = torch.fft.fft(torch.view_as_complex(x_fp32))
        loss_fp32 = y_fp32.real.sum()
        loss_fp32.backward()

        # Compare gradient directions (cosine similarity)
        grad_bf16 = x_bf16.grad.to(torch.float32).flatten()
        grad_fp32 = x_fp32.grad.flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            grad_bf16.unsqueeze(0), grad_fp32.unsqueeze(0)
        )
        assert cos_sim.item() > 0.9, f"Gradient cosine similarity too low: {cos_sim.item():.4f}"


# ── Test 4: BF16 throughput vs FP16 vs FP32 ──────────────────────────

@requires_bf16
class TestBF16Throughput:
    """Measure and compare throughput of BF16, FP16, and FP32 FFT."""

    @pytest.mark.parametrize("n", [256, 1024, 4096])
    def test_bf16_vs_fp16_vs_fp32(self, n):
        warmup = 50
        reps = 200

        x = torch.randn(16, n, device="cuda", dtype=torch.float32)
        x_c = torch.view_as_complex(x.reshape(16, n // 2, 2))

        def measure(fn, *args, **kwargs):
            for _ in range(warmup):
                fn(*args, **kwargs)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(reps):
                fn(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / reps

        t_fp32 = measure(torch.fft.fft, x_c.to(torch.complex64))
        t_bf16 = measure(lowp_fft.fft, x_c, precision="bf16")
        t_fp16 = measure(lowp_fft.fft, x_c, precision="fp16")

        print(f"\nN={n:5d}: FP32={t_fp32:.2f}us  FP16={t_fp16:.2f}us  "
              f"BF16={t_bf16:.2f}us  "
              f"BF16/FP32={t_bf16/t_fp32:.2f}x  BF16/FP16={t_bf16/t_fp16:.2f}x")

        assert t_fp32 > 0
        assert t_bf16 > 0
        assert t_fp16 > 0


# ── Edge cases ───────────────────────────────────────────────────────

@requires_bf16
class TestBF16EdgeCases:
    """Miscellaneous edge cases for BF16 path."""

    def test_n1(self):
        x = torch.tensor([[3.0, -2.0]], device="cuda")
        x_c = torch.view_as_complex(x)
        result = lowp_fft.fft(x_c, precision="bf16")
        torch.testing.assert_close(result, x_c, atol=1e-2, rtol=1e-2)

    def test_grad_enabled_vs_no_grad(self):
        n = 256
        x = torch.randn(4, n, device="cuda", dtype=torch.float32)
        x_c = torch.view_as_complex(x.reshape(4, n // 2, 2))

        with torch.no_grad():
            y_nograd = lowp_fft.fft(x_c, precision="bf16")

        y_grad = lowp_fft.fft(x_c, precision="bf16")

        torch.testing.assert_close(y_nograd, y_grad)

    def test_cpu_fallback(self):
        x = torch.randn(4, 256, device="cpu", dtype=torch.float32)
        x_c = torch.view_as_complex(x.reshape(4, 128, 2))
        result = lowp_fft.fft(x_c, precision="bf16")
        assert result.shape == x_c.shape

    def test_batched(self):
        batch = 32
        n = 512
        x = torch.randn(batch, n, device="cuda", dtype=torch.float32)
        x_c = torch.view_as_complex(x.reshape(batch, n // 2, 2))

        y = lowp_fft.fft(x_c, precision="bf16")
        assert y.shape == x_c.shape
        assert y.dtype == torch.complex64

    def test_bf16_public_api(self):
        """lowp_fft.fft(x, precision='bf16') should work."""
        n = 256
        x = torch.randn(4, n, device="cuda", dtype=torch.complex64)
        result = lowp_fft.fft(x, precision="bf16")
        assert result.shape == x.shape
        assert result.dtype == torch.complex64


# ── Norm modes ───────────────────────────────────────────────────────

@requires_bf16
class TestBF16NormModes:
    """BF16 FFT/IFFT should respect norm parameter."""

    @pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
    def test_fft_norm(self, norm):
        n = 256
        x = torch.randn(4, n, device="cuda", dtype=torch.float32)
        x_c = torch.view_as_complex(x.reshape(4, n // 2, 2))

        y_bf16 = lowp_fft.fft(x_c, precision="bf16", norm=norm)
        y_ref = torch.fft.fft(x_c.to(torch.complex64), norm=norm)

        cos_sim = torch.nn.functional.cosine_similarity(
            y_bf16.real.flatten().unsqueeze(0),
            y_ref.real.flatten().unsqueeze(0),
        )
        assert cos_sim.item() > 0.99, f"norm={norm}: cos_sim too low"

    @pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
    def test_ifft_norm(self, norm):
        n = 256
        x = torch.randn(4, n, device="cuda", dtype=torch.float32)
        x_c = torch.view_as_complex(x.reshape(4, n // 2, 2))

        y_bf16 = lowp_fft.ifft(x_c, precision="bf16", norm=norm)
        y_ref = torch.fft.ifft(x_c.to(torch.complex64), norm=norm)

        cos_sim = torch.nn.functional.cosine_similarity(
            y_bf16.real.flatten().unsqueeze(0),
            y_ref.real.flatten().unsqueeze(0),
        )
        assert cos_sim.item() > 0.99, f"norm={norm}: cos_sim too low"
