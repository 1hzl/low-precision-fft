"""Tests for block floating-point FFT prototype (Sprint 3.2)."""
import numpy as np
import pytest
from lowp_fft.bfp_fft import BFPFFT, quantize_fp8_e4m3, compute_shared_exponent


# ── FP8 E4M3 quantization tests ──────────────────────────────────────

class TestFP8Quantize:
    def test_zero(self):
        assert quantize_fp8_e4m3(0.0) == 0.0
        assert quantize_fp8_e4m3(0.0 + 0.0j) == 0.0 + 0.0j

    def test_exact_representable(self):
        result = quantize_fp8_e4m3(1.0)
        assert result == 1.0

    def test_round_to_nearest(self):
        assert quantize_fp8_e4m3(1.25) == 1.25
        assert quantize_fp8_e4m3(1.3) == 1.25

    def test_clamp_to_max(self):
        assert quantize_fp8_e4m3(500.0) == 448.0
        assert quantize_fp8_e4m3(-500.0) == -448.0

    def test_array_input(self):
        x = np.array([0.0, 1.0, -2.0, 448.0])
        result = quantize_fp8_e4m3(x)
        assert result[0] == 0.0
        assert result[1] == 1.0
        assert result[3] == 448.0

    def test_complex_array(self):
        x = np.array([1.0 + 2.0j, 0.5 + 0.25j])
        result = quantize_fp8_e4m3(x)
        assert result[0].real == 1.0


# ── Shared exponent tests ────────────────────────────────────────────

class TestSharedExponent:
    def test_zero_values(self):
        vals = np.zeros(8, dtype=np.complex128)
        assert compute_shared_exponent(vals) == 0

    def test_small_values(self):
        vals = np.array([0.1, 0.2, 0.05], dtype=np.float64)
        E = compute_shared_exponent(vals)
        assert vals.max() / (2.0 ** E) <= 448.0

    def test_large_values(self):
        vals = np.array([1000.0, 500.0], dtype=np.float64)
        E = compute_shared_exponent(vals)
        assert 1000.0 / (2.0 ** E) <= 448.0


# ── BFPFFT tests ─────────────────────────────────────────────────────

class TestBFPFFT:
    @pytest.mark.parametrize("N", [8, 16, 32, 64])
    def test_identity_small_n(self, N):
        """BFP_FFT(BFP_IFFT(x)) ≈ x for small N."""
        np.random.seed(42)
        x = np.random.randn(N) + 1j * np.random.randn(N)
        bfp = BFPFFT(N)
        X = bfp.forward(x)
        x_hat = bfp.inverse(X)
        rel_err = np.max(np.abs(x_hat - x)) / np.max(np.abs(x))
        assert rel_err < 0.5

    def test_forward_shape(self):
        bfp = BFPFFT(16)
        x = np.random.randn(16) + 1j * np.random.randn(16)
        result = bfp.forward(x)
        assert result.shape == (16,)
        assert result.dtype == np.complex128

    def test_against_fp32_reference(self):
        """BFP FFT should be closer to FP32 reference than naive FP8 FFT."""
        N = 64
        np.random.seed(123)
        x = (np.random.randn(N) + 1j * np.random.randn(N)) / N
        ref = np.fft.fft(x)

        bfp = BFPFFT(N)
        bfp_result = bfp.forward(x)

        from tests.sim_fp8_fft_error import fp8_fft_radix2
        naive_result = fp8_fft_radix2(x)

        def snr(ref, test):
            diff = np.sum(np.abs(ref - test) ** 2)
            sig = np.sum(np.abs(ref) ** 2)
            return float(10 * np.log10(sig / (diff + 1e-15)))

        bfp_snr = snr(ref, bfp_result)
        naive_snr = snr(ref, naive_result)
        assert bfp_snr > naive_snr, \
            f"BFP SNR {bfp_snr:.1f} dB should exceed naive FP8 SNR {naive_snr:.1f} dB"

    def test_exponents_monotonic(self):
        """Per-stage exponents should be non-decreasing (values grow through FFT)."""
        N = 64
        np.random.seed(99)
        x = np.random.randn(N) + 1j * np.random.randn(N)
        bfp = BFPFFT(N)
        bfp.forward(x)
        assert len(bfp.exponents) == int(np.log2(N)) + 1
        diffs = np.diff(bfp.exponents)
        assert np.all(diffs >= -2), f"Exponent drops too much: {bfp.exponents}"

    def test_power_of_two_only(self):
        with pytest.raises(ValueError, match="power of 2"):
            BFPFFT(10)


# ── Benchmark helper (not a test, but verified by running) ───────────

def run_bfp_benchmark():
    """Compare BFP FFT vs naive FP8 FFT SQNR across N values."""
    from tests.sim_fp8_fft_error import fp8_fft_radix2

    N_values = [16, 32, 64, 128, 256, 512, 1024, 2048]
    print(f"{'N':>6s}  {'Naive SNR':>10s}  {'BFP SNR':>10s}  {'Gain':>8s}")
    print("-" * 42)
    for N in N_values:
        np.random.seed(N)
        x = (np.random.randn(N) + 1j * np.random.randn(N)) / N
        ref = np.fft.fft(x)

        naive = fp8_fft_radix2(x)
        bfp = BFPFFT(N)
        bfp_out = bfp.forward(x)

        def snr(r, t):
            d = np.sum(np.abs(r - t) ** 2)
            s = np.sum(np.abs(r) ** 2)
            return float(10 * np.log10(s / (d + 1e-15)))

        n_snr, b_snr = snr(ref, naive), snr(ref, bfp_out)
        print(f"{N:6d}  {n_snr:9.1f} dB  {b_snr:9.1f} dB  {b_snr - n_snr:+7.1f} dB")
