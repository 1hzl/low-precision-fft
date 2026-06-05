"""Block Floating-Point Radix-2 DIT FFT in software FP8 (E4M3).

Reference implementation for Sprint 3.2 — proves FP8 + BFP improves SQNR
over naive FP8 FFT. Pure Python/NumPy; no CUDA dependency.

FP8 E4M3 format: 1 sign, 4 exponent (bias=7), 3 mantissa.
Max normal: 448.0 (E=15, M=6).  NaN encoding: E=15, M=7 (clamped to 0).
Min subnormal: 2^-9 ≈ 0.00195.
"""

import numpy as np

FP8_MAX = 448.0


# ── FP8 E4M3 Quantization ────────────────────────────────────────────

def _build_fp8_table():
    """Build sorted array of all 256 FP8 E4M3 values."""
    vals = []
    for bits in range(256):
        s = (bits >> 7) & 1
        e = (bits >> 3) & 0xF
        m = bits & 0x7
        if e == 0:
            val = (2.0 ** (-6)) * (m / 8.0)
        else:
            val = (2.0 ** (e - 7)) * (1.0 + m / 8.0)
        if s:
            val = -val
        vals.append(val)
    vals[0xFF] = 0.0  # negative NaN → 0
    vals[0x7F] = 0.0  # positive NaN → 0
    return np.array(vals, dtype=np.float64)


_FP8_TABLE = _build_fp8_table()
_POS_VALS = np.unique(np.sort(_FP8_TABLE[0:128]))
_BOUNDARIES = np.zeros(len(_POS_VALS))
_BOUNDARIES[0] = 0.0
for _i in range(1, len(_POS_VALS)):
    _BOUNDARIES[_i] = (_POS_VALS[_i - 1] + _POS_VALS[_i]) / 2.0


def _quantize_real(x):
    """Quantize real float/array to nearest FP8 E4M3 via searchsorted."""
    scalar = np.ndim(x) == 0
    arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    arr = np.clip(arr, -FP8_MAX, FP8_MAX)
    sign = np.where(arr >= 0, 1.0, -1.0)
    abs_arr = np.abs(arr)
    idx = np.searchsorted(_BOUNDARIES, abs_arr, side='right') - 1
    idx = np.clip(idx, 0, len(_POS_VALS) - 1)
    result = sign * _POS_VALS[idx]
    result[abs_arr == 0] = 0.0
    if scalar:
        return float(result.flat[0])
    return result


def quantize_fp8_e4m3(x):
    """Quantize float/complex scalar or array to FP8 E4M3."""
    is_complex = np.iscomplexobj(x)
    if is_complex:
        real_q = _quantize_real(x.real)
        imag_q = _quantize_real(x.imag)
        if np.ndim(x) == 0:
            return complex(real_q, imag_q)
        return real_q + 1j * imag_q
    return _quantize_real(x)


# ── Shared Exponent ──────────────────────────────────────────────────

def compute_shared_exponent(values, fp8_max=FP8_MAX):
    """Compute shared integer exponent E so all values/2^E fit in FP8 range.

    Targets mantissas in [fp8_max/2, fp8_max] for maximum precision.
    """
    if np.iscomplexobj(values):
        max_abs = max(float(np.max(np.abs(values.real))),
                      float(np.max(np.abs(values.imag))))
    else:
        max_abs = float(np.max(np.abs(values)))
    if max_abs == 0.0:
        return 0
    E = int(np.floor(np.log2(max_abs / fp8_max)))
    while max_abs / (2.0 ** E) > fp8_max:
        E += 1
    return E


# ── Bit Reversal ─────────────────────────────────────────────────────

def _bit_reverse_array(x):
    """Bit-reversal permutation on NumPy array."""
    N = len(x)
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            x[i], x[j] = x[j].copy(), x[i].copy()
    return x


# ── BFP FFT ──────────────────────────────────────────────────────────

class BFPFFT:
    """Block Floating-Point Radix-2 DIT FFT in software FP8 (E4M3).

    Each FFT stage shares one integer exponent. At the start of each stage,
    mantissas are dequantized to float64, all butterflies run in float,
    then the stage outputs are quantized back to FP8 mantissas with a new
    shared exponent. This limits FP8 quantization to once per value per
    stage rather than once per arithmetic operation.

    Parameters
    ----------
    N : int
        FFT size (must be a power of 2).
    """

    def __init__(self, N):
        if N & (N - 1) != 0:
            raise ValueError(f"N={N} must be a power of 2")
        self.N = N
        self.log2N = int(np.log2(N))
        self.exponents = []  # populated during forward/inverse

    def forward(self, x):
        """Compute BFP forward FFT.

        Parameters
        ----------
        x : ndarray
            Complex input array of length N.

        Returns
        -------
        X : ndarray
            Complex FFT result computed via FP8 BFP.
        """
        return self._run(x, inverse=False)

    def inverse(self, x):
        """Compute BFP inverse FFT."""
        return self._run(x, inverse=True)

    def _run(self, x, inverse):
        x = np.asarray(x, dtype=np.complex128).copy()
        if len(x) != self.N:
            raise ValueError(f"Input length {len(x)} != N={self.N}")
        N = self.N
        self.exponents = []

        # 1. Bit-reversal permutation
        x = _bit_reverse_array(x)

        # 2. Initial BFP encoding: quantize input to mantissa + exponent
        E = compute_shared_exponent(x)
        self.exponents.append(E)
        scale = 2.0 ** E
        for i in range(N):
            x[i] = quantize_fp8_e4m3(x[i].real / scale) + \
                   1j * quantize_fp8_e4m3(x[i].imag / scale)

        # 3. Stage-by-stage butterfly
        step = 1
        while step < N:
            jump = step << 1
            if inverse:
                twiddle_base = np.exp(2j * np.pi / jump)
            else:
                twiddle_base = np.exp(-2j * np.pi / jump)

            # Dequantize all mantissas to float for this stage
            scale_in = 2.0 ** E
            for i in range(N):
                x[i] = x[i] * scale_in

            # Run all butterflies in float64 (no per-op FP8 quantization)
            for group in range(0, N, jump):
                w = 1.0 + 0.0j
                for pair in range(step):
                    idx_a = group + pair
                    idx_b = idx_a + step
                    A = x[idx_a]
                    B = x[idx_b]
                    x[idx_a] = A + w * B
                    x[idx_b] = A - w * B
                    w = w * twiddle_base

            # After stage: compute new shared exponent, quantize to FP8 mantissas
            E = compute_shared_exponent(x)
            self.exponents.append(E)
            scale_out = 2.0 ** E
            for i in range(N):
                x[i] = quantize_fp8_e4m3(x[i].real / scale_out) + \
                       1j * quantize_fp8_e4m3(x[i].imag / scale_out)

            step = jump

        # Final dequantize
        result = x * (2.0 ** self.exponents[-1])
        if inverse:
            result = result / self.N
        return result
