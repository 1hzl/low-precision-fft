"""Block Floating-Point Radix-2 DIT FFT in software FP8 (E4M3).

Reference implementation for Sprint 3.2 — proves FP8 + BFP improves SQNR
over naive FP8 FFT. Pure Python/NumPy; no CUDA dependency.

FP8 E4M3 format: 1 sign, 4 exponent (bias=7), 3 mantissa.
Max normal: 448.0 (E=15, M=6).  NaN encoding: E=15, M=7 (clamped to 0).
Min subnormal: 2^-9 ≈ 0.00195.

Since Sprint 4.3: parameterized FP format support for ablation studies.
"""

import numpy as np


# ── FP Format Class ───────────────────────────────────────────────────

class FPFormat:
    """Configurable floating-point format with e_bits exponent, m_bits mantissa.

    Parameters
    ----------
    e_bits : int
        Number of exponent bits.
    m_bits : int
        Number of mantissa bits.
    """

    def __init__(self, e_bits, m_bits):
        self.e_bits = e_bits
        self.m_bits = m_bits
        self.bias = (1 << (e_bits - 1)) - 1
        self.n_levels = 1 << (1 + e_bits + m_bits)

        max_e = (1 << e_bits) - 1
        max_m = (1 << m_bits) - 1
        m_scale = float(1 << m_bits)

        # Max normal: e=max_e, m=max_m-1 (reserve e=max_e, m=max_m for NaN)
        self.max_val = (2.0 ** (max_e - self.bias)) * (1.0 + (max_m - 1) / m_scale)

        self._table, self._pos_vals, self._boundaries = self._build()

    def _build(self):
        max_e = (1 << self.e_bits) - 1
        max_m = (1 << self.m_bits) - 1
        m_scale = float(1 << self.m_bits)

        vals = []
        for bits in range(self.n_levels):
            s = (bits >> (self.e_bits + self.m_bits)) & 1
            e = (bits >> self.m_bits) & max_e
            m = bits & max_m

            if e == 0:
                val = (2.0 ** (1 - self.bias)) * (m / m_scale)
            else:
                val = (2.0 ** (e - self.bias)) * (1.0 + m / m_scale)

            if s:
                val = -val
            vals.append(val)

        # NaN → 0
        pos_nan = (max_e << self.m_bits) | max_m
        neg_nan = pos_nan | (1 << (self.e_bits + self.m_bits))
        vals[pos_nan] = 0.0
        vals[neg_nan] = 0.0

        table = np.array(vals, dtype=np.float64)
        pos_vals = np.unique(np.sort(table[:self.n_levels // 2]))

        boundaries = np.zeros(len(pos_vals))
        boundaries[0] = 0.0
        for i in range(1, len(pos_vals)):
            boundaries[i] = (pos_vals[i - 1] + pos_vals[i]) / 2.0

        return table, pos_vals, boundaries

    def quantize_real(self, x):
        """Quantize real float/array to nearest FP value."""
        scalar = np.ndim(x) == 0
        arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        arr = np.clip(arr, -self.max_val, self.max_val)
        sign = np.where(arr >= 0, 1.0, -1.0)
        abs_arr = np.abs(arr)
        idx = np.searchsorted(self._boundaries, abs_arr, side='right') - 1
        idx = np.clip(idx, 0, len(self._pos_vals) - 1)
        result = sign * self._pos_vals[idx]
        result[abs_arr == 0] = 0.0
        if scalar:
            return float(result.flat[0])
        return result

    def quantize(self, x):
        """Quantize float/complex scalar or array."""
        is_complex = np.iscomplexobj(x)
        if is_complex:
            real_q = self.quantize_real(x.real)
            imag_q = self.quantize_real(x.imag)
            if np.ndim(x) == 0:
                return complex(real_q, imag_q)
            return real_q + 1j * imag_q
        return self.quantize_real(x)


# Default FP8 E4M3 instance
FP8_E4M3 = FPFormat(4, 3)
FP8_MAX = FP8_E4M3.max_val


# ── Backward-compatible module-level state (from FP8_E4M3) ─────────────

def _build_fp8_table():
    """Build sorted array of all 256 FP8 E4M3 values (backward compat)."""
    return FP8_E4M3._table.copy()


_FP8_TABLE = FP8_E4M3._table
_POS_VALS = FP8_E4M3._pos_vals
_BOUNDARIES = FP8_E4M3._boundaries


def _quantize_real(x):
    """Quantize real float/array to nearest FP8 E4M3 (backward compat)."""
    return FP8_E4M3.quantize_real(x)


def quantize_fp8_e4m3(x):
    """Quantize float/complex scalar or array to FP8 E4M3 (backward compat)."""
    return FP8_E4M3.quantize(x)


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
    """Block Floating-Point Radix-2 DIT FFT with parameterized FP format.

    Each FFT stage shares one integer exponent per group of elements.
    The group_size controls exponent sharing granularity:
    - None (default): one exponent per stage (whole array) — current behavior
    - 4: one exponent per 4 elements within each stage
    - 8: one exponent per 8 elements within each stage

    At the start of each stage, mantissas are dequantized to float64,
    all butterflies run in float, then the stage outputs are quantized
    back to FP mantissas with group-wise shared exponents. This limits
    FP quantization to once per value per stage rather than once per
    arithmetic operation.

    Parameters
    ----------
    N : int
        FFT size (must be a power of 2).
    e_bits : int, default 4
        Number of exponent bits in the FP format.
    m_bits : int, default 3
        Number of mantissa bits in the FP format.
    group_size : int or None, default None
        Exponent sharing granularity. None = per-stage (1 exponent/stage),
        or a divisor of N (e.g. 4, 8) for group-wise exponents.
    """

    def __init__(self, N, e_bits=4, m_bits=3, group_size=None):
        if N & (N - 1) != 0:
            raise ValueError(f"N={N} must be a power of 2")
        if group_size is not None:
            if group_size <= 0 or N % group_size != 0:
                raise ValueError(
                    f"group_size={group_size} must evenly divide N={N}")
            if group_size & (group_size - 1) != 0:
                raise ValueError(
                    f"group_size={group_size} must be a power of 2")
        self.N = N
        self.log2N = int(np.log2(N))
        self.e_bits = e_bits
        self.m_bits = m_bits
        self.group_size = group_size
        self.fmt = FPFormat(e_bits, m_bits)
        self.exponents = []

    @property
    def max_val(self):
        return self.fmt.max_val

    def forward(self, x):
        """Compute BFP forward FFT.

        Parameters
        ----------
        x : ndarray
            Complex input array of length N.

        Returns
        -------
        X : ndarray
            Complex FFT result computed via BFP.
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
        gs = self.group_size
        n_groups = N // gs if gs else 1
        self.exponents = []
        fp8_max = self.fmt.max_val

        # 1. Bit-reversal permutation
        x = _bit_reverse_array(x)

        # 2. Initial BFP encoding: quantize input to mantissa + exponent(s)
        if gs is None:
            E = compute_shared_exponent(x, fp8_max=fp8_max)
            self.exponents.append(E)
            scale = 2.0 ** E
            for i in range(N):
                x[i] = self.fmt.quantize(x[i].real / scale) + \
                       1j * self.fmt.quantize(x[i].imag / scale)
        else:
            stage_E = np.zeros(n_groups, dtype=np.int32)
            for g in range(n_groups):
                idx0 = g * gs
                idx1 = idx0 + gs
                E = compute_shared_exponent(x[idx0:idx1], fp8_max=fp8_max)
                stage_E[g] = E
                scale = 2.0 ** E
                for i in range(idx0, idx1):
                    x[i] = self.fmt.quantize(x[i].real / scale) + \
                           1j * self.fmt.quantize(x[i].imag / scale)
            self.exponents.append(stage_E)

        # 3. Stage-by-stage butterfly
        step = 1
        while step < N:
            jump = step << 1
            if inverse:
                twiddle_base = np.exp(2j * np.pi / jump)
            else:
                twiddle_base = np.exp(-2j * np.pi / jump)

            # Dequantize all mantissas to float for this stage
            if gs is None:
                scale_in = 2.0 ** self.exponents[-1]
                for i in range(N):
                    x[i] = x[i] * scale_in
            else:
                prev_E = self.exponents[-1]
                for g in range(n_groups):
                    idx0 = g * gs
                    idx1 = idx0 + gs
                    scale_in = 2.0 ** float(prev_E[g])
                    for i in range(idx0, idx1):
                        x[i] = x[i] * scale_in

            # Run all butterflies in float64 (no per-op FP quantization)
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

            # After stage: compute shared exponent(s), quantize to FP mantissas
            if gs is None:
                E = compute_shared_exponent(x, fp8_max=fp8_max)
                self.exponents.append(E)
                scale_out = 2.0 ** E
                for i in range(N):
                    x[i] = self.fmt.quantize(x[i].real / scale_out) + \
                           1j * self.fmt.quantize(x[i].imag / scale_out)
            else:
                stage_E = np.zeros(n_groups, dtype=np.int32)
                for g in range(n_groups):
                    idx0 = g * gs
                    idx1 = idx0 + gs
                    E = compute_shared_exponent(x[idx0:idx1], fp8_max=fp8_max)
                    stage_E[g] = E
                    scale_out = 2.0 ** E
                    for i in range(idx0, idx1):
                        x[i] = self.fmt.quantize(x[i].real / scale_out) + \
                               1j * self.fmt.quantize(x[i].imag / scale_out)
                self.exponents.append(stage_E)

            step = jump

        # Final dequantize
        if gs is None:
            result = x * (2.0 ** self.exponents[-1])
        else:
            last_E = self.exponents[-1]
            for g in range(n_groups):
                idx0 = g * gs
                idx1 = idx0 + gs
                x[idx0:idx1] = x[idx0:idx1] * (2.0 ** float(last_E[g]))
            result = x
        if inverse:
            result = result / self.N
        return result
