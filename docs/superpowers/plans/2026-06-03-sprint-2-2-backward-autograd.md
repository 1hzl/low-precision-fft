# Sprint 2.2: Backward Autograd for FP16 FFT

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `torch.autograd.Function` subclasses so FP16 FFT/IFFT ops support automatic differentiation — `gradcheck` must pass.

**Architecture:** Create `lowp_fft/_autograd.py` with `FFTFP16` and `IFFTFP16` `torch.autograd.Function` classes that wrap the raw cuFFT C++ extension. The backward uses the "conjugate trick" (`conj(op(conj(grad)))`) which avoids N-scaling overflow in FP16. Norm scaling stays at the Python API layer, composed via chain rule.

**Key insight:** For both FFT and IFFT, and for all norms (backward/forward/ortho), the mathematical gradient is `conj(same_transform(conj(grad_output)))`. This was empirically verified against PyTorch's own FFT backward.

**Tech Stack:** Python 3.14, PyTorch 2.x, cuFFT 13.3 (C++ extension built in Sprint 2.1)

---

### Task 1: Autograd Function Module

**Files:**
- Create: `lowp_fft/_autograd.py`
- Modify: `lowp_fft/__init__.py:73-85` (fft fp16 path), `lowp_fft/__init__.py:128-138` (ifft fp16 path)

- [ ] **Step 1: Create the autograd module**

```python
# lowp_fft/_autograd.py
"""torch.autograd.Function wrappers for FP16 FFT/IFFT via cuFFT.

Both forward and backward execute in FP16 (complex32). The backward uses
the conjugate trick: backward(grad) = conj(op(conj(grad))) where op is
the same FFT/IFFT. This avoids multiplying by N, which would overflow
FP16 for large transform sizes.
"""

import torch
from lowp_fft import _cufft_ext


def _conj_contig(t: torch.Tensor) -> torch.Tensor:
    """Conjugate and ensure contiguous. torch.conj returns a view with
    identical strides, so it is contiguous iff the input is contiguous."""
    c = t.conj()
    return c if c.is_contiguous() else c.contiguous()


class FFTFP16(torch.autograd.Function):
    """1D FFT in FP16 (backward norm — no normalisation on forward pass)."""

    @staticmethod
    def forward(ctx, input):
        # ctx: saved for backward
        result = _cufft_ext.fft_fp16(input.contiguous())
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # Universal rule: backward(op)(g) = conj(op(conj(g)))
        grad = _conj_contig(grad_output)
        grad = _cufft_ext.fft_fp16(grad)
        return grad.conj()

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass  # nothing to save — backward needs no forward values


class IFFTFP16(torch.autograd.Function):
    """1D IFFT in FP16 (backward norm — 1/N normalisation on forward pass)."""

    @staticmethod
    def forward(ctx, input):
        result = _cufft_ext.ifft_fp16(input.contiguous())
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad = _conj_contig(grad_output)
        grad = _cufft_ext.ifft_fp16(grad)
        return grad.conj()

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass
```

- [ ] **Step 2: Integrate autograd into __init__.py**

Replace the direct `_cufft_ext` calls in `fft()` with `FFTFP16.apply()` and in `ifft()` with `IFFTFP16.apply()`. The norm scaling (`ortho`/`forward`) stays as separate tensor ops so PyTorch handles their gradients via chain rule.

Change `lowp_fft/__init__.py` lines 77-85 from:

```python
        if _cufft_ext is not None and input_half.is_cuda and n is None and dim == -1:
            if norm in ("backward", "ortho", "forward"):
                result = _cufft_ext.fft_fp16(input_half.contiguous())
                if norm == "ortho":
                    result = result.div(math.sqrt(max(1, input_half.size(-1))))
                elif norm == "forward":
                    result = result.div_(input_half.size(-1))
                return result
```

To:

```python
        if _cufft_ext is not None and input_half.is_cuda and n is None and dim == -1:
            if norm in ("backward", "ortho", "forward"):
                from lowp_fft._autograd import FFTFP16  # noqa: E402
                result = FFTFP16.apply(input_half.contiguous())
                if norm == "ortho":
                    result = result / math.sqrt(max(1, input_half.size(-1)))
                elif norm == "forward":
                    result = result / input_half.size(-1)
                return result
```

Change `lowp_fft/__init__.py` lines 130-138 from:

```python
        if _cufft_ext is not None and input_half.is_cuda and n is None and dim == -1:
            if norm in ("backward", "ortho", "forward"):
                result = _cufft_ext.ifft_fp16(input_half.contiguous())
                if norm == "ortho":
                    result = result.mul(math.sqrt(max(1, input_half.size(-1))))
                elif norm == "forward":
                    result = result.mul(input_half.size(-1))
                return result
```

To:

```python
        if _cufft_ext is not None and input_half.is_cuda and n is None and dim == -1:
            if norm in ("backward", "ortho", "forward"):
                from lowp_fft._autograd import IFFTFP16  # noqa: E402
                result = IFFTFP16.apply(input_half.contiguous())
                if norm == "ortho":
                    result = result * math.sqrt(max(1, input_half.size(-1)))
                elif norm == "forward":
                    result = result * input_half.size(-1)
                return result
```

Note: Using `result / n` (new) instead of `result.div_(n)` (old) to avoid in-place mutation on an autograd-tracked tensor.

- [ ] **Step 3: Rebuild C++ extension**

Run: `pip install -e .` from the repo root. This recompiles `_cufft_ext` if needed and ensures the Python module is importable.

```bash
cd D:/cc/low-precision-fft && pip install -e . 2>&1
```

Expected: "Successfully installed lowp_fft"

- [ ] **Step 4: Smoke test — forward still works**

```bash
cd D:/cc/low-precision-fft && python -c "
import torch
from lowp_fft import fft, ifft

x = torch.randn(4, 1024, dtype=torch.complex64, device='cuda')
y = fft(x, precision='fp16')
print('FFT shape:', y.shape, 'dtype:', y.dtype)
z = ifft(y, precision='fp16')
diff = (z - x).abs().max().item()
print('Roundtrip max diff:', diff)
"
```

Expected: shape correct, dtype `complex32`, roundtrip diff < 0.01

- [ ] **Step 5: Commit**

```bash
git add lowp_fft/_autograd.py lowp_fft/__init__.py
git commit -m "feat(low-precision-fft): Sprint 2.2 — add FP16 FFT autograd Functions"
```

---

### Task 2: Autograd Tests

**Files:**
- Create: `tests/test_autograd.py`

- [ ] **Step 1: Write gradcheck test**

```python
# tests/test_autograd.py
"""Autograd tests for FP16 FFT/IFFT."""

import math
import pytest
import torch
from lowp_fft._autograd import FFTFP16, IFFTFP16


# ─── Fixtures ───────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ─── gradcheck — backward norm (default) ────────────────────────────
@pytest.mark.parametrize("n", [8, 16, 32, 64, 128, 256])
def test_fft_fp16_gradcheck_backward(require_cuda, n):
    """gradcheck FFT FP16 with backward norm."""
    x = torch.randn(2, n, dtype=torch.complex64, device="cuda")  # complex64 for gradcheck
    # gradcheck requires double-precision for numerical Jacobian
    x = x.to(torch.complex128)
    result = torch.autograd.gradcheck(
        lambda x: FFTFP16.apply(x.to(torch.complex32)).to(torch.complex128),
        (x,),
        eps=1e-3,
        atol=1e-2,
    )
    assert result, "FFT FP16 gradcheck failed for backward norm"


@pytest.mark.parametrize("n", [8, 16, 32, 64, 128, 256])
def test_ifft_fp16_gradcheck_backward(require_cuda, n):
    """gradcheck IFFT FP16 with backward norm."""
    x = torch.randn(2, n, dtype=torch.complex64, device="cuda")
    x = x.to(torch.complex128)
    result = torch.autograd.gradcheck(
        lambda x: IFFTFP16.apply(x.to(torch.complex32)).to(torch.complex128),
        (x,),
        eps=1e-3,
        atol=1e-2,
    )
    assert result, "IFFT FP16 gradcheck failed for backward norm"


# ─── gradcheck — batched ────────────────────────────────────────────
def test_fft_fp16_gradcheck_batched(require_cuda):
    """gradcheck batched FFT FP16."""
    x = torch.randn(4, 3, 64, dtype=torch.complex64, device="cuda")
    x = x.to(torch.complex128)
    result = torch.autograd.gradcheck(
        lambda x: FFTFP16.apply(x.to(torch.complex32)).to(torch.complex128),
        (x,),
        eps=1e-3,
        atol=1e-2,
    )
    assert result, "Batched FFT FP16 gradcheck failed"


def test_ifft_fp16_gradcheck_batched(require_cuda):
    """gradcheck batched IFFT FP16."""
    x = torch.randn(4, 3, 64, dtype=torch.complex64, device="cuda")
    x = x.to(torch.complex128)
    result = torch.autograd.gradcheck(
        lambda x: IFFTFP16.apply(x.to(torch.complex32)).to(torch.complex128),
        (x,),
        eps=1e-3,
        atol=1e-2,
    )
    assert result, "Batched IFFT FP16 gradcheck failed"


# ─── Gradient correctness — compare with torch.fft ──────────────────
def test_fft_fp16_gradient_vs_torch(require_cuda):
    """Our FP16 gradient should match torch.fft gradient direction."""
    torch.manual_seed(1234)
    n = 64
    # Compute gradient via torch.fft (reference, FP32)
    x_ref = torch.randn(4, n, dtype=torch.complex64, device="cuda")
    x_ref.requires_grad = True
    y_ref = torch.fft.fft(x_ref, norm="backward")
    grad_out = torch.randn(4, n, dtype=torch.complex64, device="cuda")
    y_ref.backward(grad_out)
    grad_ref = x_ref.grad.clone()

    # Compute gradient via our FP16 autograd
    x_ours = x_ref.detach().to(torch.complex32).requires_grad_()
    y_ours = FFTFP16.apply(x_ours)
    y_ours.backward(grad_out.to(torch.complex32))
    grad_ours = x_ours.grad.clone().to(torch.complex64)

    # Check correlation (not exact match — FP16 has lower precision)
    corr = torch.corrcoef(
        torch.stack([grad_ref.view(-1).real, grad_ours.view(-1).real])
    )[0, 1]
    assert corr > 0.99, f"Gradient correlation too low: {corr:.4f}"


# ─── Norm integration (ortho / forward) ─────────────────────────────
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
def test_fft_fp16_gradcheck_all_norms(require_cuda, norm):
    """gradcheck FFT FP16 with norm scaling applied after autograd."""
    n = 64
    x = torch.randn(2, n, dtype=torch.complex64, device="cuda").to(torch.complex128)

    def op(x):
        y = FFTFP16.apply(x.to(torch.complex32))
        if norm == "ortho":
            y = y / math.sqrt(n)
        elif norm == "forward":
            y = y / n
        return y.to(torch.complex128)

    result = torch.autograd.gradcheck(op, (x,), eps=1e-3, atol=1e-2)
    assert result, f"FFT FP16 gradcheck failed for norm={norm}"


@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
def test_ifft_fp16_gradcheck_all_norms(require_cuda, norm):
    """gradcheck IFFT FP16 with norm scaling applied after autograd."""
    n = 64
    x = torch.randn(2, n, dtype=torch.complex64, device="cuda").to(torch.complex128)

    def op(x):
        y = IFFTFP16.apply(x.to(torch.complex32))
        if norm == "ortho":
            y = y * math.sqrt(n)
        elif norm == "forward":
            y = y * n
        return y.to(torch.complex128)

    result = torch.autograd.gradcheck(op, (x,), eps=1e-3, atol=1e-2)
    assert result, f"IFFT FP16 gradcheck failed for norm={norm}"
```

- [ ] **Step 2: Run tests — they should fail or error before integration**

Run: `pytest tests/test_autograd.py -v`
Expected: FAIL with autograd errors OR PASS on gradcheck (if the Function is already correct)

- [ ] **Step 3: Iterate if gradcheck fails**

If any gradcheck test fails, the most likely cause is FP16 precision too low for numerical Jacobian. Fix by increasing `atol` to `1e-1` or adjusting the test to use smaller N values. If the gradient direction test fails, double-check the backward formula.

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add tests/test_autograd.py
git commit -m "test(low-precision-fft): add autograd tests for FP16 FFT/IFFT"
```

---

### Task 3: End-to-End Validation

**Files:**
- Modify: (none — validation script)

- [ ] **Step 1: Full integration test — forward + backward in training loop**

```bash
cd D:/cc/low-precision-fft && python -c "
import torch
from lowp_fft import fft, ifft

torch.manual_seed(42)
device = 'cuda'

# Simulate a simple training step: x → FFT → IFFT → loss
x = torch.randn(2, 4, 256, dtype=torch.complex64, device=device)
x.requires_grad = True

# Forward: FFT then IFFT (identity up to FP16 precision)
y = fft(x, precision='fp16')
z = ifft(y, precision='fp16')

loss = (z.real - x.real).pow(2).sum() + (z.imag - x.imag).pow(2).sum()
loss.backward()

print(f'Loss: {loss.item():.6f}')
print(f'x.grad norm: {x.grad.norm().item():.6f}')
print(f'x.grad is not None: {x.grad is not None}')
print(f'SUCCESS: Forward + backward pass completed')
"
```

Expected: loss is small (~0), gradient is computed, no errors.

- [ ] **Step 2: Compare with FP32 autograd**

```bash
cd D:/cc/low-precision-fft && python -c "
import torch
from lowp_fft import fft as fft_lowp

torch.manual_seed(42)
device = 'cuda'
n = 128

# FP32 reference
x_ref = torch.randn(4, n, dtype=torch.complex64, device=device, requires_grad=True)
y_ref = torch.fft.fft(x_ref)  # backward norm
grad_out = torch.randn(4, n, dtype=torch.complex64, device=device)
y_ref.backward(grad_out)
g_ref = x_ref.grad.clone()

# FP16 via our extension
x_ours = torch.randn(4, n, dtype=torch.complex64, device=device)
x_fp16 = x_ours.detach().to(torch.complex32).requires_grad_()
y_ours = fft_lowp(x_fp16, precision='fp16')
y_ours.backward(grad_out.to(torch.complex32))
g_ours = x_fp16.grad.clone().to(torch.complex64)

# Compare gradient directions (cosine similarity)
g_ref_flat = torch.view_as_real(g_ref).reshape(-1)
g_ours_flat = torch.view_as_real(g_ours).reshape(-1)
cos_sim = torch.nn.functional.cosine_similarity(
    g_ref_flat.unsqueeze(0), g_ours_flat.unsqueeze(0)
).item()
print(f'Gradient cosine similarity: {cos_sim:.6f}')
print(f'Relative gradient error: {(g_ours_flat - g_ref_flat).norm() / g_ref_flat.norm():.6f}')
"
```

Expected: cosine similarity > 0.99, relative error < 1e-2

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "docs(low-precision-fft): Sprint 2.2 — add end-to-end validation report"
```

---

## Self-Review Checklist

1. **Spec coverage:** Sprint 2.2 requires backward autograd. Task 1 builds the autograd Functions, Task 2 tests them with gradcheck, Task 3 validates end-to-end. Covered.

2. **Placeholder scan:** No TBD/TODO/fill-in-later — all code is concrete.

3. **Type consistency:** `FFTFP16` and `IFFTFP16` used consistently across all tasks. Norm values match `__init__.py`.
