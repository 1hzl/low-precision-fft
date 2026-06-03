# Phase 2 Sprint 1: PyTorch C++ Extension Wrapping cuFFT FP16

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone `lowp_fft` Python package with a C++/CUDA extension that calls cuFFT's native FP16 API (`cufftXtExec` + `CUDA_C_16F`), providing `lowp_fft.fft(input)` for complex half tensors.

**Architecture:** A `torch.utils.cpp_extension.CUDAExtension` wraps cuFFT Xt API in C++/CUDA. A Python `torch.autograd.Function` subclass provides the forward pass (backward deferred to Sprint 2). CPU fallback delegates to float32 torch.fft.fft.

**Tech Stack:** Python 3.x, PyTorch 2.x, CUDA 13.3, cuFFT Xt API, pytest

**Files (all new):**
```
lowp_fft/
├── __init__.py          # Public API: fft()
├── _fft_ops.py          # torch.autograd.Function subclass
├── csrc/
│   └── fft_ops.cu       # CUDA/C++ extension source (~150 lines)
├── setup.py             # Build with CUDAExtension
└── tests/
    └── test_fft_ops.py  # Pytest tests
```

**Key design decisions:**
- Data pointer passthrough: PyTorch `complex32` stores interleaved real/imag pairs of `__half`, same layout cuFFT expects for `CUDA_C_16F`. We can pass `data_ptr<__half>()` directly without conversion.
- Plan caching: A global `std::unordered_map<int64_t, cufftHandle>` caches plans by FFT size. Plans are reused across calls — no plan creation overhead on repeated calls.
- Only power-of-2 sizes (cuFFT FP16 constraint).
- CPU fallback: Converts to float32, runs `torch.fft.fft`, converts back to float16. Not guarded by `#ifdef __CUDA_ARCH__` since it's host code — instead uses `#ifdef WITH_CUDA`.
- Forward only for Sprint 1. Backward pass raises `NotImplementedError` until Sprint 2.

---

### Task 1: Create package skeleton + setup.py

**Files:**
- Create: `lowp_fft/__init__.py`
- Create: `lowp_fft/setup.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p lowp_fft/csrc lowp_fft/tests
```

- [ ] **Step 2: Write setup.py**

```python
# lowp_fft/setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lowp_fft',
    version='0.1.0',
    author='low-precision-fft team',
    description='Low-precision FFT via cuFFT FP16 native API',
    ext_modules=[
        CUDAExtension(
            name='lowp_fft._fft_ops_cpp',
            sources=['csrc/fft_ops.cu'],
            extra_compile_args={
                'cxx': ['/std:c++17'],
                'nvcc': ['-arch=sm_120', '-O3'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.8',
)
```

- [ ] **Step 3: Write __init__.py placeholder**

```python
# lowp_fft/__init__.py
from ._fft_ops import fft

__all__ = ['fft']
```

- [ ] **Step 4: Verify directory structure**

```bash
ls -R lowp_fft/
```

- [ ] **Step 5: Commit**

```bash
git add lowp_fft/__init__.py lowp_fft/setup.py
git commit -m "feat(lowp-fft): add package skeleton and setup.py for CUDA extension"
```

---

### Task 2: Write C++/CUDA extension (fft_ops.cu)

**Files:**
- Create: `lowp_fft/csrc/fft_ops.cu`

- [ ] **Step 1: Write fft_ops.cu**

```cpp
// lowp_fft/csrc/fft_ops.cu
// PyTorch C++/CUDA extension wrapping cuFFT FP16 native API.
// Uses cufftXtMakePlanMany + cufftXtExec for half-precision FFT.
// CPU fallback delegates to float32 torch.fft.fft.

#include <torch/extension.h>
#include <unordered_map>
#include <mutex>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cufft.h>
#include <cufftXt.h>

#define CHECK_CUFFT(call)                                              \
    do {                                                               \
        cufftResult r = (call);                                        \
        TORCH_CHECK(r == CUFFT_SUCCESS,                                \
            "cuFFT error at ", __FILE__, ":", __LINE__, ": ", (int)r); \
    } while (0)

// Plan cache: size_n → cufftHandle, reused across calls.
static std::unordered_map<long long, cufftHandle> plan_cache;
static std::mutex plan_cache_mutex;

static cufftHandle get_or_create_plan(long long n) {
    std::lock_guard<std::mutex> lock(plan_cache_mutex);
    auto it = plan_cache.find(n);
    if (it != plan_cache.end()) return it->second;

    cufftHandle plan;
    CHECK_CUFFT(cufftCreate(&plan));
    size_t ws = 0;
    CHECK_CUFFT(cufftXtMakePlanMany(
        plan, 1, &n,
        nullptr, 1, n, CUDA_C_16F,   // input: FP16 complex interleaved
        nullptr, 1, n, CUDA_C_16F,   // output: FP16 complex interleaved
        1, &ws, CUDA_C_16F));

    if (ws > 0) {
        void* d_work = nullptr;
        cudaMalloc(&d_work, ws);
        cufftXtSetWorkArea(plan, &d_work);
        // Note: d_work is leaked intentionally — it lives as long as the plan.
        // The plan cache is never freed in this sprint (acceptable for a CLI tool).
    }

    plan_cache[n] = plan;
    return plan;
}

torch::Tensor fft_fp16_forward_cuda(torch::Tensor input) {
    // input: complex half, torch.complex32, contiguous, shape (..., N)
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kComplexHalf,
                "input must be complex half (torch.complex32)");

    auto sizes = input.sizes();
    auto n = sizes.back();  // FFT length (last dim)
    TORCH_CHECK((n & (n - 1)) == 0,
                "FFT size must be power of 2, got ", n);

    // Compute batch count
    long long batch = 1;
    for (int64_t i = 0; i < input.dim() - 1; ++i) {
        batch *= sizes[i];
    }

    // input.data_ptr() points to interleaved __half pairs (re, im, re, im, ...)
    // Same memory layout cuFFT expects for CUDA_C_16F complex.
    auto* d_input = reinterpret_cast<__half*>(input.data_ptr());

    // Allocate output
    auto output = torch::empty_like(input);
    auto* d_output = reinterpret_cast<__half*>(output.data_ptr());

    cufftHandle plan = get_or_create_plan(n);

    // For batch > 1, we reuse the single-size plan and loop.
    // Each FFT is n complex elements, interleaved = 2*n half values.
    size_t stride = n * 2;  // half-count per FFT
    for (long long b = 0; b < batch; ++b) {
        cufftResult r = cufftXtExec(plan,
            d_input + b * stride,
            d_output + b * stride,
            CUFFT_FORWARD);
        if (r == CUFFT_EXEC_FAILED) {
            // Re-create plan on failure (may need different workspace for this batch context)
            // For single-plan reuse across batches, this shouldn't happen.
            TORCH_CHECK(false, "cuFFT execution failed at batch ", b);
        }
    }

    return output;
}
#endif // WITH_CUDA

torch::Tensor fft_fp16_forward_cpu(torch::Tensor input) {
    // CPU fallback: float32 FFT via torch.fft
    auto input_f32 = input.to(torch::kComplexFloat);
    auto result_f32 = torch::fft::fft(input_f32, input_f32.size(-1), -1, "forward");
    return result_f32.to(torch::kComplexHalf);
}

// Main entry point — returns complex half tensor
torch::Tensor fft_fp16_forward(torch::Tensor input) {
    input = input.contiguous();
    if (input.is_cuda()) {
#ifdef WITH_CUDA
        return fft_fp16_forward_cuda(input);
#else
        TORCH_CHECK(false, "CUDA not available in this build");
#endif
    }
    return fft_fp16_forward_cpu(input);
}

// PyTorch extension bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fft_fp16_forward", &fft_fp16_forward,
          "FP16 FFT via cuFFT native API (CUDA) or float32 fallback (CPU)");
}
```

---

### Task 3: Write Python autograd wrapper

**Files:**
- Create: `lowp_fft/_fft_ops.py`

- [ ] **Step 1: Write _fft_ops.py**

```python
# lowp_fft/_fft_ops.py
"""torch.autograd.Function wrapper for cuFFT FP16 extension."""

import torch
from torch.autograd import Function

# Will be loaded by setup.py build, or fall back to inline compilation
try:
    from . import _fft_ops_cpp
except ImportError:
    _fft_ops_cpp = None


class FP16FFT(Function):
    """Custom autograd Function for FP16 FFT via cuFFT.

    Forward: calls cuFFT Xt API (CUDA_C_16F) directly.
    Backward: NOT YET IMPLEMENTED (deferred to Sprint 2).
    """

    @staticmethod
    def forward(ctx, input):
        """Compute FP16 FFT.

        Args:
            input: torch.Tensor of dtype torch.complex32, shape (..., N).
                   N must be power of 2.

        Returns:
            torch.Tensor of dtype torch.complex32, same shape.
        """
        if _fft_ops_cpp is None:
            raise RuntimeError(
                "lowp_fft extension not built. Run: pip install -e lowp_fft/"
            )
        ctx.save_for_backward(input)
        return _fft_ops_cpp.fft_fp16_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "FP16FFT backward not yet implemented (Sprint 2)"
        )


def fft(input):
    """FP16 FFT via cuFFT native API.

    Args:
        input (Tensor): complex half tensor, shape (..., N). N power of 2.

    Returns:
        Tensor: complex half tensor, same shape.
    """
    if input.dtype != torch.complex32:
        raise TypeError(
            f"Expected torch.complex32 input, got {input.dtype}"
        )
    n = input.size(-1)
    if (n & (n - 1)) != 0:
        raise ValueError(f"FFT size must be power of 2, got {n}")
    return FP16FFT.apply(input)
```

- [ ] **Step 2: Verify Python syntax**

```bash
python -c "import ast; ast.parse(open('lowp_fft/_fft_ops.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add lowp_fft/_fft_ops.py
git commit -m "feat(lowp-fft): add Python autograd wrapper for FP16 FFT"
```

---

### Task 4: Write tests

**Files:**
- Create: `lowp_fft/tests/__init__.py` (empty)
- Create: `lowp_fft/tests/test_fft_ops.py`

- [ ] **Step 1: Write tests/__init__.py**

```python
# lowp_fft/tests/__init__.py
```

- [ ] **Step 2: Write test_fft_ops.py**

```python
# lowp_fft/tests/test_fft_ops.py
"""Tests for lowp_fft extension."""

import pytest
import torch


def _requires_extension():
    """Skip tests if extension is not built."""
    try:
        from lowp_fft._fft_ops_cpp import fft_fp16_forward  # noqa: F401
        return True
    except ImportError:
        return False


requires_ext = pytest.mark.skipif(
    not _requires_extension(),
    reason="lowp_fft extension not built",
)


def _requires_cuda():
    return torch.cuda.is_available()


requires_cuda = pytest.mark.skipif(
    not _requires_cuda(),
    reason="CUDA not available",
)


class TestFFTCPU:
    """CPU fallback tests (always runnable, no CUDA needed)."""

    def test_basic_forward_cpu(self):
        """CPU fallback: output has correct shape and dtype."""
        from lowp_fft import fft

        n = 256
        x = torch.randn(n, dtype=torch.float32)
        x_complex = torch.complex(x, torch.zeros_like(x)).to(torch.complex32)

        y = fft(x_complex)
        assert y.shape == (n,)
        assert y.dtype == torch.complex32
        assert not torch.all(y.real == 0) or not torch.all(y.imag == 0), \
            "output should not be all zeros"

    def test_vs_reference_fp32(self):
        """CPU fallback: FP16 FFT matches FP32 FFT within tolerance."""
        from lowp_fft import fft

        n = 256
        x = torch.randn(n, dtype=torch.float32)
        x_complex = torch.complex(x, torch.zeros_like(x))

        y_fp16 = fft(x_complex.to(torch.complex32)).to(torch.complex64)
        y_fp32 = torch.fft.fft(x_complex, norm="forward")

        diff = (y_fp16 - y_fp32).abs()
        max_val = y_fp32.abs().max()
        rel_err = diff.max() / (max_val + 1e-10)

        assert rel_err < 1e-2, f"Relative error {rel_err:.2e} exceeds 1e-2"

    def test_batch_forward_cpu(self):
        """CPU fallback: batched input works."""
        from lowp_fft import fft

        n = 256
        batch = 4
        x = torch.randn(batch, n, dtype=torch.float32)
        x_complex = torch.complex(x, torch.zeros_like(x)).to(torch.complex32)

        y = fft(x_complex)
        assert y.shape == (batch, n)
        assert y.dtype == torch.complex32

    def test_power_of_two_check(self):
        """Non-power-of-2 size raises ValueError."""
        from lowp_fft import fft

        n = 100  # not power of 2
        x = torch.randn(n, dtype=torch.complex32)

        with pytest.raises(ValueError, match="power of 2"):
            fft(x)

    def test_dtype_check(self):
        """Non-complex32 input raises TypeError."""
        from lowp_fft import fft

        n = 256
        x = torch.randn(n, dtype=torch.float32)  # float32, not complex32

        with pytest.raises(TypeError, match="complex32"):
            fft(x)


@pytest.mark.skipif(not _requires_cuda(), reason="CUDA not available")
class TestFFTCUDA:
    """CUDA tests (require GPU)."""

    def test_basic_forward_cuda(self):
        """CUDA: output has correct shape and dtype."""
        from lowp_fft import fft

        n = 256
        x = torch.randn(n, dtype=torch.float32, device='cuda')
        x_complex = torch.complex(x, torch.zeros_like(x)).to(torch.complex32)

        y = fft(x_complex)
        assert y.shape == (n,)
        assert y.dtype == torch.complex32
        assert y.device.type == 'cuda'
        assert not torch.all(y.real == 0) or not torch.all(y.imag == 0)

    def test_vs_reference_fp32_cuda(self):
        """CUDA: FP16 FFT matches FP32 FFT within tolerance."""
        from lowp_fft import fft

        n = 256
        x = torch.randn(n, dtype=torch.float32, device='cuda')
        x_complex = torch.complex(x, torch.zeros_like(x))

        y_fp16 = fft(x_complex.to(torch.complex32)).to(torch.complex64)
        y_fp32 = torch.fft.fft(x_complex, norm="forward")

        diff = (y_fp16 - y_fp32).abs()
        max_val = y_fp32.abs().max()
        rel_err = diff.max() / (max_val + 1e-10)

        assert rel_err < 1e-2, f"Relative error {rel_err:.2e} exceeds 1e-2"

    def test_batch_forward_cuda(self):
        """CUDA: batched input works."""
        from lowp_fft import fft

        n = 256
        batch = 4
        x = torch.randn(batch, n, dtype=torch.float32, device='cuda')
        x_complex = torch.complex(x, torch.zeros_like(x)).to(torch.complex32)

        y = fft(x_complex)
        assert y.shape == (batch, n)
        assert y.dtype == torch.complex32
        assert y.device.type == 'cuda'

    def test_multi_batch_large(self):
        """CUDA: multi-batch large FFT."""
        from lowp_fft import fft

        n = 1024
        batch = 16
        x = torch.randn(batch, n, dtype=torch.float32, device='cuda')
        x_complex = torch.complex(x, torch.zeros_like(x)).to(torch.complex32)

        y = fft(x_complex)
        assert y.shape == (batch, n)
```

- [ ] **Step 3: Commit**

```bash
git add lowp_fft/tests/__init__.py lowp_fft/tests/test_fft_ops.py
git commit -m "test(lowp-fft): add tests for FP16 FFT extension"
```

---

### Task 5: Build, test, and verify

- [ ] **Step 1: Build extension**

```bash
cd lowp_fft && python setup.py build_ext --inplace
```
Expected: nvcc compiles `fft_ops.cu`, produces `_fft_ops_cpp.pyd` (Windows).

- [ ] **Step 2: Run CPU tests**

```bash
cd lowp_fft && python -m pytest tests/test_fft_ops.py::TestFFTCPU -v
```
Expected: 5 tests pass (basic_forward_cpu, vs_reference_fp32, batch_forward_cpu, power_of_two_check, dtype_check).

- [ ] **Step 3: Run CUDA tests**

```bash
cd lowp_fft && python -m pytest tests/test_fft_ops.py::TestFFTCUDA -v
```
Expected: 4 tests pass (basic_forward_cuda, vs_reference_fp32_cuda, batch_forward_cuda, multi_batch_large).

- [ ] **Step 4: Run all tests**

```bash
cd lowp_fft && python -m pytest tests/ -v
```
Expected: All 9 tests pass.

- [ ] **Step 5: Quick smoke test in Python REPL**

```python
import torch
from lowp_fft import fft

# CPU test
x = torch.randn(256, dtype=torch.complex32)
y = fft(x)
print(f"CPU: in={x.shape} out={y.shape} dtype={y.dtype}")

# CUDA test
x = torch.randn(1024, dtype=torch.float32, device='cuda')
x = torch.complex(x, torch.zeros_like(x)).to(torch.complex32)
y = fft(x)
print(f"CUDA: in={x.shape} out={y.shape} dtype={y.dtype} device={y.device}")
```

- [ ] **Step 6: Commit extension build artifacts (.pyd is in .gitignore, only commit .cu)**

```bash
git add lowp_fft/csrc/fft_ops.cu
git commit -m "feat(lowp-fft): add CUDA extension source for FP16 FFT"
```

---

### Task 6: Add .gitignore for build artifacts

**Files:**
- Modify: `lowp_fft/.gitignore` (new file, or modify root `.gitignore`)

- [ ] **Step 1: Create lowp_fft/.gitignore**

```
*.pyd
*.so
*.dll
*.exp
*.lib
*.pdb
build/
dist/
*.egg-info/
__pycache__/
*.pyc
```

- [ ] **Step 2: Commit**

```bash
git add lowp_fft/.gitignore
git commit -m "chore(lowp-fft): add .gitignore for build artifacts"
```

---

### Task 7: Final verification and TODO update

- [ ] **Step 1: Verify no binaries committed**

```bash
git status && git log --oneline -5
```

- [ ] **Step 2: Update LAPTOP-CHANGES.md**

Append a section documenting what was built. See LAPTOP-CHANGES.md for format.

- [ ] **Step 3: Mark TODO.md 2.1 as done**

Edit TODO.md: change `- [ ] 2.1 Sprint 1` to `- [x] 2.1 Sprint 1`.

- [ ] **Step 4: Final commit**

```bash
git add LAPTOP-CHANGES.md TODO.md
git commit -m "docs(lowp-fft): mark Sprint 1 complete in TODO and LAPTOP-CHANGES"
```

---

## Self-Review Checklist

1. **Spec coverage**: Does each requirement have a task?
   - [x] PyTorch C++ extension wrapping cuFFT FP16 → Task 2, 3
   - [x] `lowp_fft.fft(x)` callable from Python → Task 3
   - [x] CPU fallback → Task 2 (fft_fp16_forward_cpu)
   - [x] 🔴 No binaries committed → Task 6 (.gitignore), Task 7 (verify)
   - [x] 🟡 Benchmark data → Not applicable yet (Sprint 3 is for benchmarks)
   - [x] 🟡 Minimal reproducible test → Task 4 (tests)
   - [x] 🔴 VRAM check → FFT sizes are small (N=256-1024), VRAM << 10GB

2. **Placeholder scan**: Are there any "TBD" or "implement later"?
   - Backward pass: explicitly `NotImplementedError` with message "Sprint 2" — this is intentional, not a placeholder.
   - All steps have concrete code or commands.

3. **Type consistency**:
   - Input: `torch.complex32` (underlying `c10::Half` interleaved)
   - Output: `torch.complex32`
   - C++ function signature: `torch::Tensor → torch::Tensor`
   - Python function signature: `Tensor → Tensor`
   - Consistent throughout all tasks.

