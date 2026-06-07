# Build Compatibility Fix (Suppl.) — MSVC CCCL + libraries in setup.py

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two remaining issues in `setup.py` so `pip install -e .` works on Windows with CUDA 13.3 + MSVC: (1) missing `-Xcompiler /Zc:preprocessor` for CCCL compatibility, (2) missing `libraries=["cufft"]`.

**Architecture:** setup.py's nvcc args conditionally append the MSVC preprocessor flag on Windows. The `libraries=["cufft"]` is added to match build_ext.py.

**Tech Stack:** Python setuptools, CUDA Toolkit 13.3, MSVC

**Prerequisite:** Plan `2026-06-07-build-compat-cuda-home.md` Tasks 1-4 already completed (commit `700a640`).

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `setup.py` | **Modify** | Add `import sys`; add Windows-conditional `/Zc:preprocessor` to nvcc args; add `libraries=["cufft"]` |

---

### Task 1: Fix setup.py — MSVC CCCL flag + cuFFT library

**Files:**
- Modify: `setup.py`

- [ ] **Step 1: Add `import sys` at top of setup.py**

In the existing imports block (after `import logging`), add `import sys`:

```python
import os
import sys
import logging
```

- [ ] **Step 2: Add platform-conditional nvcc args with `/Zc:preprocessor`**

Replace the static nvcc `extra_compile_args` dict with a dynamic one:

Change from:
```python
extra_compile_args={
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        f"-arch={_GPU_ARCH}",
        "--expt-relaxed-constexpr",
    ],
},
```

To:
```python
_nvcc_args = [
    "-O3",
    "-std=c++17",
    f"-arch={_GPU_ARCH}",
    "--expt-relaxed-constexpr",
]
if sys.platform == "win32":
    _nvcc_args.extend(["-Xcompiler", "/Zc:preprocessor"])

# ... then in CUDAExtension:
extra_compile_args={
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": _nvcc_args,
},
```

- [ ] **Step 3: Add `libraries=["cufft"]` to CUDAExtension**

Add `libraries=["cufft"]` to the CUDAExtension call (matching build_ext.py):

```python
CUDAExtension(
    name="lowp_fft._cufft_ext",
    sources=["lowp_fft/csrc/cufft_fp16.cu"],
    libraries=["cufft"],
    extra_compile_args={...},
),
```

- [ ] **Step 4: Verify setup.py syntax**

Run: `python -c "import ast; ast.parse(open('setup.py').read()); print('Syntax OK')"`

---

### Task 2: End-to-end verification

- [ ] **Step 1: Verify import chain works (no CUDA required)**

Run: `python -c "import sys; sys.path.insert(0,'.'); from _cuda_detect import find_cuda_home; print('_cuda_detect OK')"`

- [ ] **Step 2: Run existing test suite**

Run: `pytest tests/ -x -q 2>&1 | tail -5`
Expected: Same pass count as before this change.

- [ ] **Step 3: Commit**

```bash
git add setup.py docs/superpowers/plans/2026-06-07-build-compat-msvc-cccl.md
git commit -m "fix(build): add MSVC /Zc:preprocessor + libraries=cufft to setup.py for CUDA 13.3 CCCL"
```
