# Build Compatibility Fix — pyproject.toml + CUDA_HOME 跨平台

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three build blockers so `pip install -e .` works on both Windows and Linux without manual CUDA_HOME setup.

**Architecture:** Extract CUDA_HOME detection into a shared `_cuda_detect.py` at repo root (importable by both `setup.py` and `build_ext.py`). Add `pyproject.toml` so pip's build isolation installs torch before running setup.py.

**Tech Stack:** Python setuptools, CUDA Toolkit, pip PEP 517 build isolation

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | **Create** | Declare build-system requirements (torch) for PEP 517 build isolation |
| `_cuda_detect.py` | **Create** | Shared cross-platform CUDA_HOME auto-detection, importable by both build scripts |
| `setup.py` | **Modify** | Replace hardcoded Windows path with `_cuda_detect.find_cuda_home()` |
| `build_ext.py` | **Modify** | Same — replace hardcoded Windows path with shared detection |

---

### Task 1: Create `_cuda_detect.py` — shared CUDA_HOME detection

**Files:**
- Create: `_cuda_detect.py`

- [ ] **Step 1: Write `_cuda_detect.py`**

```python
"""Cross-platform CUDA_HOME auto-detection.

Used by setup.py and build_ext.py — must be importable without torch
(since it runs before torch is available in the build environment).
"""

import os
import sys
import logging

_log = logging.getLogger(__name__)


def find_cuda_home():
    """Detect CUDA toolkit installation path.

    Resolution order:
    1. CUDA_HOME env var (standard)
    2. CUDA_PATH env var (Windows convention, e.g. CUDA 13.3 installer)
    3. nvcc on PATH → derive CUDA_HOME as parent of bin/
    4. Platform-specific default install paths, newest version first
    5. Raise OSError with actionable message

    Returns:
        str: Path to CUDA toolkit root (e.g. /usr/local/cuda).
    """
    # 1. Explicit env vars
    for var in ("CUDA_HOME", "CUDA_PATH"):
        path = os.environ.get(var, "")
        if path and _is_valid_cuda_home(path):
            _log.info("CUDA_HOME=%s (from %s)", path, var)
            return path

    # 2. nvcc on PATH — derive from <cuda>/bin/nvcc
    nvcc_bin = _which("nvcc")
    if nvcc_bin:
        cuda_home = os.path.dirname(os.path.dirname(nvcc_bin))
        if _is_valid_cuda_home(cuda_home):
            _log.info("CUDA_HOME=%s (derived from nvcc on PATH)", cuda_home)
            return cuda_home

    # 3. Platform-specific default install paths
    if sys.platform == "win32":
        path = _find_windows_default()
    else:
        path = _find_linux_default()
    if path:
        _log.info("CUDA_HOME=%s (platform default)", path)
        return path

    # 4. Nothing found
    raise OSError(
        "CUDA toolkit not found. Set CUDA_HOME environment variable "
        "to your CUDA toolkit installation path, or install CUDA toolkit.\n"
        "Typical paths:\n"
        '  Windows: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3\n'
        "  Linux:   /usr/local/cuda"
    )


def _is_valid_cuda_home(path):
    """Check whether path looks like a CUDA toolkit root."""
    if not path or not os.path.isdir(path):
        return False
    include_dir = os.path.join(path, "include")
    bin_dir = os.path.join(path, "bin")
    return os.path.isdir(include_dir) and os.path.isdir(bin_dir)


def _which(exe_name):
    """Cross-platform `which` — returns path to executable or None."""
    exe = exe_name + (".exe" if sys.platform == "win32" else "")
    for dirent in os.environ.get("PATH", "").split(os.pathsep):
        full = os.path.join(dirent, exe)
        if os.path.isfile(full):
            return full
    return None


def _find_windows_default():
    """Search C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/ for newest version."""
    base = os.path.join(
        os.environ.get("ProgramFiles", "C:/Program Files"),
        "NVIDIA GPU Computing Toolkit", "CUDA",
    )
    return _pick_newest_version(base)


def _find_linux_default():
    """Try /usr/local/cuda symlink, then /usr/local/cuda-* versions."""
    # Symlink installed by most package managers
    if _is_valid_cuda_home("/usr/local/cuda"):
        return "/usr/local/cuda"
    # Versioned directories
    return _pick_newest_version("/usr/local")


def _pick_newest_version(parent):
    """Given a parent directory, return the path matching 'cuda*' with the
    highest version suffix (e.g. cuda-13.3, cuda/v13.3)."""
    if not os.path.isdir(parent):
        return None
    try:
        entries = os.listdir(parent)
    except OSError:
        return None
    # Filter entries starting with 'cuda' or 'v' (Windows uses 'v13.3')
    cuda_dirs = []
    for name in entries:
        full = os.path.join(parent, name)
        if os.path.isdir(full) and (name.startswith("cuda") or name.startswith("v")):
            if _is_valid_cuda_home(full):
                cuda_dirs.append(full)
    # Sort descending by name (higher version → newer)
    cuda_dirs.sort(reverse=True)
    return cuda_dirs[0] if cuda_dirs else None
```

- [ ] **Step 2: Verify `_cuda_detect.py` runs standalone**

Run: `python _cuda_detect.py` (should do nothing — no `if __name__` block, just verify no import errors)

---

### Task 2: Create `pyproject.toml`

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=64", "wheel", "torch>=2.0"]
build-backend = "setuptools.build_meta"

[project]
name = "lowp_fft"
version = "0.1.0"
description = "Low-precision FFT for PyTorch — FP16/BF16/FP8 wrappers"
requires-python = ">=3.10"
dependencies = ["torch>=2.0"]
```

- [ ] **Step 2: Verify TOML syntax**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` (Python 3.11+)
or: `python -c "import tomli; tomli.load(open('pyproject.toml','rb'))"` (if tomli installed)

---

### Task 3: Fix `setup.py` — replace hardcoded CUDA_HOME

**Files:**
- Modify: `setup.py:7-9`

- [ ] **Step 1: Replace lines 6-9 of setup.py**

Change:
```python
# Point to CUDA Toolkit 13.3 (nvcc and headers)
# Prefer CUDA_HOME from environment, with Windows default as fallback
_default_cuda = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3"
os.environ.setdefault("CUDA_HOME", os.environ.get("CUDA_HOME", _default_cuda))
```

To:
```python
# Cross-platform CUDA_HOME detection
from _cuda_detect import find_cuda_home
try:
    os.environ.setdefault("CUDA_HOME", find_cuda_home())
except OSError:
    # Let torch.cpp_extension report the error with its own message
    pass
```

- [ ] **Step 2: Verify setup.py still imports**

Run: `python -c "import ast; ast.parse(open('setup.py').read()); print('Syntax OK')"`

---

### Task 4: Fix `build_ext.py` — replace hardcoded CUDA_HOME (same pattern)

**Files:**
- Modify: `build_ext.py:12-13`

- [ ] **Step 1: Replace lines 12-13 of build_ext.py**

Change:
```python
_default_cuda = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3"
os.environ.setdefault("CUDA_HOME", os.environ.get("CUDA_HOME", _default_cuda))
```

To:
```python
from _cuda_detect import find_cuda_home
try:
    os.environ.setdefault("CUDA_HOME", find_cuda_home())
except OSError:
    pass
```

- [ ] **Step 2: Verify build_ext.py syntax**

Run: `python -c "import ast; ast.parse(open('build_ext.py').read()); print('Syntax OK')"`

---

### Task 5: End-to-end verification

- [ ] **Step 1: Verify pyproject.toml takes effect**

Run: `pip install --dry-run -e . 2>&1 | head -20`
Expected: No `ModuleNotFoundError: No module named 'torch'`; installation plan is computed correctly.

- [ ] **Step 2: Run existing test suite**

Run: `pytest tests/ -x -q 2>&1 | tail -5`
Expected: Same pass count as before this change.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml _cuda_detect.py setup.py build_ext.py
git commit -m "fix(build): cross-platform CUDA_HOME detection + pyproject.toml build isolation"
```
