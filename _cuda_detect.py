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
    """Search Program Files/NVIDIA GPU Computing Toolkit/CUDA/ for newest version."""
    program_files = os.environ.get("ProgramFiles", "C:/Program Files")
    base = os.path.join(
        program_files, "NVIDIA GPU Computing Toolkit", "CUDA",
    )
    return _pick_newest_version(base)


def _find_linux_default():
    """Try /usr/local/cuda symlink, then /usr/local/cuda-* versions."""
    if _is_valid_cuda_home("/usr/local/cuda"):
        return "/usr/local/cuda"
    return _pick_newest_version("/usr/local")


def _pick_newest_version(parent):
    """Given a parent directory, return the path matching 'cuda*' or 'v*'
    with the highest version suffix (e.g. cuda-13.3, v13.3)."""
    if not os.path.isdir(parent):
        return None
    try:
        entries = os.listdir(parent)
    except OSError:
        return None
    cuda_dirs = []
    for name in entries:
        full = os.path.join(parent, name)
        if not os.path.isdir(full):
            continue
        if name.startswith("cuda") or name.startswith("v"):
            if _is_valid_cuda_home(full):
                cuda_dirs.append(full)
    cuda_dirs.sort(reverse=True)
    return cuda_dirs[0] if cuda_dirs else None
