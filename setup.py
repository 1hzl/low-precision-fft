import os
import sys
import logging

# Ensure project root is on sys.path so _cuda_detect is importable
# during pip build isolation (pip copies setup.py to a temp dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)


def _detect_cuda_home():
    """Try hard to find CUDA. Returns path or None."""
    # 1. Already set
    val = os.environ.get("CUDA_HOME")
    if val:
        return val

    # 2. CUDA_PATH env (Windows convention)
    val = os.environ.get("CUDA_PATH")
    if val and os.path.isdir(os.path.join(val, "include")):
        return val

    # 3. nvcc on PATH
    import shutil
    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_home = os.path.dirname(os.path.dirname(nvcc))
        if os.path.isdir(os.path.join(cuda_home, "include")):
            return cuda_home

    # 4. Known install paths (no env vars needed)
    if sys.platform == "win32":
        for root in [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA",
        ]:
            if not os.path.isdir(root):
                continue
            try:
                vers = sorted(
                    [d for d in os.listdir(root) if d.lower().startswith("v")],
                    reverse=True,
                )
            except OSError:
                continue
            for v in vers:
                path = os.path.join(root, v)
                if os.path.isdir(os.path.join(path, "include")):
                    return path
    else:
        for path in ["/usr/local/cuda", "/opt/cuda"]:
            if os.path.isdir(os.path.join(path, "include")):
                return path

    return None


_CUDA_HOME = _detect_cuda_home()
if _CUDA_HOME:
    os.environ["CUDA_HOME"] = _CUDA_HOME  # force-set, setdefault won't overwrite empty string
    logging.info("CUDA_HOME=%s", _CUDA_HOME)
else:
    logging.warning(
        "CUDA toolkit not found. CUDA extension will NOT be built.\n"
        "  Set CUDA_HOME or CUDA_PATH, or install CUDA toolkit.\n"
        "  The pure Python API (fft/ifft fallback, BFP prototype) will still work."
    )

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch


def _detect_gpu_arch():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        logging.info("Detected GPU arch: %s (device 0)", arch)
        return arch
    elif _CUDA_HOME:
        logging.warning("CUDA toolkit found but GPU not available — using sm_86")
        return "sm_86"
    else:
        return None


_GPU_ARCH = _detect_gpu_arch()

# ── Extension modules (conditional on CUDA) ──
ext_modules = []
cmdclass = {}

if _CUDA_HOME and _GPU_ARCH:
    _nvcc_args = [
        "-O3",
        "-std=c++17",
        f"-arch={_GPU_ARCH}",
        "--expt-relaxed-constexpr",
    ]
    if sys.platform == "win32":
        _nvcc_args.extend(["-Xcompiler", "/Zc:preprocessor"])

    ext_modules = [
        CUDAExtension(
            name="lowp_fft._cufft_ext",
            sources=["lowp_fft/csrc/cufft_fp16.cu"],
            libraries=["cufft"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": _nvcc_args,
            },
        ),
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="lowp_fft",
    version="0.1.0",
    description="Low-precision FFT for PyTorch — FP16/BF16/FP8 wrappers",
    packages=find_packages(exclude=["tests", "tests.*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
    install_requires=["torch>=2.0"],
    zip_safe=False,
)
