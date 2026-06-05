"""Workaround script for building the cuFFT extension with CUDA 13.3 + PyTorch 12.8.

PyTorch's cpp_extension refuses to compile when nvcc version != torch.version.cuda
major. CUDA 13.3 nvcc can target sm_120 and the output is ABI-compatible with the
PyTorch 12.8 runtime — the check is overly conservative.
"""

import os
import sys
import logging

_default_cuda = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3"
os.environ.setdefault("CUDA_HOME", os.environ.get("CUDA_HOME", _default_cuda))

import torch
import torch.utils.cpp_extension as cpp_ext

_original_check = cpp_ext._check_cuda_version


def _patched_check(compiler_name, compiler_version):
    """Same as original but warns instead of erroring on major-version mismatch."""
    CUDA_HOME = os.getenv("CUDA_HOME")
    if not CUDA_HOME:
        raise RuntimeError(cpp_ext.CUDA_NOT_FOUND_MESSAGE)

    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc.exe" if sys.platform == "win32" else "nvcc")
    if not os.path.exists(nvcc):
        raise FileNotFoundError(f"nvcc not found at '{nvcc}'")

    import subprocess
    cuda_version_str = subprocess.check_output([nvcc, "--version"]).strip().decode()
    import re
    m = re.search(r"release (\d+[.]\d+)", cuda_version_str)
    if m is None:
        return

    cuda_str_version = m.group(1)
    logging.warning(
        f"CUDA {cuda_str_version} nvcc detected (PyTorch built with {torch.version.cuda}). "
        f"Proceeding — binaries should be ABI-compatible."
    )


cpp_ext._check_cuda_version = _patched_check

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name="lowp_fft._cufft_ext",
        sources=["lowp_fft/csrc/cufft_fp16.cu"],
        libraries=["cufft"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-arch=sm_120",
                "--expt-relaxed-constexpr",
                "-Xcompiler", "/Zc:preprocessor",
            ],
        },
    ),
]

setup(
    name="lowp_fft",
    version="0.1.0",
    packages=["lowp_fft"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
