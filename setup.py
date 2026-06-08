import os
import sys
import logging

# Ensure project root is on sys.path so _cuda_detect is importable
# during pip build isolation (pip copies setup.py to a temp dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _ensure_cuda_home():
    """Set CUDA_HOME env var, with fallbacks for pip build isolation.

    Pip build isolation may not inherit CUDA_PATH / ProgramFiles env vars
    on Windows, so we search known paths directly as well.
    """
    # 1. Already set
    if os.environ.get("CUDA_HOME"):
        return

    # 2. Try _cuda_detect (covers CUDA_PATH, nvcc, platform defaults)
    try:
        from _cuda_detect import find_cuda_home
        os.environ["CUDA_HOME"] = find_cuda_home()
        return
    except (ImportError, OSError):
        pass

    # 3. Hard fallback: search common Windows CUDA paths directly
    #    (does not depend on env vars, works even in pip isolation)
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA",
        ]
        for base in candidates:
            if os.path.isdir(base):
                try:
                    versions = sorted(
                        [d for d in os.listdir(base) if d.startswith("v")],
                        reverse=True,
                    )
                    for v in versions:
                        path = os.path.join(base, v)
                        if os.path.isdir(os.path.join(path, "include")):
                            os.environ["CUDA_HOME"] = path
                            return
                except OSError:
                    continue

    # 4. Linux fallback
    for path in ["/usr/local/cuda", "/opt/cuda"]:
        if os.path.isdir(os.path.join(path, "include")):
            os.environ["CUDA_HOME"] = path
            return


_ensure_cuda_home()

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import torch


def _detect_gpu_arch():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        logging.info(f"Detected GPU arch: {arch} (device 0)")
        return arch
    else:
        logging.warning(
            "CUDA not available — using fallback arch sm_86. "
            "Install CUDA toolkit or set CUDA_VISIBLE_DEVICES."
        )
        return "sm_86"


_GPU_ARCH = _detect_gpu_arch()

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

setup(
    name="lowp_fft",
    version="0.1.0",
    description="Low-precision FFT for PyTorch — FP16/BF16/FP8 wrappers",
    packages=find_packages(exclude=["tests", "tests.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=["torch>=2.0"],
    zip_safe=False,
)
