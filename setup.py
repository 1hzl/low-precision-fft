import os
import logging
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Cross-platform CUDA_HOME detection
from _cuda_detect import find_cuda_home
try:
    os.environ.setdefault("CUDA_HOME", find_cuda_home())
except OSError:
    pass  # let torch.cpp_extension report the error with its own message

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

ext_modules = [
    CUDAExtension(
        name="lowp_fft._cufft_ext",
        sources=["lowp_fft/csrc/cufft_fp16.cu"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                f"-arch={_GPU_ARCH}",
                "--expt-relaxed-constexpr",
            ],
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
