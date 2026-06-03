import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Point to CUDA Toolkit 13.3 (nvcc and headers)
os.environ.setdefault("CUDA_HOME", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")

ext_modules = [
    CUDAExtension(
        name="lowp_fft._cufft_ext",
        sources=["lowp_fft/csrc/cufft_fp16.cu"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-arch=sm_120",
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
