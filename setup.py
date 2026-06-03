from setuptools import setup, find_packages

setup(
    name="lowp_fft",
    version="0.1.0",
    description="Low-precision FFT for PyTorch — FP16/BF16/FP8 wrappers",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["torch>=2.0"],
    zip_safe=False,
)
