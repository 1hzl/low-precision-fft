Low Precision FFT for Pytorch

We are looking for implementing low precision (FP16, FP8) FFT implementation on GPU (with CUDA) and CPU (with intel/arm/risc-v SIMD intrinsics) to support applications like finetuning LLMs https://arxiv.org/pdf/2505.00582 . The final goal is to contribute code to Pytorch community, where they are lack of such support.

Reference

https://www.leetgpu.com/challenges (cuda practice)

https://www.youtube.com/playlist?list=PLzn6LN6WhlN06hIOA_ge6SrgdeSiuf9Tb

https://arm-software.github.io/acle/main/acle.html

https://openreview.net/forum?id=oWnAlRn3X1