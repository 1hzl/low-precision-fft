# Low Precision FFT for PyTorch · 低精度 FFT

> 低精度 FFT 研究项目：面向 LLM 微调的 FP16 / BF16 / FP8 块浮点 FFT 实现  
> Low-precision FFT for LLM fine-tuning with circulant adapters

**GPU**: NVIDIA RTX 5070 Ti (SM_120, 12 GB VRAM, CUDA 13.3)  
**论文参考**: [arXiv 2505.00582](https://arxiv.org/abs/2505.00582) — Block Circulant Adapter for LLMs

---

## 支持精度 · Supported Precisions

| 精度 Precision | 后端 Backend       | 字节/元素 | FFT  | IFFT | 典型 SQNR (N=1024) |
|---------------|--------------------|-----------|------|------|--------------------|
| FP32          | torch.fft (cuFFT)  | 8         | ✅   | ✅   | 参考基准            |
| FP16          | cuFFT Xt           | 4         | ✅   | ✅   | 56–61 dB           |
| BF16          | cuFFT Xt           | 4         | ✅   | ✅   | 53.1 dB            |
| BFP FP8       | 自研 CUDA / NumPy   | 2         | ✅   | ✅   | 20–22 dB           |

- **FP16 / BF16**：通过 PyTorch C++ 扩展调用 cuFFT Xt 原生半精度接口，支持自动微分
- **BFP FP8**：自研块浮点 Radix-2 DIT FFT，每级蝶形共享指数，蝶形运算在 float32 精度完成，输出量化回 FP8 尾数。同时提供纯 Python/NumPy 原型，无需 GPU 即可跑

*SQNR 以 FP64 参考 FFT 为基准，在 RTX 5070 Ti 上测得*  

---

## 安装 · Installation

```bash
pip install -e .
```

**环境要求**：

| 组件 | 版本 / 说明 |
|------|------------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0（CUDA 版本） |
| CUDA Toolkit | 13.3 |
| C++ 编译器 | Windows: Visual Studio Build Tools（勾选「C++ 桌面开发」） |
| GPU | SM_80+（FP16/BF16 cuFFT Xt 需要。无 GPU 可跑 BFP CPU 原型） |

**编译 BFP CUDA kernel（可选，仅供性能基准测试）**：

```bash
build_bfp.bat   # 需 VS Developer Command Prompt + CUDA 13.3
```

---

## 用法 · Usage

```python
import torch
import lowp_fft

x = torch.randn(256, dtype=torch.complex64, device="cuda")

# FP32（默认）— 走 torch.fft
y_fp32 = lowp_fft.fft(x)

# FP16 — cuFFT Xt 原生半精度
y_fp16 = lowp_fft.fft(x, precision="fp16")

# BF16 — cuFFT Xt bfloat16
y_bf16 = lowp_fft.fft(x, precision="bf16")

# BFP FP8 — 自研块浮点（CPU 原型，无需 GPU）
y_fp8 = lowp_fft.fft(x, precision="fp8")

# 逆变换同理
x_hat = lowp_fft.ifft(y_fp16, precision="fp16")
```

---

## 独立验证 · Reproducibility

想验证本项目的结果？→ 打开 [`docs/VERIFY.md`](docs/VERIFY.md)，按步骤跑。

- CPU 路径（1 分钟，无 GPU）：22 个 BFP 算法测试
- GPU 路径（5 分钟，需 CUDA）：全量 108 个测试

---

## 项目结构 · Project Structure

```
low-precision-fft/
├── lowp_fft/                    # Python 包
│   ├── __init__.py              # 公开 API：fft(), ifft()
│   ├── _autograd.py             # FP16/BF16 自动微分
│   ├── bfp_fft.py               # BFP FP8 Python/NumPy 原型
│   └── csrc/
│       └── cufft_fp16.cu        # cuFFT Xt PyTorch C++ 扩展
├── src/cuda/                    # 独立 CUDA kernel
│   ├── bfp_fft.cu               # BFP FP8 Radix-2 DIT FFT + IFFT
│   └── bfp_fft.h                # 公开 C API
├── tests/                       # 测试 & 基准
│   ├── test_bfp_fft.py          # BFP 单元/边界测试
│   ├── test_autograd.py         # FP16/BF16 自动微分测试
│   ├── bench_bfp_throughput.py  # 吞吐量对比
│   └── bench_bfp_memory.py      # 内存带宽基准
├── data/                        # 消融实验数据 (CSV)
├── docs/                        # 设计文档、误差分析、验证指南
├── setup.py                     # 包安装
├── build_bfp.bat                # BFP CUDA kernel 编译
└── TODO.md                      # 任务跟踪
```

---

## 消融实验关键发现 · Ablation Findings

| 实验 | 变量 | 结论 |
|------|------|------|
| 尾数位宽 | E4M2 / E4M3 / E4M4 / E5M3 | 每 bit ~6 dB，**E4M3 为甜点** |
| 指数共享粒度 | per-stage / group-4 / group-8 | **per-stage 已最优**，更细粒度无收益 |

详见 `data/ablation-mantissa-bits.md` 和 `data/ablation-group-size.md`

---

## 参考文献 · References

- [Block Circulant Adapter for LLMs](https://arxiv.org/abs/2505.00582) (arXiv 2505.00582)
- PyTorch `CuFFTPlanCache.h:308-311` — `kHalf → CUDA_C_16F` 类型映射
