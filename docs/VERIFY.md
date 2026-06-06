# 独立验证指南

低精度 FFT 项目的独立可复现验证。5-10 分钟跑通。

---

## 前置条件

- Python 3.10+
- (可选) NVIDIA GPU + CUDA Toolkit 12.x

---

## 快速验证 — CPU 路径（任何人可跑，1 分钟）

无需 GPU，仅需 NumPy。验证 BFP FP8 算法正确性。

```bash
git clone <repo-url>
cd low-precision-fft
pip install numpy
python -m pytest tests/test_bfp_fft.py -v
```

**预期输出**：

```
tests/test_bfp_fft.py::test_bfp_forward_shape      PASSED
tests/test_bfp_fft.py::test_bfp_forward_finite      PASSED
tests/test_bfp_fft.py::test_bfp_inverse              PASSED
tests/test_bfp_fft.py::test_bfp_roundtrip            PASSED
tests/test_bfp_fft.py::test_bfp_dc_input             PASSED
tests/test_bfp_fft.py::test_bfp_extreme_values       PASSED
tests/test_bfp_fft.py::test_bfp_all_zeros            PASSED
tests/test_bfp_fft.py::test_bfp_n2                   PASSED
tests/test_bfp_fft.py::test_bfp_n4                   PASSED
tests/test_bfp_fft.py::test_bfp_mantissa_bits        PASSED
tests/test_bfp_fft.py::test_bfp_group_size           PASSED
...

======================== 22 passed ========================
```

---

## 完整验证 — GPU 路径（需 NVIDIA GPU，5 分钟）

验证 FP16 cuFFT 封装 + BFP CUDA kernel + 自动微分。

```bash
git clone <repo-url>
cd low-precision-fft
pip install -e .
python -m pytest tests/ -v
```

**预期输出**：

```
======================== 108 passed ========================
```

> GPU 架构会自动检测（`torch.cuda.get_device_capability()`），无需手动修改。

---

## 复现消融实验关键数字（可选，2 分钟）

验证论文核心数据：

```bash
# 尾数位宽消融（Task 2a）
python tests/bench_bfp_ablation_mantissa.py

# 指数共享粒度消融（Task 2b）
python tests/bench_bfp_ablation_group_size.py
```

**关键数字**（N=1024, E4M3, 允许偏差 ±0.3 dB）：

| Benchmark | 配置 | SQNR |
|-----------|------|------|
| 尾数消融 | E4M3 uniform | ~21.2 dB |
| 尾数消融 | E4M3 normal | ~21.5 dB |
| 粒度消融 | per-stage uniform | ~21.2 dB |

---

## 如果遇到问题

| 问题 | 原因 | 修法 |
|------|------|------|
| `nvcc not found` | CUDA Toolkit 未安装 | `apt install nvidia-cuda-toolkit` |
| `torch not found` | PyTorch 未安装 | `pip install torch` |
| 编译失败 | 驱动版本过旧 | 更新 NVIDIA Driver ≥ 570 |
| SQNR 偏差 >0.5 dB | 不同 OS/NumPy 版本的浮点累计误差 | 正常，记录即可 |

---

## 验证记录模板

```
=== 低精度 FFT 独立验证记录 ===

姓名：________
日期：2026-__-__
硬件：________（GPU 型号，或 "CPU only"）
OS：________
Python：3.__.__

□ CPU 路径：__/22 passed
□ GPU 路径：__/108 passed（如有 GPU）
□ 尾数消融 E4M3 uniform SQNR：____ dB
□ 粒度消融 per-stage SQNR：____ dB

备注：
```
