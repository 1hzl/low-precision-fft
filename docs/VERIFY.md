# 独立验证指南

低精度 FFT 项目的独立可复现验证。5-10 分钟跑通。

---

## 前置条件

- Python 3.10+
- (可选) NVIDIA GPU + CUDA Toolkit 12.x

> 获取代码：联系项目负责人获取仓库地址，然后 `git clone <url> && cd low-precision-fft`

---

## 快速验证 — CPU 路径（任何人可跑，1 分钟）

无需 GPU，仅需 NumPy + pytest。验证 BFP FP8 算法正确性。

```bash
pip install numpy pytest
python -m pytest tests/test_bfp_fft.py -v
```

**预期输出（节选）**：

```
test_zero ............................... PASSED
test_exact_representable ................ PASSED
test_round_to_nearest ................... PASSED
test_forward_shape ...................... PASSED
test_against_fp32_reference ............. PASSED
test_exponents_monotonic ................ PASSED
test_power_of_two_only .................. PASSED
test_dc_only_signal ..................... PASSED
test_extreme_dynamic_range .............. PASSED
test_all_zeros_input .................... PASSED
test_identity_small_n[8] ................ PASSED
test_identity_small_n[16] ............... PASSED
test_minimum_legal_input[2] ............. PASSED
...

======================== 22 passed ========================
```

---

## 完整验证 — GPU 路径（需 NVIDIA GPU，5 分钟）

验证 FP16 cuFFT 封装 + BFP CUDA kernel + 自动微分。

```bash
# 先装 PyTorch（从 pytorch.org 按你的 CUDA 版本选）
pip install torch
# 再编译安装项目
pip install -e .
# 跑全量测试
python -m pytest tests/ -v
```

**预期输出**：

```
tests/test_bfp_fft.py ................... 22 passed
tests/test_bfp_cuda.py ................... 6 passed
tests/test_autograd.py .................. 61 passed
tests/test_bf16.py ...................... 19 passed

======================== 108 passed ========================
```

> GPU 架构会自动检测（`torch.cuda.get_device_capability()`），任何 NVIDIA 卡均可编译，无需手动修改。

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

## 常见问题

| 问题 | 原因 | 修法 |
|------|------|------|
| `ModuleNotFoundError: numpy` | NumPy 未安装 | `pip install numpy` |
| `nvcc not found` | CUDA Toolkit 未安装 | `apt install nvidia-cuda-toolkit` |
| `torch not found` | PyTorch 未安装 | 从 [pytorch.org](https://pytorch.org) 安装 |
| 编译报错（GPU 路径） | 驱动/CUDA 版本不兼容 | 确认 `nvidia-smi` 显示 CUDA ≥ 12.x |
| SQNR 偏差 >0.5 dB | 不同 OS/NumPy/PyTorch 版本的浮点累计误差 | 正常现象，记录即可 |

---

## 验证记录

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
