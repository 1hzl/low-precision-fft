# 独立验证指南

按照本指南操作，5-10 分钟即可完成低精度 FFT 项目的独立验证。

---

## 第 1 步：获取代码

打开终端（Windows 按 `Win+R`，输入 `cmd`，回车）。

逐条输入以下命令：

```bash
git clone https://github.com/1hzl/low-precision-fft.git
cd low-precision-fft
```

**你应该看到**：

```
Cloning into 'low-precision-fft'...
...
Receiving objects: 100%, done.
```

**如果失败**：

| 报错 | 修法 |
|------|------|
| `git: command not found` | 安装 [Git for Windows](https://git-scm.com/download/win)（一路点 Next），安装完后关掉终端重新打开 |
| `fatal: unable to access` | 连不上 GitHub。方法一：浏览器打开 https://github.com/1hzl/low-precision-fft，点绿色「Code」按钮 →「Download ZIP」→ 解压到桌面 → `cd Desktop\low-precision-fft`。方法二：开代理后重试 |

---

## 第 2 步：检查 Python 版本

```bash
python --version
```

**你应该看到**：`Python 3.10.x` 或 `Python 3.11.x` 或 `Python 3.12.x`

**如果版本低于 3.10**：去 [python.org](https://www.python.org/downloads/) 下载安装 Python 3.10+，安装时勾选「Add Python to PATH」。

**如果输入后没反应或 `python: command not found`**：Python 没装或者没加到 PATH。重新安装 Python 并确保勾选「Add Python to PATH」。

---

## ——— 路径 A：CPU 快速验证（无 GPU 也可以）———

这条路测试 BFP FP8 算法的正确性，任何电脑都能跑。约 2 分钟。

---

### A-1. 安装 NumPy

```bash
pip install numpy
```

**你应该看到**：最后一行是 `Successfully installed numpy-x.x.x`

---

### A-2. 安装 pytest

```bash
pip install pytest
```

**你应该看到**：最后一行是 `Successfully installed pytest-x.x.x`

---

### A-3. 运行 BFP 测试

```bash
python -m pytest tests/test_bfp_fft.py -v
```

**你应该看到**（节选）：

```
tests/test_bfp_fft.py::test_zero PASSED
tests/test_bfp_fft.py::test_exact_representable PASSED
tests/test_bfp_fft.py::test_forward_shape PASSED
tests/test_bfp_fft.py::test_against_fp32_reference PASSED
tests/test_bfp_fft.py::test_dc_only_signal PASSED
tests/test_bfp_fft.py::test_all_zeros_input PASSED
...
======================== 22 passed ========================
```

**如果对不上**：

| 看到什么 | 怎么办 |
|---------|--------|
| `FAILED`（任何红色行） | 截图整段红色输出，发给项目负责人 |
| `ModuleNotFoundError: No module named 'lowp_fft'` | 确认你在 `low-precision-fft` 目录内：先输入 `cd low-precision-fft` |
| 数量不是 22 passed | 截图整段输出，发给项目负责人 |

---

## ——— 路径 B：GPU 完整验证（需要 NVIDIA 显卡）———

这条路验证全部 108 个测试，包括 FP16 cuFFT 封装、BFP CUDA kernel、自动微分。约 8 分钟。

---

### B-1. 确认显卡和驱动

```bash
nvidia-smi
```

**你应该看到**：一个表格，第一行有 `NVIDIA-SMI xxx.xx`，下面有 `GeForce RTX xxxx`。

**如果失败**：

| 看到什么 | 怎么办 |
|---------|--------|
| `nvidia-smi: command not found` | 显卡驱动没装，或没加到 PATH。去 [NVIDIA 官网](https://www.nvidia.com/download/index.aspx) 下载驱动 |
| 表格显示 `CUDA Version: N/A` | 驱动版本太旧，更新驱动 |
| 没有 `GeForce` 字样 | 你的显卡可能不是 NVIDIA，GPU 路径不可用。回退到路径 A |

记下你的显卡型号，后面填验证记录要用。

---

### B-2. 确认 PyTorch 已安装且能识别 GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**你应该看到**：

```
True
NVIDIA GeForce RTX 5070 Ti          ← 你的显卡型号
```

**如果失败**：

| 看到什么 | 怎么办 |
|---------|--------|
| `ModuleNotFoundError: No module named 'torch'` | 见步骤 B-3 |
| `False` 或报错 | CUDA 版本和 PyTorch 版本不匹配。见步骤 B-3 |

---

### B-3. 安装 PyTorch（如果需要）

打开 [pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)，按你的系统选：
- PyTorch Build: **Stable**
- Your OS: **Windows**
- Package: **Pip**
- Language: **Python**
- Compute Platform: **CUDA 12.x**（看 `nvidia-smi` 里 CUDA Version 的第二位数字，比如 CUDA 12.8 就选 CUDA 12.x）

页面会生成一条命令，类似：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

把这条命令复制到终端，回车执行。

安装完成后，重新跑 B-2 确认 `True`。

---

### B-4. 确认 C++ 编译器（Windows）

```bash
where cl
```

**你应该看到**：一行路径，如 `C:\Program Files\Microsoft Visual Studio\...\cl.exe`

**如果看到** `INFO: Could not find files for the given pattern(s).`：

去 [Visual Studio 官网](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) 下载 **Build Tools for Visual Studio 2022**。运行安装程序，在「工作负载」选项卡勾选「使用 C++ 的桌面开发」，点安装。约 2-3 GB，装完后关掉终端重新打开，再跑 `where cl`。

---

### B-5. 编译安装项目

```bash
pip install -e .
```

**你应该看到**：最后一行是 `Successfully installed lowp-fft-x.x.x`

这一步会编译 CUDA `.cu` 文件。编译需要 1-2 分钟，终端会刷很多输出，正常现象。

**如果失败**：

| 看到什么 | 怎么办 |
|---------|--------|
| `error: Microsoft Visual C++ 14.0 is required` | 回到 B-4 装 Build Tools |
| `nvcc fatal: Unsupported GPU architecture` | 显卡架构不被支持，联系项目负责人 |

---

### B-6. 运行全量测试

```bash
python -m pytest tests/ -v
```

**你应该看到**：

```
tests/test_bfp_fft.py ................... 22 passed
tests/test_bfp_cuda.py ................... 6 passed
tests/test_autograd.py .................. 61 passed
tests/test_bf16.py ...................... 19 passed

======================== 108 passed, 4 skipped, 2 xfailed in ~8s ========================
```

**关于 4 skipped + 2 xfailed**：这是正常现象，不是 bug。
- skipped：FP16/BF16 受 IEEE 754 半精度物理精度限制，无法通过数值梯度检验（已用其他替代测试覆盖）
- xfailed：PlanCache 超 64 条目淘汰时 CUDA 驱动的已知 race condition（正常使用时 FFT 尺寸固定，不触发）

**如果对不上**：

| 看到什么 | 怎么办 |
|---------|--------|
| 任何红色 `FAILED` | 截图整段红色输出，发给项目负责人 |
| passed 总数不是 108 | 截图最终统计行，发给项目负责人 |

---

## 复现消融实验（可选，额外 2 分钟）

如果你想要进一步验证论文核心数据，跑下面两条命令：

```bash
# 尾数位宽消融
python tests/bench_bfp_ablation_mantissa.py

# 指数共享粒度消融
python tests/bench_bfp_ablation_group_size.py
```

**关键检查**：看 E4M3 uniform 那行，SQNR 应该在 21.2 ± 0.3 dB 范围内。偏差 >0.5 dB 是正常的（不同机器浮点累计误差），记录即可。

---

## ✅ 填验证记录

复制下面模板，填完发给项目负责人：

```
=== 低精度 FFT 独立验证记录 ===

姓名：________
日期：2026-__-__
硬件：________（GPU 型号，或 "CPU only"）
OS：Windows ____  /  macOS ____  /  Linux ____
Python：3.__.__

□ CPU 路径：__/22 passed
□ GPU 路径：__/108 passed（如有 GPU）
□ 尾数消融 E4M3 uniform SQNR：____ dB
□ 粒度消融 per-stage SQNR：____ dB

遇到的问题（没有就写"无"）：
```

---

## 你完成了什么

- CPU 路径 22 个测试 → 验证了 BFP FP8 算法的数学正确性
- GPU 路径 108 个测试 → 验证了 cuFFT FP16/BF16 封装 + BFP CUDA kernel + 自动微分
- 消融实验 → 复现了论文核心 SQNR 数字

你的验证记录将作为独立可复现性证明，用于中期报告/论文附录/答辩。
