# 独立验证指南

按照本指南操作，5-10 分钟即可完成低精度 FFT 项目的独立验证。

> **v3.0 更新**：基于 7 台独立机器 8 位验证者的踩坑经验，新增验证前检查清单、版本兼容性矩阵、
> 验证日志正确采集方法、Linux build_bfp 说明、更精确的预期输出（区分 Volta/非Volta/Linux）。

---

## 验证前检查清单

开始前，确认下面 4 项全部通过：

```bash
# 1. Python ≥ 3.10（实测通过 3.12.3 / 3.12.9 / 3.14.4 / 3.14.5）
python --version

# 2. Git（任意版本）
git --version

# 3. setuptools（必须！缺少会导致 pip install -e . 失败）
pip install setuptools

# 4. pytest + numpy（测试所需）
pip install pytest numpy
```

**Linux 额外检查：**
```bash
g++ --version              # CUDA 编译需要 C++ 编译器
which nvcc 2>/dev/null || echo "需要安装 CUDA Toolkit"
```

没有就去装：[Python](https://www.python.org/downloads/)、[Git](https://git-scm.com/download/win)。

### 实测版本兼容性矩阵

| Python | PyTorch | CUDA Toolkit | 驱动 | OS | 实测结果 |
|--------|---------|-------------|------|----|----------|
| 3.12.3 | 2.8.0+cu128 | 12.8 | 580.x | Linux | ✅ 通过 |
| 3.12.9 | — | 13.3 | 610.47 | Win 11 | ✅ 通过 |
| 3.14.4 | 2.11.0+cu128 | 13.3 | 580.97 | Win 11 | ✅ 通过 |
| 3.14.5 | 2.8.x | 13.3.33 | 610.47 | Win 11 | ✅ 通过 |

> 如果你的版本不在表中，通常也能通过。遇到问题参考下方常见问题速查。

---

## 第 1 步：获取代码

```bash
git clone https://github.com/1hzl/low-precision-fft.git
cd low-precision-fft
```

**你应该看到**：`Cloning into 'low-precision-fft'... Receiving objects: 100%, done.`

| 报错 | 修法 |
|------|------|
| `git: command not found` | 安装 Git 后关掉终端重新打开 |
| `fatal: unable to access` | 浏览器打开 GitHub → Code → Download ZIP → 解压 → `cd` 进目录 |

---

## ——— 路径 A：无 CUDA / 快速验证（2 分钟）———

适合没有 NVIDIA 显卡、或只想快速验证 BFP 算法的场景。

### A-1. 安装

```bash
pip install -e .
```

**你应该看到**：`Successfully installed lowp_fft-0.1.0`

> 这一步会自动检测 CUDA。没有 CUDA 时 CUDA 扩展跳过编译，纯 Python 部分照常安装。

| 看到什么 | 怎么办 |
|---------|--------|
| `ModuleNotFoundError: No module named 'torch'` | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `ModuleNotFoundError: No module named 'setuptools'` | `pip install setuptools` 然后重新 `pip install -e .` |
| `Successfully installed` 但有一行 `CUDA toolkit not found` | ✅ 正常，纯 Python 模式 |
| `pip: WARNING: Cache entry deserialization failed` | ✅ 无害警告，忽略 |

### A-2. 跑测试

```bash
pip install pytest numpy -q

# BFP 原型测试（纯 Python，不需要 GPU）
python -m pytest tests/test_bfp_fft.py -v

# API fallback 测试（CPU fallback 路径，不需要 GPU）
python -m pytest tests/test_autograd.py tests/test_bf16.py -v --tb=short
```

**你应该看到**：
```
test_bfp_fft.py .... 22 passed
test_autograd.py .... 33 passed, 2 skipped, 2 xfailed
test_bf16.py .... 39 passed, 2 skipped
```

> skipped = gradcheck（受半精度物理精度限制，已用替代测试覆盖）
> xfailed = PlanCache race condition（正常使用时 FFT 尺寸固定，不触发）

| 看到什么 | 怎么办 |
|---------|--------|
| 任何 `FAILED` | 截图红色部分发给项目负责人 |
| `ModuleNotFoundError: No module named 'lowp_fft'` | 确认在 `low-precision-fft` 目录内 |

---

## ——— 路径 B：GPU 完整验证（~8 分钟）———

需要 NVIDIA 显卡 + CUDA Toolkit。全量测试包括 cuFFT FP16/BF16 扩展 + 自动微分。

> ⚠️ **如果你用的是 V100 / Volta 架构 (sm_70)，请先阅读 [V100 专用说明](#v100-volta-sm_70-专用说明)。**

---

### B-1. 确认驱动和 CUDA

**Windows:**
```bash
nvidia-smi
nvcc --version
```

**Linux:**
```bash
nvidia-smi
nvcc --version
```

**你应该看到**：
```
NVIDIA-SMI xxx.xx  ...  Driver Version: xxx  ...  CUDA Version: 1x.x
Cuda compilation tools, release 1x.x
```

> ⚠️ **nvidia-smi 里的 CUDA Version 必须 ≥ nvcc 的 release 版本号**。
> 如果 nvidia-smi 显示 CUDA 12.9 但 nvcc 是 13.3，需要更新显卡驱动。
> 经验：CUDA 13.3 + 驱动 580.x 组合已验证正常；CUDA 12.8 + 驱动 580.x 也正常。
> 实测覆盖：CUDA 12.8 ~ 13.3.33，驱动 580.76 ~ 610.47。

| 看到什么 | 怎么办 |
|---------|--------|
| `nvidia-smi: command not found` | 装显卡驱动：[NVIDIA 官网](https://www.nvidia.com/download/index.aspx) |
| `nvcc: command not found` | 装 CUDA Toolkit：[NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) |
| | 安装后 Linux: `export PATH=/usr/local/cuda/bin:$PATH` |
| | 安装后 Windows: `setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X"` |

---

### B-2. 确认 PyTorch + GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); cc=torch.cuda.get_device_capability(); print(f'sm_{cc[0]}{cc[1]}')"
```

**你应该看到**：
```
True
NVIDIA GeForce RTX xxxx
sm_XX
```

> 记下 SM 版本 — 如果看到 `sm_70` (V100/Volta)，请跳到 [V100 专用说明](#v100-volta-sm_70-专用说明)。

| 看到什么 | 怎么办 |
|---------|--------|
| `ModuleNotFoundError` | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| `False` | CUDA 版本和 PyTorch 不匹配。打开 [pytorch.org](https://pytorch.org/get-started/locally/) 按你的 CUDA 版本生成安装命令 |

---

### B-3. 确认 C++ 编译器

**Windows:**
```bash
where cl
```
**你应该看到**：`C:\Program Files\Microsoft Visual Studio\...\cl.exe`

| 看到什么 | 怎么办 |
|---------|--------|
| `Could not find files` | 装 [Build Tools for VS 2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)，勾选「使用 C++ 的桌面开发」 |

**Linux:**
```bash
g++ --version
```
**你应该看到**：`g++ (Ubuntu ...) 1x.x.x`

| 看到什么 | 怎么办 |
|---------|--------|
| `command not found` | Ubuntu: `sudo apt install build-essential` |

---

### B-4. 编译安装

```bash
pip install -e . --no-build-isolation
```

**你应该看到**：
```
running build_ext
building 'lowp_fft._cufft_ext' extension
...（编译输出，大量 warning 正常）...
Successfully built lowp_fft
Successfully installed lowp_fft-0.1.0
```

> `--no-build-isolation` 让 pip 使用系统已安装的 PyTorch 和 CUDA 环境，避免隔离环境缺少 CUDA 库导致链接失败。

| 看到什么 | 怎么办 |
|---------|--------|
| `ModuleNotFoundError: No module named 'setuptools'` | `pip install setuptools` 然后重试 |
| `CUDA version mismatches (13.3 vs 12.8)` | ✅ 已自动跳过版本检查，不影响。CUDA 13.x 用户常见 |
| `error: Microsoft Visual C++ 14.0 is required` | 回到 B-3 装 Build Tools |
| `LINK : fatal error LNK1181: c10_cuda.lib` | 确认用了 `--no-build-isolation` |
| `nvcc fatal: Unsupported GPU architecture` | 显卡太新/太旧。发 GPU 型号给项目负责人 |
| `cuFFT error 16` (CUFFT_EXEC_FAILED) | CUDA Toolkit 版本 > 驱动支持的 CUDA 版本。更新显卡驱动到最新 |

---

### B-5. 跑全量测试

```bash
python -m pytest tests/ -v --tb=short
```

**非 Volta (sm_80+) 你应该看到**：
```
test_bfp_fft.py .... 22 passed
test_bfp_cuda.py .... 1 failed, 14 skipped
test_autograd.py .... 33 passed, 2 skipped, 2 xfailed
test_bf16.py .... 39 passed, 2 skipped

综合: 1 failed, 94 passed, 17 skipped, 2 xfailed
```

> 🔑 **1 failed = test_exe_exists**，原因是 BFP CUDA 独立 exe 未编译。
> 这个 exe 走独立 Makefile 构建（与 `pip install -e .` 安装的 PyTorch 扩展无关）。
> **94 passed 已经是完整验证结果，不算失败。**
>
> **Linux 用户注意**：`build_bfp.bat` 是 Windows 批处理文件，Linux 下 test_exe_exists FAILED 同样是预期行为。

| 看到什么 | 怎么办 |
|---------|--------|
| 只有 1 个 FAILED (`test_exe_exists`) | ✅ 完整验证通过，94 passed |
| 其他非 BF16 测试 `FAILED` | 截图发给项目负责人 |
| 大量 BF16 测试 FAILED (30+) | 可能用 **V100/Volta** → 看下方专用说明 |
| passed 不到 94 (非 Volta) | 截图最终统计行 |
| `ModuleNotFoundError` | 回到检查清单确认 setuptools/pytest 已安装 |

---

## V100 / Volta (sm_70) 专用说明

**如果你的 GPU 是 V100 或其他 Volta 架构 (sm_70)，请先读这里。**

V100 的 Tensor Core 是第 1 代，**不支持 BF16 格式**。运行全量测试时：
- **54 个非 BF16 测试全部通过** ✅
- **41 个 BF16 测试会 FAIL** ⚠️ — 这是**预期行为，不是 bug**

你应该看到：
```
综合: 41 failed, 54 passed, 17 skipped, 2 xfailed
```

**验证标准 (V100)**：54 passed (非 BF16) + E4M3 SQNR 21.15±0.15 dB = 验证通过。

> 项目 API 层已通过 `_supports_bf16_cufft()` 自动检测 sm_70 并降级为 FP32 fallback。
> 用户使用 `fft(x, precision="bf16")` 在任何 GPU（包括 V100）上都能正常工作。
> pytest 测试套件的 BF16 测试尚未自动跳过 sm_70（改进中）。

**V100 验证记录模板**：
```
□ 路径 B (V100): 54 passed (非 BF16), 41 BF16 failed (预期), 17 skipped, 2 xfailed
```

---

## 复现消融实验（可选，2 分钟）

```bash
python tests/bench_bfp_ablation_mantissa.py
python tests/bench_bfp_ablation_group_size.py
```

**关键检查**：E4M3 uniform 的 SQNR 应在 **21.2 ± 0.3 dB**。
偏差 >0.5 dB 记录即可（不同机器浮点误差）。

> 基于 6 台机实测：E4M3 SQNR 极差仅 0.02 dB (21.15–21.17 dB)，跨机器高度一致。

---

## 常见问题速查

| 错误 | 场景 | 原因 | 解决 |
|------|------|------|------|
| `ModuleNotFoundError: No module named 'setuptools'` | pip install 时 | 部分 Python 环境不自带 setuptools | `pip install setuptools` 然后重试 |
| `ModuleNotFoundError: No module named 'torch'` | import 时 | 未安装 PyTorch | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| `CUDA toolkit not found` | 安装时 | 无 CUDA 环境 | 正常 — 自动启用纯 Python 模式 |
| `CUDA_HOME is not set` | 安装时 | 环境变量缺失 | 路径A 忽略；路径B 加 `--no-build-isolation` |
| `CUDA version mismatches (13.3 vs 12.8)` | CUDA 13.x 编译时 | nvcc 版本 > 驱动 CUDA 版本 | ✅ 已自动跳过，不影响 |
| `c10_cuda.lib` 找不到 | 编译链接时 | pip 隔离环境缺少 CUDA 库 | 加 `--no-build-isolation` |
| `could not find ninja` | 编译时 | ninja 未安装 | 正常 — 自动回退 setuptools，仅影响速度 |
| `Error checking compiler version for cl` | 编译时 | nvcc 与 MSVC 版本交互 | 正常 — nvcc 用 `--use-local-env` 找到 MSVC |
| `pip: WARNING: Cache entry deserialization failed` | pip 操作时 | pip 缓存损坏 | ✅ 无害，忽略 |
| `cuFFT error 16` (CUFFT_EXEC_FAILED) | V100 BF16 测试 | Volta 无 BF16 Tensor Core | ✅ 预期行为，见 V100 专用说明 |
| `cuFFT error 16` (CUFFT_EXEC_FAILED) | 非 V100 | CUDA Toolkit > 驱动支持版本 | 更新显卡驱动到最新 |
| `BackendUnavailable: No module named 'setuptools'` | pip install 时 | setuptools 缺失 | `pip install setuptools` 然后重试 |
| pytest 显示 `ImportError` | 跑测试时 | 安装不完整或目录错误 | 确认在 `low-precision-fft` 目录内，重新 `pip install -e .` |

---

## 🔧 遇到问题？一键诊断

### Windows

```powershell
.\scripts\collect-diagnostics.ps1
```

### Linux / 无 PowerShell

```bash
python -c "
import torch, sys, platform, subprocess, os
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'pip: ', end=''); subprocess.run([sys.executable, '-m', 'pip', '--version'])
print(f'setuptools: ', end=''); subprocess.run([sys.executable, '-c', 'import setuptools; print(setuptools.__version__)'])
try:
    print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    cc = torch.cuda.get_device_capability()
    print(f'SM: sm_{cc[0]}{cc[1]}')
except Exception as e:
    print(f'PyTorch/GPU: {e}')
print(); subprocess.run(['nvidia-smi'], stderr=subprocess.STDOUT)
print(); subprocess.run(['nvcc', '--version'], stderr=subprocess.STDOUT)
print(); subprocess.run(['g++', '--version'], stderr=subprocess.STDOUT)
" > diagnostics-$(date +%Y%m%d-%H%M%S).log 2>&1
```

诊断信息包括：
- 系统版本、Python/pip 版本
- PyTorch 版本、GPU 型号、Compute Capability
- nvidia-smi + nvcc 版本
- C++ 编译器版本
- CUDA 环境变量

将生成的 `.log` 文件发给项目负责人。

---

## ✅ 验证记录

复制对应模板，填完发给项目负责人：

### 非 Volta (sm_80+) 模板

```
=== 低精度 FFT 独立验证记录 ===

姓名：________
日期：2026-__-__
硬件：________ (GPU 型号)
SM：sm_XX
OS：Windows ____ / Linux ____
Python：3.__.__
CUDA Toolkit：v__.__
PyTorch：2.__.__
驱动：xxx.xx

□ 路径 A：BFP 22/22 passed, API fallback 72/72 passed
□ 路径 B：94 passed, 17 skipped, 2 xfailed, 1 failed (test_exe_exists)
□ 尾数消融 E4M3 SQNR：____ dB

遇到的问题（没有就写"无"）：
```

### V100 / Volta (sm_70) 模板

```
=== 低精度 FFT 独立验证记录 ===

姓名：________
日期：2026-__-__
硬件：Tesla V100 / V100S
SM：sm_70
OS：Linux ____
Python：3.__.__
CUDA Toolkit：v__.__
PyTorch：2.__.__
驱动：xxx.xx

□ 路径 A：BFP 22/22 passed, API fallback 72/72 passed
□ 路径 B (V100)：54 passed (非 BF16), 41 BF16 failed (预期), 17 skipped, 2 xfailed
□ 尾数消融 E4M3 SQNR：____ dB

遇到的问题（没有就写"无"）：
```

---

## 你完成了什么

- 路径 A：验证了 BFP FP8 算法 + API fallback 的正确性
- 路径 B：验证了 cuFFT FP16/BF16 扩展 + 自动微分（V100 仅验证非 BF16 部分）
- 消融实验：复现了论文核心 SQNR 数据

你的验证记录将作为独立可复现性证明，用于论文附录/答辩/中期报告。
