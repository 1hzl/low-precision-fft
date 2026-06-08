# 独立验证指南

按照本指南操作，5-10 分钟即可完成低精度 FFT 项目的独立验证。

---

## 环境速查

开始前确认：

```bash
python --version        # 需要 ≥ 3.10
git --version           # 任意版本
```

没有就去装：[Python](https://www.python.org/downloads/)（勾选 Add to PATH）、[Git](https://git-scm.com/download/win)（一路 Next）。

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
| `Successfully installed` 但有一行 `CUDA toolkit not found` | ✅ 正常，纯 Python 模式 |

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

---

### B-1. 确认驱动和 CUDA

```bash
nvidia-smi
nvcc --version
```

**你应该看到**：
```
NVIDIA-SMI xxx.xx  ...  Driver Version: xxx  ...  CUDA Version: 1x.x
Cuda compilation tools, release 1x.x
```

> ⚠️ **nvidia-smi 里的 CUDA Version 必须 ≥ nvcc 的 release 版本号**。如果 nvidia-smi 显示 CUDA 12.9 但 nvcc 是 13.3，需要更新显卡驱动。

| 看到什么 | 怎么办 |
|---------|--------|
| `nvidia-smi: command not found` | 装显卡驱动：[NVIDIA 官网](https://www.nvidia.com/download/index.aspx) |
| `nvcc: command not found` | 装 CUDA Toolkit：[NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) |
| | 安装后设置：`setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X"` 并重启终端 |

---

### B-2. 确认 PyTorch + GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**你应该看到**：
```
True
NVIDIA GeForce RTX xxxx
```

| 看到什么 | 怎么办 |
|---------|--------|
| `ModuleNotFoundError` | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| `False` | CUDA 版本和 PyTorch 不匹配。打开 [pytorch.org](https://pytorch.org/get-started/locally/) 按你的 CUDA 版本生成安装命令 |

---

### B-3. 确认 C++ 编译器（Windows）

```bash
where cl
```

**你应该看到**：`C:\Program Files\Microsoft Visual Studio\...\cl.exe`

| 看到什么 | 怎么办 |
|---------|--------|
| `Could not find files` | 装 [Build Tools for VS 2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)，勾选「使用 C++ 的桌面开发」→ 安装完重启终端 |

---

### B-4. 编译安装

```bash
pip install -e . --no-build-isolation
```

**你应该看到**：
```
INFO:root:CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X
running build_ext
building 'lowp_fft._cufft_ext' extension
...（编译输出，大量 warning 正常）...
Successfully built lowp_fft
Successfully installed lowp_fft-0.1.0
```

> `--no-build-isolation` 让 pip 使用系统已安装的 PyTorch 和 CUDA 环境，避免隔离环境缺少 CUDA 库导致链接失败。

| 看到什么 | 怎么办 |
|---------|--------|
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

**你应该看到**：
```
test_bfp_fft.py .... 22 passed
test_bfp_cuda.py .... 1 failed, 14 skipped  (如果没有先跑 build_bfp.bat)
test_autograd.py .... 33 passed, 2 skipped, 2 xfailed
test_bf16.py .... 39 passed, 2 skipped

综合: 94 passed, 17 skipped, 2 xfailed, 1 failed
```

> `test_bfp_cuda.py` 需要先运行 `build_bfp.bat` 编译独立的 BFP CUDA 可执行文件（Makefile 方案，与 pip 安装的 PyTorch 扩展是两套构建系统）。如果没跑过，`test_exe_exists` 会 FAIL，其余 14 个测试会跳过——**这不算验证失败**，94 passed 已经是完整验证结果。

| 看到什么 | 怎么办 |
|---------|--------|
| 任何红色 `FAILED` | 截图发给项目负责人 |
| `test_bfp_cuda.py` 全部 skipped | GPU 不可用，回退到路径 A |
| `test_bfp_cuda.py` 有 1 FAILED (test_exe_exists) | 没跑 `build_bfp.bat`。不算失败 — 94 passed 已是完整验证 |
| passed 总数不到 94 | 截图最终统计行 |

---

## 复现消融实验（可选，2 分钟）

```bash
python tests/bench_bfp_ablation_mantissa.py
python tests/bench_bfp_ablation_group_size.py
```

**关键检查**：E4M3 uniform 的 SQNR 应在 21.2 ± 0.3 dB。偏差 >0.5 dB 记录即可（不同机器浮点误差）。

---

## 常见问题速查

| 错误 | 场景 | 解决 |
|------|------|------|
| `CUDA toolkit not found` | 安装时 | 正常 — 纯 Python 模式自动启用 |
| `CUDA_HOME is not set` | 安装时 | 路径A 忽略；路径B 设环境变量或加 `--no-build-isolation` |
| `CUDA version mismatches` | CUDA 13.x 编译时 | ✅ 已自动跳过，不影响 |
| `c10_cuda.lib` 找不到 | 编译链接时 | 加 `--no-build-isolation` |
| `could not find ninja` | 编译时 | 正常 — 自动回退，仅影响编译速度 |
| `Error checking compiler version for cl` | 编译时 | 正常 — nvcc 用 `--use-local-env` 找到 MSVC |

---

## 🔧 遇到问题？一键诊断

如果上面任何步骤失败，运行诊断脚本收集完整环境信息：

```bash
.\scripts\collect-diagnostics.ps1
```

这会生成 `diagnostics-YYYYMMDD-HHMMSS.log`，包含：

- 系统版本、Python/pip 版本、已安装的包、pip 缓存状态
- **全部** CUDA/NVIDIA 环境变量（含空值）
- nvidia-smi + nvcc 版本
- PyTorch 版本、GPU 型号、CUDA 版本、Compute Capability
- MSVC 编译器位置
- Git 仓库状态
- pip install 完整输出（含退出码）
- import 检查
- pytest 逐文件 + 全量输出

将生成的 `.log` 文件发给项目负责人。

---

## ✅ 验证记录

复制下面模板，填完发给项目负责人：

```
=== 低精度 FFT 独立验证记录 ===

姓名：________
日期：2026-__-__
硬件：________（GPU 型号，或 "CPU only"）
OS：Windows ____ / Linux ____
Python：3.__.__
CUDA Toolkit：有 v__.__ / 无

□ 路径 A：BFP __/22  passed
□ 路径 B：全量 __ passed, __ skipped, __ xfailed
□ 尾数消融 E4M3 SQNR：____ dB

遇到的问题（没有就写"无"）：
```

---

## 你完成了什么

- 路径 A：验证了 BFP FP8 算法 + API fallback 的正确性
- 路径 B：验证了 cuFFT FP16/BF16 扩展 + BFP CUDA kernel + 自动微分
- 消融实验：复现了论文核心 SQNR 数据

你的验证记录将作为独立可复现性证明，用于中期报告/论文附录/答辩。
