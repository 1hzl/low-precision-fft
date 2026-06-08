# Windows + NVIDIA GPU 空白环境测试指南

从零开始的完整测试流程。

---

## 环境要求总览

| 组件 | 用途 | 必需？ |
|------|------|--------|
| Python 3.10+ | 运行环境 | ✅ |
| PyTorch 2.x (CUDA版) | 深度学习框架 | ✅ |
| NVIDIA GPU 驱动 | GPU 计算 | ✅ |
| CUDA Toolkit | 编译 CUDA 扩展 | ⚠️ 测试A必需 |
| MSVC Build Tools | C++ 编译器 (Windows) | ⚠️ 测试A必需 |
| Git | 克隆仓库 | ✅ |

---

## 第一步：安装 Python 3.10+

**下载**：https://www.python.org/downloads/ （选 3.10 ~ 3.14 均可）

**安装时勾选**：
- ✅ `Add Python to PATH`

**验证**：
```powershell
python --version
# 应输出: Python 3.10+ 或更新
pip --version
```

> **问题**：`python` 命令找不到
> **解决**：重新运行安装程序 → 选 `Modify` → 勾选 `Add Python to PATH`，或手动添加 `C:\Python3XX` 到系统 PATH。

---

## 第二步：安装 Git

**下载**：https://git-scm.com/download/win

**验证**：
```powershell
git --version
```

---

## 第三步：安装 PyTorch (CUDA 版)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**验证**：
```powershell
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

预期：显示 PyTorch 版本 + `CUDA available: True`

> **问题**：`CUDA available: False`
> **解决**：检查 NVIDIA 驱动是否安装 → https://www.nvidia.com/drivers

---

## 第四步：安装 CUDA Toolkit + MSVC（仅测试A需要）

**如果只做测试B（无CUDA），跳到第五步。**

### 4a. CUDA Toolkit

**下载**：https://developer.nvidia.com/cuda-downloads

安装后验证：
```powershell
nvcc --version
# 应输出: Cuda compilation tools, release 1x.x
```

> **问题**：`nvcc` 命令找不到
> **解决**：
> 1. 确认 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin` 存在
> 2. 设置环境变量：`setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X"`
> 3. 重启终端

### 4b. MSVC Build Tools

**下载**：https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

安装时勾选：
- ✅ `MSVC v143 - VS 2022 C++ x64/x86 build tools`
- ✅ `Windows 11 SDK`（或 Windows 10 SDK）

> **问题**：编译报 `cl.exe` 找不到
> **解决**：运行 `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"` 后再执行 pip install，或重启终端。

---

## 第五步：克隆项目

```powershell
git clone https://github.com/1hzl/low-precision-fft.git
cd low-precision-fft
```

> 如果 GitHub 未推送，联系项目负责人获取仓库地址。

---

## 第六步：安装项目

### 测试 A：完整 CUDA 编译

```powershell
pip install -e . --no-build-isolation
```

**预期输出关键行**：
```
INFO:root:CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X
INFO:root:Detected GPU arch: sm_xx (device 0)
running build_ext
building 'lowp_fft._cufft_ext' extension
...编译过程（大量 warning 正常）...
Successfully built lowp_fft
Successfully installed lowp_fft-0.1.0
```

### 测试 B：纯 Python 安装（不需要 CUDA）

```powershell
pip install -e .
```

**预期**：安装成功（无 CUDA 扩展编译）。import 时使用 torch.fft fallback。

---

## 第七步：验证安装

```powershell
python -c "import lowp_fft; print('OK:', lowp_fft.fft)"
```

> **问题**：`ModuleNotFoundError: No module named 'lowp_fft'`
> **解决**：确认在项目根目录执行了 `pip install -e .`

---

## 第八步：运行测试

```powershell
pip install pytest numpy -q
python -m pytest tests/ -v --tb=short
```

**预期**：~97 passed, 4 skipped, 2 xfailed

### 各测试说明

| 文件 | 测试数 | 需要GPU | 说明 |
|------|--------|---------|------|
| `test_bfp_fft.py` | 22 | ❌ | BFP 原型（纯 Python） |
| `test_autograd.py` | 37 | 部分 | cuFFT FP16 自动微分 |
| `test_bf16.py` | 41 | 部分 | BF16 正/逆变换 |
| `test_bfp_cuda.py` | GPU only | ✅ | BFP CUDA kernel |

> **问题**：`test_bfp_cuda.py` 全部 skipped
> **解释**：正常，测试B 无 CUDA 扩展时自动跳过。

---

## 常见问题速查

| 错误 | 原因 | 解决 |
|------|------|------|
| `CUDA_HOME environment variable is not set` | 系统没配 CUDA_HOME | 设置环境变量或运行测试B |
| `The detected CUDA version mismatches` | CUDA 大版本 ≠ PyTorch 编译版本 | 可忽略 — 已自动跳过检查 |
| `error: Microsoft Visual C++ 14.0 is required` | 没装 MSVC Build Tools | 安装步骤 4b |
| `c10_cuda.lib` 找不到 | pip 隔离环境缺 CUDA 库 | 测试A 加 `--no-build-isolation` |
| `Defaulting to user installation` | 无权限写系统目录 | 正常 — 装到用户目录不影响 |
| `Attempted to use ninja...Falling back` | 没装 ninja | 正常 — 自动回退，仅影响编译速度 |

---

## 报告模板

完成测试后请填写：

```
OS:             [Windows 11 24H2 / Windows 10]
Python:         [3.12.0]
PyTorch:        [2.5.1+cu124]
CUDA Toolkit:   [有 v12.4 / 无]
GPU:            [RTX 5060 / 无]
测试类型:        [A 完整 / B 纯Python]
pytest 结果:     [97 passed / xx passed, xx failed]
安装结果:        [成功 / 失败]

如有失败，附 pip install 完整输出。
```
