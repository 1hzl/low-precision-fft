# 独立性测试指南

验证 low-precision-fft 在不同环境下能否正常安装和运行。

## 准备工作

```bash
git clone https://github.com/1hzl/low-precision-fft.git
cd low-precision-fft
```

> 如果还没有推 GitHub，用 `git clone <repo-url>` 替代。

---

## 测试矩阵

| 测试 | 环境 | 预期结果 |
|------|------|---------|
| A | 有 CUDA（GPU + nvcc） | CUDA 扩展编译通过，pytest 全绿 |
| B | 无 CUDA（CPU only） | 纯 Python 安装通过，import 正常 |

---

## 🪟 Windows 用户

### 前置条件

```powershell
# 确认有 Python 3.10+ 和 PyTorch
python --version
python -c "import torch; print(torch.__version__)"
```

### 测试 A：有 CUDA

```powershell
# 安装（跳过 pip 隔离，用系统环境编译 CUDA 扩展）
pip install -e . --no-build-isolation

# 运行测试
pip install pytest numpy -q
python -m pytest tests/ -v --tb=short

# 可选：运行自动化测试脚本
.\docker\independence-test-win.ps1
```

**预期**：`pip install` 编译 CUDA 扩展，显示 `building 'lowp_fft._cufft_ext' extension`，链接成功。pytest 97 tests passed。

### 测试 B：无 CUDA / CUDA 环境未配

```powershell
# 正常安装（pip 隔离环境）
pip install -e .

# 验证 import
python -c "import lowp_fft; print('OK')"

# 纯 Python 测试（不需要 GPU）
python -m pytest tests/test_bfp_fft.py -v
```

**预期**：`pip install` 成功（无 CUDA 扩展编译），`import lowp_fft` 成功，`test_bfp_fft.py` 22 passed。

---

## 🐧 Linux 用户

### 前置条件

```bash
python3 --version  # 3.10+
python3 -c "import torch; print(torch.__version__)"
```

### 测试 A：有 CUDA

```bash
pip install -e . --no-build-isolation
pip install pytest numpy -q
python -m pytest tests/ -v --tb=short
```

### 测试 B：无 CUDA（Docker）

```bash
# 构建并运行 Linux Docker 测试
bash docker/run-independence-test.sh
```

或直接裸跑：

```bash
pip install -e .
python3 -c "import lowp_fft; print('OK')"
python3 -m pytest tests/test_bfp_fft.py -v
```

---

## 报告结果

请提供以下信息：

```
OS:          [Windows 11 / Ubuntu 22.04 / ...]
Python:      [3.10 / 3.12 / ...]
PyTorch:     [2.x.x+cuXXX]
CUDA:        [有/无, 版本号]
GPU:         [型号或"无"]
测试:        [A / B]
结果:        [通过 / 失败]

如果失败，附上完整错误输出。
```
