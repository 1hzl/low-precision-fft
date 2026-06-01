# PyTorch FFT 源码研读 + 集成方案

> 日期: 2026-06-02 | 来源: PyTorch main branch

## 一、PyTorch cuFFT 集成架构

### 关键文件

| 文件 | 作用 |
|---|---|
| `aten/src/ATen/native/cuda/CuFFTPlanCache.h` | cuFFT plan 缓存 + dtype 映射 |
| `aten/src/ATen/native/cuda/CuFFTUtils.h` | cuFFT 错误处理 + 常量 |
| `aten/src/ATen/native/cuda/SpectralOps.cu` | CUDA FFT 执行 |
| `aten/src/ATen/native/SpectralOps.cpp` | FFT 前端分发 (CPU/CUDA dispatch) |

### 当前 dtype → cuFFT 类型映射（CuFFTPlanCache.h:297-314）

```cpp
// 当前支持的 3 种精度：
if (dtype == ScalarType::Float) {
    itype = complex ? CUDA_C_32F : CUDA_R_32F;
    exec_type = CUDA_C_32F;
} else if (dtype == ScalarType::Double) {
    itype = complex ? CUDA_C_64F : CUDA_R_64F;
    exec_type = CUDA_C_64F;
} else if (dtype == ScalarType::Half) {
    itype = complex ? CUDA_C_16F : CUDA_R_16F;  // ✅ FP16 已有！
    exec_type = CUDA_C_16F;
} else {
    TORCH_CHECK(false, "cuFFT doesn't support tensor of type: ", dtype);
}
```

**关键发现**: PyTorch **已有** FP16 cuFFT 支持！`kHalf` → `CUDA_C_16F` / `CUDA_R_16F`。

### FP16 的限制（CuFFTPlanCache.h:266-271）

```cpp
if (dtype == ScalarType::Half) {
    // cuFFT on half requires compute capability of at least SM_53
    auto dev_prop = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(dev_prop->major >= 5 && !(dev_prop->major == 5 && dev_prop->minor < 3),
                "cuFFT doesn't support …");
}
```

要求 SM_53+。笔记本 RTX 5070 Ti (SM_120) 远超要求。

## 二、我们的工作

### 2.1 验证现有 FP16 路径

在笔记本上测试：
```python
import torch
x = torch.randn(1024, dtype=torch.complex64).cuda().half()
y = torch.fft.fft(x)  # 应该直接用 cuFFT FP16
print(y.dtype)  # 期望 torch.complex32
```

### 2.2 添加 BFloat16 支持

在 `CuFFTPlanCache.h` 的 dtype 映射中加：

```cpp
} else if (dtype == ScalarType::BFloat16) {
    TORCH_CHECK(dev_prop->major >= 8,  // SM_80+ for BF16
                "cuFFT bfloat16 requires SM_80+");
    itype = complex_input ? CUDA_C_16BF : CUDA_R_16BF;
    otype = complex_output ? CUDA_C_16BF : CUDA_R_16BF;
    exec_type = CUDA_C_16BF;
}
```

在 `SpectralOps.cpp` 的 `promote_type_fft()` 中添加 `kBFloat16` 放行。

### 2.3 FP8 自研 kernel（核心创新）

cuFFT 不支持 FP8，需要自己写 CUDA kernel。

**方案：块浮点数 (Block Floating-Point)**

```
对于 N 点 FFT (log₂N 层蝶形):
  每层: 共享指数 (shared exponent) ← 所有蝶形输出共享一个 scale
  输入: FP8 E4M3, exponent = max_exp_in_block
  蝶形: 转换成 int16 做定点运算, 结果 scale 后存回 FP8
  输出: FP8 E4M3, 带 scale 元数据
```

**伪代码**：
```cuda
__global__ void fft_fp8_block_kernel(
    __nv_fp8_e4m3* data,  // FP8 complex interleaved
    float* scales,         // 每层每个 block 的 scale
    int n, int batch, int log2_n)
{
    extern __shared__ int16_t smem[];
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    
    // 步1: FP8 → int16 (带 block scale)
    // 步2: log₂N 层 Cooley-Tukey 蝶形 (定点运算)
    // 步3: int16 → FP8 (反量化)
    // 步4: 存储 scale 元数据
}
```

**误差预估**：
- FP8 E4M3: 3 位尾数，相对误差 ≈ 2⁻³ = 0.125
- log₂N 层累积: 理论最大误差 ≈ log₂N × 0.125
- N=1024: max error ≈ 10 × 0.125 = 1.25（不可接受）
- 需要 block floating-point 方案将有效尾数提升到 ~8 位
- 目标: 相对误差 < 1e-1 (10%)

## 三、PyTorch 集成入口

### 需要修改的文件（优先级排序）

| 优先级 | 文件 | 改动 |
|---|---|---|
| P0 | `CuFFTPlanCache.h` | 加 BFloat16 dtype 映射 |
| P0 | `SpectralOps.cpp` | `promote_type_fft()` 放行 kBFloat16 |
| P1 | `cuda/SpectralOps.cu` | 加 FP8 custom kernel dispatch |
| P1 | `native_functions.yaml` | 注册 `fft_lowp` 新算子 |
| P2 | `test_spectral_ops.py` | 精度 + 性能测试 |

### API 设计

```python
# 方案 A: 新增 precision 参数（推荐）
torch.fft.fft(x, precision="fp16")   # 走 cuFFT FP16
torch.fft.fft(x, precision="bf16")   # 走 cuFFT BF16 (新增)
torch.fft.fft(x, precision="fp8")    # 走自研 FP8 kernel

# 方案 B: 直接传低精度 tensor
x_fp16 = x.to(torch.float16)
torch.fft.fft(x_fp16)  # 自动走 FP16 路径
```

**方案 B 更符合 PyTorch 惯例**，且已有 FP16 路径支持。我们只需加 BF16 和 FP8。
