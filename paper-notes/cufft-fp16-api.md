# cuFFT FP16 API 研读笔记

> 日期: 2026年5月 | 来源: NVIDIA cuFFT 13.3 文档 + PyTorch PR #180766

## 一、cuFFT 低精度支持概况

cuFFT 13.3 支持三种精度：
| 精度 | cudaDataType | 说明 |
|---|---|---|
| FP16 | `CUDA_C_16F` / `CUDA_R_16F` | 半精度，最高性能 |
| BF16 | `CUDA_C_16BF` / `CUDA_R_16BF` | SM_80+ (Ampere) |
| FP32 | `CUDA_C_32F` / `CUDA_R_32F` | 标准精度 |

**关键限制**: 转换尺寸必须为 2 的幂（power-of-two），这是 cuFFT 低精度路径的约束。

## 二、核心 API 流程

```c
// 1. 创建 plan
cufftHandle plan;
cufftCreate(&plan);

// 2. 设置尺寸和类型
size_t ws;
cufftXtMakePlanMany(plan, 1, &n,
    NULL, 1, 1, CUDA_C_16F,  // input: FP16 complex
    NULL, 1, 1, CUDA_C_16F,  // output: FP16 complex
    1, &ws, CUDA_C_16F);     // execution type

// 3. 分配 workspace
cudaMalloc(&d_work, ws);
cufftSetWorkArea(plan, d_work);

// 4. 执行
cufftXtExec(plan, d_input, d_output, CUFFT_FORWARD);

// 5. 清理
cufftDestroy(plan);
```

**与标准 cuFFT 的区别**: 低精度必须用 `cufftXt*` 系列 API（Extensible API），不能用旧的 `cufftPlan1d` + `cufftExecC2C`。

## 三、PyTorch 现有 FFT 实现路径

来自 PR #180766 的分析：

```
torch.fft.fft()
  → ATen dispatch
    → SpectralOps.cpp: promote_type_fft()
      → CUDA: CuFFTPlanCache.h → cufftXtMakePlanMany()
      → CPU: pocketfft
```

**PyTorch 已改动的文件**（PR #180766）:
- `aten/src/ATen/native/SpectralOps.cpp` — 类型提升逻辑
- `aten/src/ATen/native/cuda/SpectralOps.cu` — CUDA 后端
- `aten/src/ATen/native/cuda/CuFFTPlanCache.h` — cuFFT plan 缓存

**PR #180766 的状态**:
- **f16 on CUDA**: 已有原生路径（pre-existing native cuFFT path）
- **bf16 on CUDA**: PR 新增，SM_80+ 原生路径，否则降级到 FP32
- **f16/bf16 on XPU**: 提升到 FP32
- **bf16 R2C on CUDA**: 分配 ComplexHalf 代理 buffer

## 四、我们与 PR #180766 的差异化

| 维度 | PR #180766 | 我们的方向 |
|---|---|---|
| f16 CUDA | 已有原生路径 | 验证 + 基准测试 |
| bf16 CUDA | 通过 promotion 或代理 | 直接封装 cuFFT 原生 |
| **FP8** | 无 | **核心创新点** |
| CPU | 无低精度支持 | x86 AVX2 / ARM NEON |
| 应用验证 | 无 | LLM PEFT 场景 |

## 五、下一步：cuFFT FP16 hello world

CC 在笔记本执行：
```c
// 验证 cuFFT FP16 可用性
#include <cufft.h>
// 1D C2C, n=1024, FP16
// 对比 cufftXtExec(FP16) vs cufftExecC2C(FP32) 的:
//   - 执行时间
//   - 相对误差 (max |FP16 - FP32| / max |FP32|)
```

需要确认:
1. 笔记本 GPU 架构 ≥ SM_75？（`nvidia-smi --query-gpu=compute_cap --format=csv`）
2. CUDA 版本 ≥ 13.0？
3. cuFFT FP16 是否可用？
