# 低精度 FFT — 技术设计文档 v2

> 更新: 2026-06-02 | 基于 cuFFT 13.3 调研结论

## 一、背景调研

### cuFFT 现状
- NVIDIA cuFFT 13.3 官方支持 **Half-precision (FP16)** 和 **Bfloat16 (BF16)** 变换
- API: `cufftXtExec()` + `CUDA_C_16F` / `CUDA_C_16BF` 数据类型
- **不支持 FP8**

### PyTorch 现状
- `torch.fft` 仅支持 FP32/FP64，低精度输入会自动提升精度
- PR #180766（open）：试图支持 f16/bf16 输入，但内部提升到 FP32 计算——未利用 cuFFT 原生半精度
- **PyTorch 没有暴露 cuFFT 的低精度能力**

### 文献
- 低精度 FFT 的学术论文极少，FP8 FFT 全网未搜到实现
- 参考论文 2505.00582 验证了 FFT 在 LLM PEFT 中的实用价值（频域适配器）
- gearshifft（MPI-CBG）是目前最全面的 FFT 评测框架

---

## 二、技术方案

### 2.1 FP16/BF16 FFT via cuFFT → PyTorch（主力）

**架构**：
```
torch.fft.fft_lowp(input, precision="fp16")
  → ATen dispatch (CUDA)
    → cuFFT cufftXtExec() with CUDA_C_16F
    → 返回 FP16 Tensor
```

**技术要点**：
1. 使用 `cufftXtMakePlanMany()` 创建低精度 plan
2. 输入/输出均为 `__half` 类型，避免 FP32 中间提升
3. 对比基准：cuFFT FP32 vs FP16 的精度-吞吐量 trade-off
4. 自动微分：实现 `torch.autograd.Function` 的 forward/backward

### 2.2 FP8 FFT（创新点，自研 kernel）

cuFFT 不支持，需要自研：

```
算法方案（候选）:
A. 块浮点数 (Block Floating-Point): 共享指数，适合 FFT 蝶形结构
B. 动态缩放: 每层蝶形运算前后做 scale + clamp
C. 混合精度: 关键层 FP16，非关键层 FP8
```

**技术挑战**：
- FP8 E4M3 只有 3 位尾数，FFT 的 log₂N 级误差累积严重
- 需要误差补偿策略（Kahan summation / stochastic rounding）

### 2.3 CPU SIMD 低精度 FFT

cuFFT 仅支持 GPU。CPU 侧需要自研：

| 架构 | 指令集 | 数据类型 |
|---|---|---|
| x86_64 | AVX-512 (VNNI) / AVX2 | FP16 via `_mm512_cvtph_ps` |
| ARM | NEON / SVE | FP16 via `__fp16` |

优先 x86 AVX2（覆盖面最广），ARM NEON 后续补充。

---

## 三、API 设计

```python
# 新增接口
torch.fft.fft(input, n=None, dim=-1, norm=None, *, precision=None)
torch.fft.ifft(input, n=None, dim=-1, norm=None, *, precision=None)
torch.fft.rfft(input, n=None, dim=-1, norm=None, *, precision=None)
torch.fft.irfft(input, n=None, dim=-1, norm=None, *, precision=None)

# precision: "fp16" | "bf16" | "fp8" | None（默认 FP32）
# 返回 dtype 与 precision 一致
```

**行为**：
- `precision=None` → 现有行为不变（向后兼容）
- `precision="fp16"` → CUDA 走 cuFFT，CPU 走自研 kernel，返回 `float16`
- `precision="fp8"` → 全部自研 kernel，返回 `float8_e4m3fn`

---

## 四、精度基准

目标精度（相对 cuFFT FP32）：

| precision | 目标相对误差 | 吞吐量提升 |
|---|---|---|
| FP16 | < 1e-3 | 1.5× - 2× |
| BF16 | < 1e-2 | 1.5× - 2× |
| FP8 | < 1e-1 | 3× - 4×（粗估） |

---

## 五、决策记录

| 日期 | 决策 | 理由 |
|---|---|---|
| 6/2 | 放弃从零写 CUDA kernel，改用 cuFFT 封装 | cuFFT 13.3 已有原生 FP16/BF16 |
| 6/2 | FP8 作为核心创新点 | cuFFT 不支持，全网无实现 |
| 6/2 | CPU 优先级降低（先 GPU） | cuFFT 已解决 GPU 侧，CPU 后续补充 |
| 6/2 | 三人团队 → Han 统筹 + OpenClaw 调度 + CC 执行 | 目前只有两人在推进，去掉詹世显/刘子渊的分工 |
| 6/2 | 底层基础设施优先于任务委派 | 先搭脚手架（审查标准/实验模板/MemPalace），再派开发任务 |
| 6/2 | PLAN.md + DESIGN.md 为单一信息源 | DEV-PLAN.md/SPEC.md 已合并，去重避免维护多份 |

---

## 六、参考资料

- cuFFT 13.3 docs: https://docs.nvidia.com/cuda/cufft/index.html
- PyTorch FFT PR #180766: https://github.com/pytorch/pytorch/pull/180766
- gearshifft FFT benchmark: https://github.com/mpicbg-scicomp/gearshifft
- 频域 PEFT: arXiv 2505.00582 (Block Circulant Adapter)
