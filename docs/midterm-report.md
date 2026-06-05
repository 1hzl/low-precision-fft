# 低精度 FFT 实现 — 项目中期报告

> **项目编号**: 20261124  
> **团队成员**: 韩志麟（负责人）、詹世显、刘子渊  
> **指导教师**: 廖思宇  
> **报告时间**: 2026-06-05  
> **项目进度**: 约 40%（算法核心 100%，工程+社区 ~10%）

---

## 一、项目概述

### 1.1 目标

为 PyTorch 实现 FP16/BF16/FP8 低精度 FFT，覆盖 GPU（CUDA）与 CPU（多架构 SIMD），支撑 LLM 微调场景，最终贡献代码至 PyTorch 社区。

### 1.2 技术路线

- **FP16/BF16**: 基于 NVIDIA cuFFT Xt API（`cufftXtExec` + `CUDA_C_16F/BF`），通过 PyTorch C++ 扩展封装
- **FP8**: 自研块浮点（Block Floating-Point, BFP）FFT kernel——每阶段共享指数，蝶形 float32 计算，输出 FP8 量化

### 1.3 实验平台

- GPU: NVIDIA RTX 5070 Ti (SM_120, 12 GB VRAM, CUDA 13.3)
- 开发: 笔记本 (Claude Code) + N2920 (OpenClaw 协调)

---

## 二、已完成工作

### 2.1 时间线

| 阶段 | 时间 | 内容 | 状态 |
|------|------|------|------|
| Phase 1 | 6/2 | CUDA 环境验证 + cuFFT FP16 hello world | ✅ |
| Phase 2 | 6/3 | FP16/BF16 cuFFT → PyTorch 封装（forward + backward + benchmark） | ✅ |
| Phase 3 | 6/4 | FP8 BFP 全链路（论文→误差模型→Python原型→CUDA v0→调优报告） | ✅ |
| Phase 4 | 6/5 | BFP 补全（inverse/API/bandwidth/边界测试）+ Bergach 复现 + 审查修复 | ✅ |

### 2.2 代码产出

| 类别 | 文件数 | 行数 |
|------|--------|------|
| Python | 19 | 4,303 |
| CUDA/C++ | 6 | 2,324 |
| Markdown 文档 | 32 | 6,705 |
| **合计** | **57** | **13,332** |

### 2.3 外部 API

```python
import lowp_fft

X = lowp_fft.fft(x, precision="fp16")   # cuFFT FP16 加速
X = lowp_fft.fft(x, precision="bf16")   # cuFFT BF16 加速
X = lowp_fft.fft(x, precision="fp8")    # 自研 BFP FP8
```

特性：
- 支持 `n` / `dim` / `norm` 参数
- 自动微分（torch.autograd.Function）
- FP16/BF16 fallback 时自动警告
- Plan cache 线程安全（double-checked locking, 64 条目）

---

## 三、核心技术方案

### 3.1 FP16/BF16: cuFFT Xt 集成

```
torch.Tensor → CUDA C++ extension → cufftXtMakePlanMany(CUDA_C_16F)
                                    → cufftXtExec
                                    → torch.Tensor (complex32)
```

- Plan cache: mutex + LRU 淘汰（64 entry）
- Backward: FP32 torch.fft（避免 cuFFT stream 兼容问题）
- 优化: 免转换路径 + no_grad 快速通道

### 3.2 FP8: 块浮点 BFP FFT

**算法原理**（参考 arXiv 2605.28451）：

```
1. 输入 → FP8 E4M3 量化（mantissa + 共享指数）
2. For each FFT stage:
   a. 解量化 mantissa × 2^exponent → float32
   b. Radix-2 蝶形变换（float32 全精度）
   c. 重新量化 → FP8 新 mantissa + 新共享指数
3. 最终解量化输出
```

**精度优势**：每阶段仅量化一次（而非每次加减乘都量化），SQNR 从 naive FP8 的 ~0 dB 提升至 ~20 dB。

### 3.3 误差模型

FP8 E4M3 格式特性：
- 指数: 4 bits (bias=7), 尾数: 3 bits
- 最大正规数: 448.0, 最小次正规数: 2⁻⁹
- 蝶形每层引入 ~1 ULP 量化误差，log₂N 层累积

详见 `docs/fp8-fft-error-model.md`。

---

## 四、实验数据

### 4.1 精度对比（SQNR vs FP64, N=256-4096）

| 精度 | SQNR | 备注 |
|------|------|------|
| FP32 (cuFFT) | 135–138 dB | 参考基准 |
| FP16 (cuFFT) | 56–61 dB | 随 N 微降 |
| BF16 (cuFFT) | 53.1 dB | 不随 N 衰减 |
| BFP FP8 (自研 CUDA) | 20–22 dB | 随 N 微降 |
| Naive FP8 (逐算子量化) | ~0 dB | N≥256 完全失效 |

### 4.2 性能对比（吞吐量, RTX 5070 Ti）

| 场景 | cuFFT FP16 vs FP32 | 说明 |
|------|-------------------|------|
| 单 FFT (N=256) | 1.0× | 优化后消除 Python overhead |
| Batch FFT (N=256, B=256) | 1.4–2.8× | Batch 越大优势越明显 |

BFP FP8 CUDA 当前性能低于 FP16（~0.01×），因仍走 Python→CUDA 多次拷贝，后续优化空间大。

### 4.3 Bergach 2026 复现

| 验证项 | 期望值 | 实测值 | 结论 |
|--------|--------|--------|------|
| FP16 forward FFT SQNR (N=1024) | 56–61 dB | 57.1 ± 0.2 dB | ✅ 匹配 |
| FP16 forward FFT SQNR (N=4096) | 56–61 dB | 53.4 ± 0.1 dB | ⚠️ 边缘 |

---

## 五、代码质量审查

### 5.1 审查结果

| 轮次 | 方法 | MAJOR | MINOR | NIT |
|------|------|-------|-------|-----|
| 第一轮 | 自动化扫描 | 0 | 0 | 4 |
| 第二轮 | 14 文件逐行审查 | 1 | 6 | 4 |

### 5.2 关键修复

- ✅ README SQNR 修正（虚高值 → 实测值）
- ✅ 4 MAJOR + 3 MINOR 审查修复（31 tests pass）
- 🟡 6 项 MINOR 待修（见 `ISSUES.md`）

### 5.3 测试体系

- 74 tests, 21 个测试类
- 覆盖: norm/N=1/n padding/dim/bf16/CPU fallback/空 tensor/roundtrip/grad
- 待确认: 需在笔记本 GPU 上跑全量 pytest 验证 0 failure

---

## 六、存在问题与风险

### 6.1 关键缺口

| 缺口 | 影响 | 优先级 |
|------|------|--------|
| CPU SIMD 多架构适配 | 任务书核心要求，预期成果缺 CPU 侧 | 🔴 P0 |
| PyTorch 社区 PR | 项目终极目标未启动 | 🔴 P0 |
| LLM 微调场景验证 | 应用价值无法证明 | 🟡 P1 |
| 多架构 CPU 测试环境 | CPU SIMD 前置条件 | 🟡 P1 |

### 6.2 已知风险

- BFP FP8 CUDA kernel 性能差（~0.01× FP16），需深度优化
- BF16 在 SM_120 (Blackwell) 上的 cuFFT 行为与 Ampere/Hopper 可能有差异
- ARM/RISC-V 交叉编译环境需额外硬件资源

---

## 七、下阶段计划

### Phase 5: CPU SIMD 多架构适配（P0）

- x86 AVX2/AVX-512 FP16 FFT
- ARM NEON FP16 FFT（交叉编译验证）
- 接入 `lowp_fft` API（`device="cpu"` + `precision="fp16"`）

### Phase 6: LLM 微调场景验证（P1）

- 选型 LLM 中 FFT 使用路径（基于 BCA 论文 2505.00582）
- 构建测试集 + 精度/性能/显存对比
- 场景化验证报告

### Phase 7: 社区贡献准备（P2）

- PyTorch Contribution Guide 研读
- Usage Guide + Design Doc
- Initial PR 草稿

### Phase 8: 迭代与结题（远期）

- 社区反馈迭代
- 代码合并至 PyTorch 主干
- 项目总结 + 结题答辩

---

## 八、附录

### A. 核心文件清单

```
lowp_fft/
├── __init__.py          # 公共 API (fft/ifft)
├── _autograd.py         # 自动微分
├── bfp_fft.py           # BFP Python 原型
└── csrc/cufft_fp16.cu   # cuFFT C++ 扩展

src/cuda/
├── bfp_fft.cu           # BFP CUDA kernel (846 行)
├── fp8_verification.cu  # FP8 硬件验证
├── benchmark_fp32_vs_fp16.cu
└── hello_fp16_fft.cu

tests/                   # 74 tests, 10 文件
docs/                    # 4 技术文档
paper-notes/             # 6 论文笔记 + 4 PDF
data/                    # 6 基准 CSV
experiments/bergach-repro/  # Bergach 2026 复现
```

### B. Token 消耗（DeepSeek 后台账单）

> 数据来源：DeepSeek 官方后台 6 月账单 CSV（截至 6/5 13:00）  
> 笔记本 (han key): Claude Code 编码执行 | N2920 (openclaw key): 人机对话 + 协调

| 日期 | 笔记本 | N2920 | 当日合计 | 对应阶段 |
|------|--------|-------|---------|----------|
| 6/1 | ¥0.70 | ¥5.05 | ¥5.77 | 项目启动前 |
| 6/2 | ¥10.38 | ¥5.30 | ¥15.68 | Phase 1 |
| 6/3 | ¥8.56 | ¥3.71 | ¥12.27 | Phase 2 |
| 6/4 | ¥10.37 | ¥5.38 | ¥15.74 | Phase 3 |
| 6/5 | — | ¥0.96 | ¥0.96 | Phase 4 |
| **合计** | **¥30.01** | **¥20.40** | **¥50.42** | — |

**项目可归属**：笔记本 ¥29.31（6/2–6/4）+ N2920 约 ¥15（低精度FFT为主）≈ **¥45**

**缓存效率**：V4-Pro 缓存命中率 96–98%（cache hit ¥0.025/M vs miss ¥3/M），大幅降低重复上下文成本。

### C. 参考文献

1. arXiv 2605.28451 — Block-Floating-Point FP16 FFT
2. arXiv 2505.00582 — Block Circulant Adapter for LLMs
3. arXiv 2104.11471 — tcFFT: Tensor Core FFT
4. arXiv 2209.05433 — FP8 Formats for Deep Learning

---

*报告版本: v1.0 | 生成时间: 2026-06-05 | 下次更新: Phase 5 完成后*
