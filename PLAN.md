# 低精度 FFT — 实施计划 v2

> **团队**: 韩志麟（统筹决策） + OpenClaw（研究/文档/协调） + Claude Code（编码/实验）  
> **日期**: 2026-06-02  
> **目标**: FP16/FP8 低精度 FFT 多架构适配 → PyTorch 社区贡献

---

## 角色分工

| 角色 | 职责 | 工具 |
|---|---|---|
| **韩志麟** | 方向决策、算法方案确认、代码审查、导师对接 | 笔记本 + Claude Code |
| **OpenClaw** | 文献搜索分析、技术文档、任务拆解、进度追踪 | N2920 + pipeline |
| **Claude Code** | 编码实现、CUDA kernel、PyTorch 封装、实验 | 笔记本 GPU |

**协作流**: 韩志麟决策 → OpenClaw 拆解委派 → CC 实现 → 韩志麟审查 → 迭代

---

## 阶段一：方案定稿（6/2 - 6/8）

### 1.1 文献补充

> 当前只有 2505.00582 一篇。需要至少 10 篇覆盖以下领域。

| 领域 | 关键词 | 责任人 |
|---|---|---|
| 低精度 FFT 理论 | "low precision FFT" "mixed precision FFT" | OpenClaw |
| 混合精度训练 | "mixed precision training" "FP16 training" "autocast" | OpenClaw |
| CUDA FFT 优化 | "CUDA FFT optimization" "cufft fp16" | OpenClaw |
| PyTorch 算子开发 | "PyTorch custom op" "torch.autograd.Function" | OpenClaw |
| LLM PEFT + 频域 | "frequency domain fine-tuning" "Fourier PEFT" | OpenClaw |

产出：每篇 → `paper-notes/<id>.md`（REF-NOTE 格式），由 OpenClaw 搜索 + 初稿，CC 补充技术细节。

### 1.2 技术方案升级

当前 DESIGN.md 只有一句话。需要升级为可执行方案：

| 章节 | 内容 | 责任人 |
|---|---|---|
| 算法设计 | FP16/FP8 FFT 的误差模型、动态缩放策略 | OpenClaw 起草 → 韩志麟确认 |
| CUDA 内核设计 | 线程块策略、共享内存布局、Cooley-Tukey 分解 | OpenClaw 起草 → CC 细化 |
| CPU SIMD 选型 | x86 AVX-512 vs ARM NEON，优先级排序 | 韩志麟决策 |
| PyTorch 接口 | `torch.fft.fft_lowp()` API 设计 | OpenClaw 提案 |
| 误差补偿 | Kahan summation / block floating-point 方案比较 | OpenClaw |

产出：`DESIGN.md` ≥ 2000 字，含算法伪代码和 API 签名。

### 1.3 开发环境确认

| 检查项 | 命令/方法 |
|---|---|
| GPU + CUDA 可用 | `nvidia-smi` |
| PyTorch CUDA 版本 | `python -c "import torch; print(torch.cuda.is_available())"` |
| cuFFT 可用 | 编译一个 cuFFT hello world |
| CMake ≥ 3.18 | `cmake --version` |

---

## 阶段二：MVP 原型（6/9 - 6/22）

### 目标：FP16 FFT 单层 CUDA kernel，能跑通精度测试

| Sprint | 任务 | 产出 |
|---|---|---|
| **Sprint 1** (6/9-6/12) | CPU 端 Python 原型：纯 NumPy/PyTorch 实现 FP16 FFT + 精度对比 | `prototype/fp16_fft.py` + 误差数据 |
| **Sprint 2** (6/13-6/16) | CUDA kernel v0：并行 Cooley-Tukey，共享内存 | `cuda/fft_fp16.cu` + 编译通过 |
| **Sprint 3** (6/17-6/19) | 精度验证：kernel vs FP32 cuFFT 对比 | `experiments/` 实验日志 |
| **Sprint 4** (6/20-6/22) | 误差补偿优化 + kernel v1 | 误差降低到可接受范围 |

验收标准：FP16 FFT 相对误差 < 1e-3，吞吐量接近 FP32 cuFFT 的 1.5×。

---

## 阶段三：多精度 + 封装（6/23 - 7/6）

| 任务 | 产出 |
|---|---|
| FP8 FFT kernel | `cuda/fft_fp8.cu` |
| PyTorch C++ 扩展 | `torch.fft.fft_lowp(input, precision="fp16")` |
| 自动微分支持 | backward pass 通过测试 |
| CPU SIMD 第一版 | x86 AVX-512 或 ARM NEON（取决于硬件） |

验收标准：`torch.fft.fft_lowp()` 可在 LLM 微调 pipeline 中替换 `torch.fft.fft()`。

---

## 阶段四：测试 + 社区提交（7/7 - 7/31）

| 任务 |
|---|
| 基准测试矩阵（精度 × 性能 × 架构） |
| 单元测试（CPU + GPU） |
| 文档（Usage Guide + Design Doc） |
| PyTorch PR 初稿 + Discuss 论坛发帖 |

---

## 协作协议

**韩志麟 → OpenClaw**（飞书）：
- 说「做 xxx」→ OpenClaw 拆解 → 写入 pipeline HANDSHAKE
- 审查完代码说「通过」/「驳回」→ 触发下一轮

**OpenClaw → CC**（pipeline HANDSHAKE + wake-laptop）：
- 每轮只派 1 个任务（Ivy Lee 规则）
- 任务格式: `[ ] 任务标题` + 具体描述 + 期望产出
- CC 完成 → 更新 LAPTOP-CHANGES.md → OpenClaw pull 查看

**CC → OpenClaw**（git push → LAPTOP-CHANGES.md）：
- 代码 push 到 N2920 bare repo
- 结果写在 LAPTOP-CHANGES.md

---

*版本: v2 | 更新: 2026-06-02 | 下一步: 阶段一 1.1 文献补充*
