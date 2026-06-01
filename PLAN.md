# 低精度 FFT — 实施计划 v3

> **团队**: 韩志麟（统筹决策） + OpenClaw（研究/文档/协调） + Claude Code（编码/实验）  
> **更新**: 2026-06-02（基于 cuFFT 13.3 调研结论）  
> **目标**: 将低精度 FFT 引入 PyTorch → FP16/BF16（cuFFT封装） + FP8（自研） → 社区贡献

---

## 角色分工

| 角色 | 职责 |
|---|---|
| **韩志麟** | 方向决策、算法确认、代码审查、导师对接 |
| **OpenClaw（N2920）** | 文献分析、技术文档、任务拆解委托 |
| **Claude Code（笔记本）** | 编码实现、CUDA 实验、PyTorch 封装 |

---

## 阶段一：方案定稿 + 环境验证（6/2 - 6/5）

| 任务 | 产出 | 负责 |
|---|---|---|
| 研读 cuFFT FP16 API（`cufftXtExec` + `CUDA_C_16F`） | API 使用笔记 | OpenClaw |
| 研读 PyTorch `torch.fft` 源码（C++ 后端 + ATen dispatch） | 源码走读笔记 | OpenClaw + CC |
| 确认笔记本 CUDA 环境 | `nvidia-smi` + cuFFT 版本号 | CC |
| 写 cuFFT FP16 FFT hello world | 编译通过 + 跑通 | CC |
| 更新 DESIGN.md | 补充 API 细节和伪代码 | OpenClaw |

---

## 阶段二：FP16 cuFFT → PyTorch 封装（6/6 - 6/15）

| Sprint | 任务 | 验收标准 |
|---|---|---|
| **Sprint 1** (6/6-6/8) | PyTorch C++ 扩展调用 cuFFT FP16 | `torch.fft.fft(x, precision="fp16")` 能跑 |
| **Sprint 2** (6/9-6/11) | 实现 backward（自动微分） | gradcheck 通过 |
| **Sprint 3** (6/12-6/13) | 精度基准：FP16 vs FP32 误差测量 | 误差 < 1e-3 |
| **Sprint 4** (6/14-6/15) | 性能基准：吞吐量对比 | 相比 FP32 提升 ≥ 1.5× |

---

## 阶段三：FP8 自研 kernel（6/16 - 6/30，创新点）

| 任务 | 产出 |
|---|---|
| FP8 误差模型分析（蝶形运算的误差传播） | 理论分析文档 |
| 块浮点数方案实现 | CPU Python 原型 |
| CUDA FP8 FFT kernel v0 | 编译通过 + 基础精度测试 |
| 精度-性能 trade-off 调优 | 实验数据矩阵 |
| 与 FP16/FP32 基准对比 | 综合评测报告 |

---

## 阶段四：补充 + 社区准备（7/1 - 7/15）

| 任务 | 产出 |
|---|---|
| BF16 cuFFT 封装 | 复用 FP16 框架 |
| CPU SIMD 第一版（x86 AVX2） | 至少 FP16 支持 |
| 单元测试全覆盖 | 精度 + 性能 + 多架构 |
| PyTorch PR 草稿 | 代码 + 文档 + 基准测试结果 |

---

## 协作协议

- **韩志麟 → OpenClaw**（飞书）：说「做 xxx」→ 拆解委派
- **OpenClaw → CC**（HANDSHAKE + wake-laptop）：每次 1 个任务
- **CC → OpenClaw**（git push → LAPTOP-CHANGES.md）：完成回报

---

*版本: v3 | 更新: 2026-06-02 | 下一步: 阶段一 — cuFFT API 研读*
