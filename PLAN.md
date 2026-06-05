# 低精度 FFT — 实施计划 v4

> **对齐**: `资料/任务书.docx`（项目编号 20261124）  
> **团队**: 韩志麟（统筹） + 詹世显 + 刘子渊 | 导师: 廖思宇  
> **AI 辅助**: OpenClaw（N2920, 研究/拆解/协调） + Claude Code（笔记本, 编码/实验）  
> **更新**: 2026-06-05 | **当前进度**: ~40%（算法 100%, 工程+社区 ~10%）

---

## 已完成 ✅

### 阶段一：方案定稿 + 环境验证（6/2–6/5）✅

| 任务 | 产出 |
|---|---|
| cuFFT FP16 API 研读 | `paper-notes/cufft-fp16-api.md` |
| PyTorch `torch.fft` 源码走读 | `paper-notes/pytorch-cufft-integration.md` |
| CUDA 环境确认 + hello world | 编译通过, RTX 5070 Ti |
| DESIGN.md | 架构设计文档 |

### 阶段二：FP16/BF16 cuFFT → PyTorch 封装（6/3–6/5）✅

| 产出 | 验收 |
|---|---|
| C++ 扩展 + cuFFT FP16/BF16 | `lowp_fft.fft(x, precision="fp16")` 可用 |
| backward 自动微分 | gradcheck 通过 |
| 精度基准 | FP16 56–61 dB, BF16 53.1 dB |
| 性能基准 | FP16 1.4–2.8× vs FP32 |
| 审查修复 | 4 MAJOR + 3 MINOR + 2 bugs → 31/31 tests |

### 阶段三：FP8 BFP 自研 kernel（6/4）✅

| 产出 | 验收 |
|---|---|
| 论文研读 — arXiv 2605.28451 BFP FFT | 分析 + 复现 |
| FP8 误差模型 + 仿真 | `docs/fp8-fft-error-model.md` |
| 三种方案对比 → 推荐 BFP | `docs/fp8-strategy-comparison.md` |
| BFP Python 原型 | `lowp_fft/bfp_fft.py` |
| BFP CUDA kernel v0 | `src/cuda/bfp_fft.cu` (846 行) |
| 精度-性能调优 + 最终报告 | `docs/sprint-3.4-final-report.md` |
| Bergach 2026 复现 | 4 信号 × 5N, FP64 基准对齐 |

### 阶段四：收尾 + 补全（6/5）✅

| 产出 | 验收 |
|---|---|
| BFP API 接入 | `precision="fp8"` 可调用 |
| BFP CUDA inverse | roundtrip 验证 19.5 dB |
| 内存带宽 benchmark | 大 batch × 大 N |
| 边界测试 | 22/22 全部通过 |
| README | 精度表 + 安装 + 示例 |

---

## 未完成 — 对齐原始任务书

> 以下各阶段对应 `资料/任务书.docx` 的原始计划。已完成内容已跳过。

---

### 阶段五：CPU SIMD 多架构适配 ⬜ **P0**

> 📄 对应任务书 **二-3**「CPU端SIMD多架构适配」+ 预期成果「GPU+CPU多架构可复用代码」

**前置条件**: 多架构 CPU 测试环境（任务书一-3）

| # | 任务 | 产出 | 负责 |
|---|---|---|---|
| 5.1 | x86 AVX2/AVX-512 CPU FFT（FP16） | C 扩展, SIMD intrinsics | CC |
| 5.2 | ARM NEON CPU FFT（FP16） | 交叉编译验证 | CC |
| 5.3 | CPU 端接入 `lowp_fft` API | `device="cpu"` + `precision="fp16"` | CC |
| 5.4 | CPU BFP FP8 原型 | 复用 GPU BFP 逻辑 | CC |
| 5.5 | CPU vs GPU 精度/性能对比 | 基准报告 | CC |

**验收标准**:
- `lowp_fft.fft(x.cpu(), precision="fp16")` 可用
- x86 + ARM 双架构编译通过
- 精度与 GPU 版本一致

**硬件需求**: N2920 为 x86_64 (无 AVX-512)，需确认 ARM 交叉编译环境或租用 ARM 云实例

---

### 阶段六：LLM 微调场景验证 ⬜ **P1**

> 📄 对应任务书 **三-1**「LLM微调场景测试集 + 验证报告」

| # | 任务 | 产出 |
|---|---|---|
| 6.1 | 选型：LLM 微调中 FFT 的使用路径 | 文献/代码调研 |
| 6.2 | 构建测试集（典型 FFT 尺寸 + batch） | benchmark 脚本 |
| 6.3 | 精度对比：低精度 FFT vs FP32 在训练中的梯度收敛性 | 实验数据 |
| 6.4 | 性能对比：吞吐量 + 显存占用 | 基准报告 |
| 6.5 | 场景化应用验证报告 | `docs/llm-validation-report.md` |

---

### 阶段七：社区贡献准备 ⬜ **P2**

> 📄 对应任务书 **一-2**「PyTorch社区规范研究」+ **三-2**「文档完善」+ **三-3**「Initial PR」

| # | 任务 | 产出 |
|---|---|---|
| 7.1 | PyTorch Contribution Guide 研读 | 贡献流程笔记 |
| 7.2 | Usage Guide（用户文档） | `docs/usage-guide.md` |
| 7.3 | Design Doc（开发者文档） | 更新 `DESIGN.md` |
| 7.4 | 测试补全（多架构 + 回归测试） | CI-ready test suite |
| 7.5 | PR 草稿（代码 + 文档 + 基准） | PyTorch fork + PR description |
| 7.6 | PyTorch Discuss 论坛发起讨论 | 社区反馈收集 |

---

### 阶段八：迭代与结题 ⬜ **远期**

> 📄 对应任务书 **四**「迭代优化与结题」

| # | 任务 |
|---|---|
| 8.1 | 社区反馈迭代修复 |
| 8.2 | 代码合并至 PyTorch 主干 |
| 8.3 | 项目总结报告 + 结题答辩材料 |

---

## 预期成果 vs 现状

| 任务书承诺 | 现状 |
|---|---|
| FP16/FP8 低精度 FFT 算法模型 | ✅ 已完成 |
| GPU + CPU 多架构可复用代码 | ❌ 仅 GPU |
| PyTorch 社区贡献 + 通过评审 | ❌ 未启动 |
| LLM 微调场景验证报告 | ❌ 未启动 |

---

## 协作协议

- **韩志麟 → OpenClaw**（飞书）：说「做 xxx」→ 拆解委派
- **OpenClaw → CC**（HANDSHAKE + wake-laptop）：每次 1 个任务（Ivy Lee 规则）
- **CC → OpenClaw**（git push → LAPTOP-CHANGES.md）：完成回报

---

*版本: v4 | 更新: 2026-06-05 | 对齐: 任务书 20261124 | 下一步: 阶段五 — CPU SIMD*
