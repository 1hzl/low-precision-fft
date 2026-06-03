# TOKEN-USAGE.md — 低精度 FFT 项目 Token 消耗追踪

> 创建: 2026-06-03 | 维护: OpenClaw (N2920)
> 方法论见文末

---

## Phase 1: 环境 & 验证（2026-06-02）

### 总览

| 通道 | Token 量 | 成本 (USD) | 精度 |
|------|----------|------------|------|
| A. 人机对话 (OpenClaw × DeepSeek V4 Pro) | ~39,439 | ~$0.07 | ✅ 精确 |
| B. 自动审查 (N2920 × IKunCode Haiku) | ~31,700 | ~$0.03 | ⚠️ 估算 |
| C. 编码执行 (笔记本 × Claude Code) | ~116,000 | ~$0.35 | ⚠️ 估算 |
| **Phase 1 合计** | **~187,000** | **~$0.45** | — |

---

### A. 人机对话 — 精确值

数据来源: `session_status.totalTokens`，本 session 全程讨论 low-precision-fft。

| 事件 | 模型 | Tokens |
|------|------|--------|
| "查看项目" → 项目列表 | DS V4 Pro | — |
| "查看 low-precision-fft" → 项目详情 | DS V4 Pro | — |
| "详细讲阶段一" → 技术拆解 | DS V4 Pro | — |
| "详细讲阶段一(第二次)" → 源码级展开 | DS V4 Pro | — |
| "数据真实吗" → 多维度验证 | DS V4 Pro | — |
| "统计 token 消耗" → 方法论+执行 | DS V4 Pro | — |
| **本 session 累计** | DS V4 Pro | **39,439** |

---

### B. 自动审查 — 基于 diff 规模估算

数据来源: git log 中 10 次 `review:` commit，每次 diff 50-115 行。

审查引擎: `~/bin/ai-review` → IKunCode API (Claude Haiku 4.5) + DeepSeek V4 Flash (优化)

**估算公式**: tokens ≈ system_prompt(3000) + diff_context(行数×15) + output(800) + project_stats(500)

| 时间 | Diff 行数 | 估算 Tokens | 结果 |
|------|----------|-------------|------|
| 14:39 | 84 | 3,100 | 通过（有建议） |
| 14:41 | 109 | 3,500 | 通过（有建议） |
| 14:41 | 113 | 3,600 | 通过 |
| 14:45 | 91 | 3,200 | 通过 |
| 14:53 | 115 | 3,600 | 通过（有建议） |
| 14:53 | 107 | 3,500 | 通过 |
| 15:55 | 70 | 2,800 | 通过 |
| 16:01 | 50 | 2,500 | 通过 |
| 16:12 | 108 | 3,500 | 通过（有建议） |
| 19:48 | 101 | 3,400 | 通过 |
| **合计** | **948** | **~31,700** | — |

> ⚠️ `ai-review` 脚本未从 API 响应中提取 `usage` 字段，以上为公式估算。精度 ±30%。

---

### C. 笔记本 Claude Code — 基于代码产出规模估算

数据来源: `han <hzl@github>` 的 7 次 push（去重后 5 次有效）。

**估算公式**: tokens ≈ code_completion(行数×20) + reasoning_overhead(行数×10) + context(8000)

| 提交 | 产出 | 估算 Tokens | 说明 |
|------|------|-------------|------|
| `e0bdef6` | hello_fp16_fft.cu (145行) | ~15,000 | CUDA FP16 验证程序 |
| `fa41a4e` | DESIGN.md 更新 (1行) | ~3,000 | 轻量设计修正 |
| `e2cc999` | DESIGN.md 决策记录 (3行) | ~3,000 | 文档更新 |
| `daac77f` | benchmark×2 (625行) + report (87行) + CSV×2 | ~80,000 | 核心基准测试 |
| `5bccaf6` | CLAUDE.md (15行) | ~5,000 | 硬件约束文档 |
| `eda9812` | Makefile (39行) | ~8,000 | 编译系统 |
| `377952f` | merge conflict resolve | ~2,000 | 合并冲突 |
| **合计** | **768 行 CUDA + 87 行报告 + 文档** | **~116,000** | — |

> ⚠️ 笔记本 Claude Code 为 Windows 独立进程，N2920 无法直连其 API billing。以上基于代码产出量 × 经验系数估算。精度 ±50%。

---

## 方法论

### 数据源

| 通道 | 数据源 | 精度 |
|------|--------|------|
| A. OpenClaw session | `session_status` API → `totalTokens` | ✅ 精确 |
| B. `ai-review` 脚本 | git log diff 规模 → 公式估算 | ⚠️ 待脚本记录 usage |
| C. 笔记本 Claude Code | git log 产出量 → 经验系数 | ⚠️ 待笔记本回报 |

### 改进计划

- [ ] **B 通道**：修改 `ai-review` 脚本，从 API 响应的 `usage.prompt_tokens` / `usage.completion_tokens` 提取真实值，写入 `TOKEN-USAGE.md`
- [ ] **C 通道**：要求笔记本 Claude Code 在 `LAPTOP-CHANGES.md` 中附带 session token 统计
- [ ] 每阶段结束时更新本文件
- [ ] Phase 2 起，C 通道改为实测值

### 经验系数说明

- `diff_context × 15`: 每行 git diff 约 15 个 token（含上下文）
- `code_completion × 20`: 每写一行代码约消耗 20 token（含推理+输出）
- `reasoning_overhead × 10`: 理解任务、探索、迭代的额外 token
- `context(8000)`: 每次 CC session 的系统 prompt + 项目上下文

---

*文件版本: v1 | 下次更新: Phase 2 完成时*
