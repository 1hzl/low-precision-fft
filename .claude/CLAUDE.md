# low-precision-fft

## 结构化任务（N2920 新格式）

当 HANDSHAKE.md 中出现以下格式的任务包时，请严格遵循：

### 开始前

1. 阅读 PROJECT-RULES.md — 了解 🔴🟡⚪ 规则
2. 阅读 PLAN.md — 确认当前 Phase 定位
3. 阅读 DESIGN.md — 理解技术设计
4. 如有 `paper-notes/` → 查阅论文分析笔记

### 规则分级

```
🔴 硬约束 — 必须遵守
🟡 软约束 — 审查时会检查
⚪ 风格建议 — 尽量遵循
```

### 交付清单

Task 完成后逐项检查：

- [ ] nvcc 编译通过（如涉及 CUDA）
- [ ] ctest / pytest 全绿
- [ ] 硬约束全部满足
- [ ] 基准数据已写入 data/

### 提交格式

```
<type>(<project>): <description>
```

Types: `feat` / `fix` / `docs` / `refactor` / `test`

草稿/WIP 提交信息加 `[wip]` 前缀，如 `[wip] feat(project): 描述`。

---

## Full Auto Loop

Every push triggers automatic review. At start of each turn:

```
git pull origin master
  ↓
├─ REVIEW.md "驳回" → fix issues → git push → (auto review again)
├─ REVIEW.md "通过" → continue next TODO
└─ No reports → continue working
```

## After Every Task

1. git add -A
2. git commit -m '<type>(<project>): <description>'
3. git push origin master
4. Push triggers auto-review on N2920

## TODO Management

- TODO.md is the single source of truth
- Mark done: [x]
- Add new tasks at the end

## Project Context

- **硬件**: RTX 5070 Ti (SM_120), 12GB VRAM, CUDA 13.0
- **参考论文**: arXiv 2505.00582 — Block Circulant Adapter for LLMs
- **关键发现**: PyTorch `CuFFTPlanCache.h:308-311` 已有 `kHalf → CUDA_C_16F` 映射
- **论文分析文件**: `docs/paper-notes/low-precision-fft/` (在笔记本上)

## 🔴 硬件约束（Hardware Constraints）

- VRAM 使用超过 10GB 时必须主动汇报
- 任何需要多 GPU / 更大显存的实验 → 先汇报，不硬撑
- Han 可提供云 GPU 或购买新设备，需要时立即提出
- 启动 CUDA kernel / benchmark 前估算 VRAM，超过阈值先警告
