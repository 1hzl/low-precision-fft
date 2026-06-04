# FP8 FFT 三种候选方案对比分析

> Phase 3 Sprint 3.1d — 方案可行性评估 + 推荐方案（含论文验证）
> 前置: `docs/fp8-e4m3-basics.md`, `docs/fp8-fft-error-model.md`, `paper-notes/2605.28451-analysis.md`
> 更新: 2026-06-04 — 整合 arXiv 2605.28451 (Bergach 2026) 的实证结论

---

## 概述

cuFFT 不支持 FP8，需要自研 CUDA kernel。根据 FFT 蝶形运算的结构特性和 FP8 E4M3 的表示能力（max=448, mantissa=3bit），评估三种候选方案。

**关键背景** (来自 Bergach 2026):
- FP16 BFP FFT 在 Apple M1 上达到 **56-61 dB SQNR**，2.2× 加速
- FP8 崩塌到 **14-20 dB SQNR**，无法实用
- Fixed-shift BFP (1/N 缩放) 仅需 2 行代码，无需逐层动态指数
- 论文已验证: FFT 的瓶颈是**动态范围**（指数位数），而非**尾数精度**

---

## 论文验证 (Bergach 2026)

| 论文结论 | 我们的方案 | 一致性 |
|---------|-----------|--------|
| FP16 BFP: 56-61 dB SQNR, 2.2× 加速 | 方案A (BFP): 理论预测 SNR ~8-19 dB for FP8 | ✅ 方向一致: BFP 是正确的 |
| Fixed-shift 1/N 缩放足够 | 方案A: 曾考虑逐层动态指数 | ⚠️ 可简化: 论文证明单次缩放足够 |
| FP8 崩塌到 14-20 dB | 方案A-B: FP8 目标 | ⚠️ FP8 实用性受限 |
| 瓶颈是范围而非精度 | 方案A: 共享指数解决范围问题 | ✅ 理论一致 |
| Apple M1 2.2× 加速 | 方案A-C: GPU 加速预期 | ❓ 需 NVIDIA 实测 |

## 详细对比

### 方案 A: 块浮点数 (Block Floating-Point, BFP)

**原理**: 每层蝶形输出的 N 个值共享一个指数（scale factor），尾数以 FP8 存储。

```
// 第 s 层蝶形
shared_exp[s] = ceil(log2(max(|vals|))) + margin
for each butterfly output x:
    fp8_mantissa = quantize_fp8(x / 2^shared_exp[s])
```

**数据结构**:
```
┌──────────────────────────────────────┐
│ float8[N]    mantissa (FP8 E4M3)    │
│ int8[log₂N]  shared_exponents       │
│ int8         global_offset (opt)    │
└──────────────────────────────────────┘
存储: N + log₂(N) 字节 (vs FP32 的 4N 字节)
```

| 维度 | 评价 |
|------|------|
| **精度** | ⭐⭐⭐⭐⭐ 每层仅 1× 舍入，log₂(N) 次总舍入，理论 SNR 随 N 缓慢下降 |
| **性能** | ⭐⭐⭐⭐ 指数计算需要 reduction（max per stage），O(N) 开销，相对计算可忽略 |
| **实现复杂度** | ⭐⭐⭐ 需要每层蝶形后做 max reduction + scale + quantize，但 CUDA warp-level reduction 成熟 |
| **内存** | ⭐⭐⭐⭐⭐ ~2×(N + log₂N) 字节（复数），比 FP32 的 8N 节省 ~4× |
| **理论基础** | ⭐⭐⭐⭐⭐ 有成熟文献支持（Oppenheim & Weinstein 1972, Welch 1969） |

**优点**:
- FFT 蝶形结构天然适配：每层频率幅值增长规律，值域范围可预测
- 指数开销极小：log₂(N) 个指数管理 N 个浮点值
- 有现成的理论误差界（见 §2.4 误差分析）

**缺点**:
- 需要每层蝶形后的全局 max reduction（需要同步点）
- 指数必须是 2 的幂次（方便硬件）或任意值（更高精度）

**实现要点**:
```cuda
// 每层蝶形后的 block-float 重缩放
float max_val = block_reduce_max(abs_vals);  // 1 次 warp/block reduction
int shared_exp = (int)ceilf(log2f(max_val + 1e-10f));
float scale = ldexpf(1.0f, -shared_exp);
for (int i = 0; i < N; i++) {
    fp8_out[i] = float_to_fp8(vals[i] * scale);
}
```

---

### 方案 B: 动态缩放 (Dynamic Scaling)

**原理**: 每层蝶形运算后对每个值独立做 scale+clamp 到 FP8 范围，不共享指数。

```
for each butterfly output x:
    if |x| > FP8_MAX:
        x = FP8_MAX * sign(x)  // clamp
    else:
        x = quantize_fp8(x)    // quantize
```

| 维度 | 评价 |
|------|------|
| **精度** | ⭐⭐ 溢出点被硬截断，信息损失不可逆；小值精度浪费（不需要的量程留着） |
| **性能** | ⭐⭐⭐ 无需 reduction，但每个值需要 scale+clamp，指令数多 |
| **实现复杂度** | ⭐⭐⭐⭐⭐ 最简单——无全局同步，无额外状态 |
| **内存** | ⭐⭐⭐ N 字节（纯 FP8），最紧凑 |
| **理论基础** | ⭐⭐ 无专门文献支持 FFT 场景 |

**优点**:
- 实现极其简单：if-clamp，无全局同步
- 纯 FP8 内存布局，无辅助数据
- 适合流式/实时处理场景

**缺点**:
- **精度不可控**: clamp 损失的能量在后续层传播
- 小值浪费精度：归一化后期层信号幅值差异大（DC bin vs 高频 bin 可能差 10-100×），统一 clamp 阈值导致小 bin 精度浪费
- 没有误差界——精度依赖于信号分布，难以预测
- FFT 后期层的值域范围大 -> clamp 频率高 -> 精度崩溃

**定量分析**: 对于 N=1024 的随机信号，第 9 层最大值 ~27，最小值 ~1e-3。如果 clamp 阈值 = 448，小值仅使用尾数 3 位中的 ~0 位（因为指数很大但尾数小而无法表示）——实际上小值会被量化到 0。

**为什么不适合 FFT**: FFT 每层输出值的动态范围远大于 FP8 的表示能力。clamp 只能保护大值不溢出，但小值在后期层会因指数不足而 underflow（FP8 的 min normal = 0.0156，而后期层很多 bin < 0.01）。

---

### 方案 C: 混合精度 (Mixed Precision)

**原理**: 前 k 层用 FP16 cuFFT，后 (log₂(N)−k) 层用 FP8 自研 kernel。

```
Layer 0..k-1: cuFFT FP16 (高精度)
Layer k..log₂(N)-1: 自研 FP8 kernel
```

| 维度 | 评价 |
|------|------|
| **精度** | ⭐⭐⭐⭐ 关键前期层保精度，后期层用 FP8 加速 |
| **性能** | ⭐⭐ 跨精度边界需要格式转换（FP16 ↔ FP8），开销大 |
| **实现复杂度** | ⭐ 需要两个不同的 FFT 后端协调，边界对齐复杂 |
| **内存** | ⭐⭐ 前期层用 FP16 (2× 内存)，后期用 FP8 (1×) |
| **理论基础** | ⭐⭐⭐ 混合精度在 DL 训练中成熟（NVIDIA APEX, Transformer Engine） |

**优点**:
- 前期层（值域小但级联放大）用 FP16 保证精度
- 后期层（值域已展开）切换到 FP8 加速

**缺点**:
- **格式转换开销**: FP16 ↔ FP8 转换需要额外的 kernel launch（~10μs），可能在性能上得不偿失
- **切换点选择困难**: 最优 k 依赖 N、信号类型、精度要求——无通用最优解
- **实现复杂度翻倍**: 需要维护 FP16 和 FP8 两个 FFT 路径
- **cuFFT 耦合**: 方案依赖 cuFFT FP16，cuFFT 的 plan 创建开销 (~50-100μs) 影响小 FFT 场景
- **整体收益有限**: 前期层（值域小）的计算量也小（N/2 蝶形 vs 后期 N/2 蝶形），后期层切换 FP8 的加速效果被前期 FP16 拉平

**切换点分析**:
```
Stage 0-3 (size 1-8):   极小计算量，切换开销不划算
Stage 4-6 (size 16-64): 计算量中等，但值域已开始增长（2-9×）
Stage 7-9 (size 128+):  计算量大(>50% total)，值域大(>10×)，FP8 收益最高
```

推荐切换点（如果需要）：k = log₂(N) − 3，即最后 3 层用 FP8。但切换开销 (~10μs) 对 N < 8192 的影响较大。

---

## 综合评分矩阵

| 维度（权重） | A. 块浮点数 | B. 动态缩放 | C. 混合精度 |
|-------------|:----------:|:----------:|:----------:|
| 精度 (35%) | ⭐⭐⭐⭐⭐ 5 | ⭐⭐ 2 | ⭐⭐⭐⭐ 4 |
| 性能 (25%) | ⭐⭐⭐⭐ 4 | ⭐⭐⭐ 3 | ⭐⭐ 2 |
| 实现复杂度 (20%) | ⭐⭐⭐ 3 | ⭐⭐⭐⭐⭐ 5 | ⭐ 1 |
| 理论基础 (10%) | ⭐⭐⭐⭐⭐ 5 | ⭐⭐ 2 | ⭐⭐⭐ 3 |
| 内存效率 (10%) | ⭐⭐⭐⭐⭐ 5 | ⭐⭐⭐ 3 | ⭐⭐ 2 |
| **加权总分** | **4.35** | **2.85** | **2.55** |

---

## 推荐方案: A. 块浮点数

### 理由总结

1. **精度可预测**: 有闭合形式的误差界（Oppenheim & Weinstein 1972），SNR 随 N 增长缓慢
2. **FFT 结构完美适配**: 每层蝶形的值域范围相近（per-stage 分析已证实），使得共享指数几乎无精度损失
3. **实现可行**: CUDA warp/block level reduction 是成熟技术，max reduction 开销 < 1% 总计算量
4. **内存高效**: ~2N 字节存储 (FP8 mantissa × 2 for complex + log₂N bytes exponents)，相比 FP32 节省 ~3.5×
5. **仿真验证**: per-stage 值域分析证实范围增长平缓（~1.5×/stage），FP8 有充足余量

### 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 每层 max reduction 增加延迟 | 对大 N 的 latency 指标 | 用 warp-level reduction + shared memory，O(log₂(warp_size)) |
| 指数选为 2 的幂损失精度 | SNR 可能比理论差 1-2 dB | 允许非 2 的幂指数（乘法代替移位）提高 ~6% 精度 |
| FP8 硬件细节 (Blackwell SM_120) | 与 A100/H100 行为可能不同 | Phase 3.2 编写 CUDA 原型前先测 RTX 5070 Ti 的 FP8 实际精度 |
| 大 N (>16384) 时 SNR 降至 < 5 dB | 不适合高精度场景 | 文档明确标注精度边界，或提供 FP16 fallback |

### 实施路线

```
Phase 3.2: BFP FFT — CPU Python 原型
  ├─ 精确 BFP 模拟器（每层 scale）
  ├─ SNR vs N 数据矩阵
  └─ 验证误差 < 10% for N ≤ 4096

Phase 3.3: BFP FFT — CUDA kernel v0
  ├─ warp-level max reduction
  ├─ FP8 E4M3 量化器（CUDA intrinsic）
  └─ 与 FP32/FP16 cuFFT 对比

Phase 3.4: BFP FFT — 调优
  ├─ 非 2 的幂指数
  ├─ 共享内存优化
  └─ batch/stream 并行
```

---

## 为什么不选 B 或 C

### 方案 B（动态缩放）的根本问题

方案 B 等同于"在 FP8 上做无保护的 FFT"——我们已经在仿真中看到，朴素 FP8 在 N ≥ 256 时 SNR = 0 dB（完全崩溃）。clamp 无法拯救精度，因为：
- 它不是精度问题，是**表示能力不足**的问题
- FP8 的 14.8 位动态范围不足以覆盖 FFT 后期层的值域变化
- clamp 只能防止溢出，不能恢复已在前期层丢失的低位信息

### 方案 C（混合精度）的根本问题

跨精度 FFT 边界形成"精度断层"——前 FP16 层的量化噪声和相位误差被后 FP8 层放大，效果不如全 FP8 + 块浮点。且切换开销（格式转换 + 两个 plan）在 latency-sensitive 场景中不划算。

---

## 参考资料

- Oppenheim, A.V. & Weinstein, C.J. (1972). "Effects of finite register length in digital filtering and the fast Fourier transform." *Proc. IEEE*, 60(8), 957-976.
- Welch, P.D. (1969). "A fixed-point fast Fourier transform error analysis." *IEEE Trans. Audio Electroacoust.*, 17(2), 151-157.
- Constantinides, G.A. et al. (2004). "Numerical data representations for FPGA-based signal processing." *IEEE Trans. VLSI*.
- Micikevicius, P. et al. (2022). "FP8 Formats for Deep Learning." arXiv:2209.05433.
- Bergach, M.A. (2026). "Range, Not Precision: Block-Floating-Point Half-Precision FFT and SAR Imaging on Apple Silicon." arXiv:2605.28451.
- NVIDIA Blackwell Architecture Whitepaper (2024).

---

## Bottom Line (Post-Bergach 2026)

| Question | Answer |
|----------|--------|
| Is FP8 FFT practical? | **No** — 14-20 dB SQNR per Bergach 2026, confirmed by our simulation |
| Is BFP the right approach? | **Yes** — validated by paper (FP16 BFP: 56-61 dB) |
| Should we build FP8 BFP FFT? | **As research exploration only** — target: prove/disprove > 20 dB is possible |
| What's the practical recommendation? | **FP16 BFP FFT first** (proven), then FP8 BFP as research extension |
| What's simplest to implement? | **Fixed-shift BFP (1/N)** — paper proves single-scale works for convolution pipelines |
