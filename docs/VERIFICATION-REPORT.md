# 独立性验证报告

项目：low-precision-fft (低精度 FFT for PyTorch)
基准提交：0325b7a "docs: update VERIFY.md"

---

## 验证矩阵

| # | 日期 | 验证者 | GPU | OS | Python | CUDA | 路径A | 路径B | SQNR | 状态 |
|---|------|--------|-----|----|--------|------|-------|-------|------|:--:|
| 1 | 2026-06-08 | 韩志麟 | RTX 5070 Ti | Win11 | 3.14.4 | 13.3 | — | 94 passed, 17 skipped, 2 xfailed | — | ✅ |
| 2 | 2026-06-08 | 刘子渊 | RTX 4060 Laptop | Win11 | 3.14.5 | 13.3.33 | BFP 22 + API 72 passed, 4 skipped, 2 xfailed | 94 passed, 17 skipped, 2 xfailed | 21.17 ± 0.16 dB | ✅ |

## 通过标准

| 检查项 | 阈值 | 说明 |
|--------|------|------|
| pip install | 成功 | 有 CUDA 环境需正常编译 cuFFT 扩展 |
| import | 成功 | `import lowp_fft` + `_cufft_ext` 加载 |
| 路径A passed | ≥ 94 | BFP 22 + API fallback 72 |
| 路径B passed | ≥ 94 | 全量 cuFFT FP16/BF16 + BFP + autograd |
| 消融 SQNR | 21.2 ± 0.5 dB | E4M3 uniform 模式 |
| CUDA 版本适配 | 无崩溃 | 12.8(驱动) ~ 13.3(Toolkit) 均通过 |

## 已知异常项（非缺陷）

- **skipped 17**: gradcheck 受 IEEE 754 半精度物理精度限制（4）+ BFP CUDA 独立 exe 未编译（13）
- **xfailed 2**: PlanCache race condition（正常使用不触发）
- **FAILED 1**: test_bfp_cuda exe_exists（需要单独跑 build_bfp.bat，与 pip 安装的 PyTorch 扩展无关）

## 已验证的平台兼容性

- ✅ Windows 11 + RTX 5070 Ti + CUDA 13.3
- ✅ Windows 11 + RTX 4060 Laptop + CUDA 13.3.33（更新驱动至 610.47）
- ⚠️ 已知：CUDA 13.3 Toolkit + CUDA 12.9 驱动 → cuFFT error 16（驱动需 ≥13.0）
