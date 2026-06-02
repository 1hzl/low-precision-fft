# Low Precision FFT — 项目规则

## 🔴 硬约束

- 所有 CUDA kernel 必须有 CPU fallback，`#ifdef __CUDA_ARCH__` 保护
- 任何 PR 必须先通过 `nvcc` 编译 + `ctest` 全绿
- 禁止提交编译后的 `.ptx` / `.cubin` / `.so` 二进制
- PyTorch ATen 封装遵循 `torch/utils/cpp_extension.py` 标准

## 🟡 软约束

- CUDA kernel 行数 > 100 需有注释说明 block/thread 策略
- 基准测试数据保存到 `data/` 目录，命名含日期-精度-尺寸
- 每个新 kernel 附带最小可复现测试

## ⚪ 风格建议

- Python: PEP 8 / `black` formatting
- CUDA: NVIDIA 官方风格指南，变量名用 `snake_case`
- C++ ATen: 跟随 PyTorch 源码风格 (clang-format)
- 文档用英文，注释可中英混合

## 🔴 硬件约束

- 开发 GPU：RTX 5070 Ti (12GB VRAM)
- VRAM 使用超过 10GB 时必须主动汇报
- 任何需要多 GPU / 更大显存的实验 → 先汇报，不硬撑
- Han 可提供云 GPU 或购买新设备，需要时立即提出

## 规则进化

- 每次 push → auto-review 可能产出新规则
- 新规则追加到本文末，标注日期和触发来源
- 2026-06-02: 硬件资源策略（Han 决策）
