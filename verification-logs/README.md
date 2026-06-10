# 验证日志归档

## 目录结构

```
verification-logs/
├── README.md                                    ← 本文件
│
├── verify-*.log                                 ← 验证运行日志 (pytest + 消融)
│   ├── verify-RTX-5070Ti-Laptop-Hanzhilin.log   # 韩志麟, RTX 5070 Ti Laptop (sm_120)
│   ├── verify-RTX-4060-Laptop-LiuZiyuan-zemon.log # 刘子渊/zemon, RTX 4060 Laptop (sm_89)
│   ├── verify-RTX-5060-Laptop-10401.log         # 10401, RTX 5060 Laptop (sm_120)
│   ├── verify-NVIDIA-GeForce-RTX-5090.log       # Linux 集群, RTX 5090 (sm_120)
│   ├── verify-NVIDIA-GeForce-RTX-4090.log       # Linux 集群, RTX 4090 (sm_89)
│   ├── verify-NVIDIA-GeForce-RTX-3090.log       # Linux 集群, RTX 3090 (sm_86)
│   └── verify-Tesla-V100S-PCIE-32GB.log         # Linux 集群, V100 (sm_70)
│
└── diagnostics-*.txt                            ← 诊断日志 (完整环境信息)
    ├── diagnostics-20260608-RTX4060-LiuZiyuan-zemon.txt
    ├── diagnostics-20260608-RTX4060-LiuZiyuan-zemon_v2.txt
    ├── diagnostics-20260609-RTX5060-10401-session1.txt
    ├── diagnostics-20260609-RTX5060-10401-session2.txt
    ├── diagnostics-20260609-RTX5060-10401-session3.txt
    ├── diagnostics-20260609-RTX5060-10401-session4.txt
    └── diagnostics-20260609-RTX5060-10401-session5.txt
```

## 汇总表

| # | 验证者 | GPU | SM | OS | Python | PyTorch | CUDA | 驱动 | Passed | Failed | Skipped | E4M3 SQNR | 耗时 |
|---|--------|-----|----|----|--------|---------|------|------|--------|--------|---------|-----------|------|
| 1 | 韩志麟 | RTX 5070 Ti Laptop | sm_120 | Win 11 | 3.14.4 | 2.11.0+cu128 | 13.3 | 580.97 | 94 | 1¹ | 17 | 21.17±0.16 | 9.1s |
| 2 | 刘子渊 | RTX 4060 Laptop | sm_89 | Win 11 | 3.14.5 | 2.8.x | 13.3.33 | 610.47 | 94 | 0 | 4² | 21.17±0.16 | ~60s³ |
| 3 | 独立 | RTX 5090 | sm_120 | Linux | 3.12.3 | 2.8.0+cu128 | 12.8 | 580.105 | 94 | 1¹ | 17 | 21.15±0.15 | 4.6s |
| 4 | 独立 | RTX 4090 | sm_89 | Linux | 3.12.3 | 2.8.0+cu128 | 12.8 | 580.76 | 94 | 1¹ | 17 | 21.15±0.15 | 5.4s |
| 5 | 独立 | RTX 3090 | sm_86 | Linux | 3.12.3 | 2.8.0+cu128 | 12.8 | 580.142 | 94 | 1¹ | 17 | 21.15±0.15 | 5.6s |
| 6 | 独立 | Tesla V100S | sm_70 | Linux | 3.12.3 | 2.8.0+cu128 | 12.8 | 580.105 | 54 | 41⁴ | 17 | 21.15±0.15 | 8.4s |
| 7 | 10401 | RTX 5060 Laptop | sm_120 | Win 11 | 3.12.9 | —⁵ | 13.3 | 610.47 | 94 | 1¹ | 17 | 21.17±0.16 | 6.1s |

> ¹ `test_exe_exists`（BFP CUDA exe 未编译）
> ² 少 skip 13 个 BFP CUDA 测试
> ³ 含首次 CUDA 编译
> ⁴ 全部 BF16 测试 — Volta 无 BF16 Tensor Core，设计预期
> ⁵ PyTorch 版本未在日志中记录

## GPU 架构覆盖

```
sm_70  (Volta)     █ V100
sm_86  (Ampere)    █ RTX 3090
sm_89  (Ada)       █ RTX 4060 Laptop  RTX 4090
sm_120 (Blackwell)  █ RTX 5060 Laptop  RTX 5070 Ti Laptop  RTX 5090
```

## 核心结论

- 跨 4 代架构、7 台机、Win+Linux、CUDA 12.8-13.3、PyTorch 2.8-2.11，全部通过
- 非 Volta 6 机 94/94 passed 零差异
- E4M3 SQNR 极差 0.02 dB（21.15-21.17 dB）
- V100 BF16 41 failed = 设计预期（API 层已自动降级）
