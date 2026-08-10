# TokenScaleFP4

[![CI](https://github.com/naturaljam/TokenScaleFP4/actions/workflows/ci.yml/badge.svg)](https://github.com/naturaljam/TokenScaleFP4/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-2F8552.svg)](LICENSE)

[![English](https://img.shields.io/badge/README-English-111827?style=for-the-badge)](README.md)
[![Chinese](https://img.shields.io/badge/README-中文-0f766e?style=for-the-badge)](README.zh-CN.md)

**面向 NVIDIA Blackwell SM120 的 NVFP4 推理实验项目，重点解决 per-token 激活缩放问题。**

TokenScaleFP4 从量化规则和参考实现做起，目标是在 FlashInfer 中实现 fused GEMM，并进一步接入 vLLM 的在线 dense quantization。项目要解决的问题很具体：per-token 激活量化会为每一行产生一个 scale，GEMM 需要把它应用到对应的输出行，同时避免再启动一个单独的 row-scale kernel。

## 要解决的问题

动态 NVFP4 激活量化会为每个逻辑输入行生成一个 scale。FlashInfer 现有 `mm_fp4` 接口中的 `alpha` 表示全局权重 scale，不能同时承载一个长度为 `M` 的激活 scale。

目标计算如下：

```text
Y[m, n] = BF16(
    per_token_scale[m] * alpha * block_scaled_fp4_dot(A[m, :], W[n, :])
)
```

TokenScaleFP4 先固定这套计算规则，再用反量化后的 FP32 结果做对照，并补齐 API 校验和测试，为后续把 `per_token_scale[m]` 融合进 SM120 b12x GEMM epilogue 做准备。

## 目标链路

```text
BF16 activation [M, K]
  -> per-token NVFP4 quantization
  -> packed activation + block scales + row scale [M]
  -> FlashInfer b12x FP4 GEMM
  -> fused alpha * row_scale[m]
  -> BF16 output [M, N]
  -> vLLM dense linear dispatch
```

当前 oracle 会在 `mm_fp4` 之后单独做 row scaling。目标实现会把这一步放进 GEMM epilogue，同时保持原有 scalar 路径、预分配输出、编译缓存和 CUDA Graph replay 行为不变。

## 已完成

| 模块 | 当前内容 |
| --- | --- |
| Contract | SM120 NVFP4 shape、block size、BF16 输出和 FP32 row-scale 校验 |
| Reference | E2M1 解包、128x4 E4M3 scale 解码、FP32 GEMM 和 BF16 输出 |
| Oracle | per-token 激活量化，以及 unfused `mm_fp4 + row scale` 路径 |
| Kernel lab | FlashInfer b12x dispatch、Qwen shape 处理和固定的 upstream revision |
| Evidence | 数值指标、质量/显存门槛、版本记录和 evidence 数据校验 |
| Tests | 167 个可移植测试，以及需要 SM120 的 smoke/oracle 测试 |

fused row-scale epilogue、完整的 vLLM dense online path 和租用 GPU 上的最终性能数据仍在推进中，README 不把这些内容写成已经交付的能力。

## 当前范围

- NVIDIA Blackwell **SM120**
- NVFP4，group size 16，128x4 scale storage
- BF16 activation 和 output
- FlashInfer b12x backend
- TP1 dense linear layer
- 以 Qwen2.5 projection shape 作为主要验证集合

SM100/SM103、MoE、FP8、训练、KV cache 和多 GPU 不在当前范围内。在完成模型级测量之前，项目不宣称相对 BF16 有端到端加速。

## 快速开始

普通开发环境需要 Python 3.11-3.13。运行 SM120 测试还需要 Linux 或 WSL2、CUDA 13+、SM120 GPU，以及固定版本的 FlashInfer checkout。

```bash
git clone https://github.com/naturaljam/TokenScaleFP4.git
cd TokenScaleFP4
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:    .venv\\Scripts\\Activate.ps1
python -m pip install -e ".[dev]"

python -m pytest -m "not sm120" -q
python -m ruff check .
python -m pyright src
python scripts/audit_public_repo.py --tracked
```

## 仓库结构

```text
src/tokenscalefp4/contracts/      输入和 shape 约束
src/tokenscalefp4/reference/      NVFP4 反量化与 FP32 oracle
src/tokenscalefp4/kernel_lab/     FlashInfer dispatch 与 unfused 路径
src/tokenscalefp4/reporting/      evidence 数据结构与校验
tests/                            可移植测试和 SM120 测试
scripts/                          环境、评测、审计和 smoke 工具
configs/                          固定的 shape 与质量评测配置
upstream/                         FlashInfer 和 vLLM 的 base revision
```

## 验证方式

项目使用固定门槛，不靠单个样例下结论。reference 和 oracle 测试覆盖 row-scale 的 shape 与 layout、scalar 路径兼容性、BF16 输出、数值误差、非法输入、预分配输出和 Qwen 常见维度。硬件结果与普通 CI 分开，公开 CI 通过不等于 SM120 性能已经达标。

## 许可证

TokenScaleFP4 使用 [Apache License 2.0](LICENSE) 发布。
