# TokenScaleFP4

[![CI](https://github.com/naturaljam/TokenScaleFP4/actions/workflows/ci.yml/badge.svg)](https://github.com/naturaljam/TokenScaleFP4/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-2F8552.svg)](LICENSE)

[![English](https://img.shields.io/badge/README-English-111827?style=for-the-badge)](README.md)
[![中文](https://img.shields.io/badge/README-中文-0f766e?style=for-the-badge)](README.zh-CN.md)

**面向消费级 Blackwell SM120 GPU，让动态 per-token NVFP4 推理真正可验证、可复现的 AI Infra 项目。**

TokenScaleFP4 是我目前主推的开源项目，聚焦量化契约、GPU kernel dispatch 与可复现实验证据之间的工程边界。项目先建立小而严谨的基础设施，再逐步推进 fused kernel 和推理框架集成，不用未经验证的端到端加速数字包装成果。

## 项目成果

- 精确定义 NVFP4 契约，并提供 FP32/BF16 reference implementation。
- 提供 per-token row-scale 的 unfused oracle，用于验证后续 fused kernel。
- 提供面向 FlashInfer 的 SM120 shape、backend、patch 和 dispatch 辅助层。
- 提供公开的输入校验与 evidence-reporting schema，支持可重复实验。
- 提供 CI 覆盖的 Python package，并将硬件相关测试与可移植测试分开。

公开仓库只包含源码、测试、脚本和可复现所需的元数据。内部调研笔记、租用机器信息、原始 profiler 文件和模型产物均不进入仓库。

## 当前边界

当前公开基线是一个刻意收窄范围的实验性项目：

- 仅支持 NVIDIA Blackwell **SM120**
- NVFP4，group size 16，使用约定的 128x4 scale layout
- BF16 输出，面向 TP1 shape
- 当前优先完成 reference/oracle 与 API validation；生产级 fused kernel 和 vLLM 集成仍是后续工作

项目目前不宣称支持 SM100/SM103、MoE quantization、FP8、训练、KV cache、多 GPU 执行，也不承诺相对 BF16 必然获得端到端加速。

## 快速开始

基础环境要求：Python 3.11-3.13。GPU smoke test 还需要 SM120 GPU、CUDA 13+ 以及固定版本的 upstream 环境。

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

可移植测试覆盖 reference contract、shape 与 dispatch 逻辑、patch ownership、reporting schema 以及公开仓库审计。`sm120` 测试需要显式运行；缺少硬件或 Linux/CUDA 环境时会明确失败并退出。

## 仓库结构

```text
src/tokenscalefp4/       公开 package：contracts、reference、kernel lab、reporting
tests/                   可移植 contract 测试与 opt-in SM120 测试
scripts/                 环境、revision、审计、内存、质量和 smoke 辅助脚本
configs/                 确定性的 shape 与 evaluation 配置
schemas/                 机器可读的 evidence 契约
upstream/                固定的 upstream base revision
```

## 工程立场

项目的核心选择是让每一条性能结论都可追溯：量化语义由 reference contract 固化，kernel 行为与 unfused oracle 对照，报告从结构化 evidence 生成。目标是形成一条可信的 FlashInfer/vLLM 贡献路径，而不是脱离环境的 benchmark 截图。

## 项目状态

当前已公开 reference layer、可移植 validation suite、public audit 和初始 SM120 kernel-lab scaffolding。fused row-scale kernel、完整 vLLM dense online path、租用 GPU 上的质量/性能 evidence 以及 upstream submission 尚未完成，README 不将它们描述为已交付能力。

## 许可证

TokenScaleFP4 使用 [Apache License 2.0](LICENSE) 发布。
