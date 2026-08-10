# TokenScaleFP4

[![CI](https://github.com/naturaljam/TokenScaleFP4/actions/workflows/ci.yml/badge.svg)](https://github.com/naturaljam/TokenScaleFP4/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-2F8552.svg)](LICENSE)

[![English](https://img.shields.io/badge/README-English-111827?style=for-the-badge)](README.md)
[![Chinese](https://img.shields.io/badge/README-中文-0f766e?style=for-the-badge)](README.zh-CN.md)

**Experimental NVFP4 inference infrastructure for per-token activation scaling on NVIDIA Blackwell SM120 GPUs.**

TokenScaleFP4 is building an end-to-end path from a precise quantization contract to a fused FlashInfer GEMM and, eventually, vLLM online dense quantization. The project focuses on one concrete problem: preserving the scale produced by per-token activation quantization and applying it to the correct output row without adding a separate post-GEMM kernel.

## The Problem

Dynamic NVFP4 activation quantization produces a scale for every logical input row. FlashInfer's existing `mm_fp4` scalar `alpha` represents a global weight scale, so it cannot also carry an `M`-element activation scale.

The target computation is:

```text
Y[m, n] = BF16(
    per_token_scale[m] * alpha * block_scaled_fp4_dot(A[m, :], W[n, :])
)
```

TokenScaleFP4 defines that contract, validates it against a dequantized FP32 reference, and prepares the API and test surface required to fuse `per_token_scale[m]` into the SM120 b12x GEMM epilogue.

## Target Path

```text
BF16 activation [M, K]
  -> per-token NVFP4 quantization
  -> packed activation + block scales + row scale [M]
  -> FlashInfer b12x FP4 GEMM
  -> fused alpha * row_scale[m]
  -> BF16 output [M, N]
  -> vLLM dense linear dispatch
```

The fused path is intended to remove the extra row-scaling launch used by the current oracle while preserving the scalar path, preallocated output, compilation cache behavior, and CUDA Graph replay.

## What Is Implemented

| Area | Available now |
| --- | --- |
| Contract | SM120 NVFP4 shape, block-size, BF16 output, and FP32 row-scale validation |
| Reference | E2M1 unpacking, 128x4 E4M3 scale decoding, FP32 GEMM, and BF16 output |
| Oracle | Per-token activation quantization and unfused `mm_fp4 + row scale` execution |
| Kernel lab | FlashInfer b12x dispatch helpers, Qwen shape handling, and pinned upstream revisions |
| Evidence | Numerical metrics, quality/memory gates, revision capture, and validated evidence records |
| Tests | 167 portable tests plus opt-in SM120 smoke/oracle tests |

The fused row-scale epilogue, complete vLLM dense online path, and final rented-GPU performance evidence are still in progress. This repository does not present them as finished features.

## Scope

- NVIDIA Blackwell **SM120**
- NVFP4 with group size 16 and 128x4 scale storage
- BF16 activation and output
- FlashInfer b12x backend
- TP1 dense linear layers
- Qwen2.5 projection shapes as the primary validation set

SM100/SM103, MoE, FP8, training, KV cache, and multi-GPU execution are outside the current scope. There is no claim of an end-to-end speedup over BF16 until model-level measurements are complete.

## Quick Start

Portable development requires Python 3.11-3.13. SM120 tests additionally require Linux or WSL2, CUDA 13+, an SM120 GPU, and the pinned FlashInfer checkout.

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

## Repository Map

```text
src/tokenscalefp4/contracts/      public input and shape contracts
src/tokenscalefp4/reference/      NVFP4 dequantization and FP32 oracle
src/tokenscalefp4/kernel_lab/     FlashInfer dispatch and unfused execution
src/tokenscalefp4/reporting/      evidence records and validation
tests/                            portable and opt-in SM120 coverage
scripts/                          environment, evaluation, audit, and smoke tools
configs/                          fixed shape and quality configurations
upstream/                         pinned FlashInfer and vLLM base revisions
```

## Validation Rules

The project uses fixed gates instead of hand-picked examples. The reference and oracle suites check row-scale shape and layout, scalar-path compatibility, BF16 output behavior, numerical error, invalid inputs, output buffers, and representative Qwen dimensions. Hardware results remain separate from portable CI so that a green public build does not imply an SM120 performance result.

## License

TokenScaleFP4 is available under the [Apache License 2.0](LICENSE).
