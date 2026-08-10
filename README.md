# TokenScaleFP4

[![CI](https://github.com/naturaljam/TokenScaleFP4/actions/workflows/ci.yml/badge.svg)](https://github.com/naturaljam/TokenScaleFP4/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-2F8552.svg)](LICENSE)

[![English](https://img.shields.io/badge/README-English-111827?style=for-the-badge)](README.md)
[![中文](https://img.shields.io/badge/README-中文-0f766e?style=for-the-badge)](README.zh-CN.md)

**A focused AI Infra project for making dynamic, per-token NVFP4 inference practical on consumer Blackwell SM120 GPUs.**

TokenScaleFP4 is my primary open-source project for studying the boundary between quantization contracts, GPU kernel dispatch, and reproducible inference evidence. It builds a small, testable foundation around NVFP4 rather than presenting an unverified end-to-end speedup claim.

## What It Delivers

- A precise NVFP4 contract and FP32/BF16 reference implementation.
- An unfused per-token row-scale oracle for checking future fused kernels.
- FlashInfer-oriented SM120 shape, backend, patch, and dispatch helpers.
- Public input-validation and evidence-reporting schemas for repeatable experiments.
- A CI-tested Python package that keeps hardware-specific tests separate from portable tests.

The public repository intentionally contains source, tests, scripts, and reproducibility metadata only. Private research notes, rented-machine details, raw profiler output, and model artifacts stay outside the repository.

## Current Scope

The current public baseline is experimental and deliberately narrow:

- NVIDIA Blackwell **SM120 only**
- NVFP4, group size 16, with the documented 128x4 scale layout
- BF16 output and TP1-oriented shapes
- Reference/oracle and API-validation layers first; production fused-kernel and vLLM integration remain follow-up work

This project does not claim support for SM100/SM103, MoE quantization, FP8, training, KV cache, multi-GPU execution, or a guaranteed end-to-end win over BF16.

## Quick Start

Requirements: Python 3.11-3.13. GPU-only smoke tests additionally require an SM120 GPU, CUDA 13+, and the pinned upstream environment.

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

The portable suite covers the reference contract, shape and dispatch logic, patch ownership, reporting schemas, and public-repository checks. `sm120` tests are opt-in and fail closed when the required hardware or Linux/CUDA environment is unavailable.

## Repository Map

```text
src/tokenscalefp4/       public package: contracts, reference, kernel lab, reporting
tests/                   portable contracts and opt-in SM120 tests
scripts/                 environment, revision, audit, memory, quality, and smoke helpers
configs/                 deterministic shape and evaluation configuration
schemas/                 machine-readable evidence contracts
upstream/                pinned upstream base revisions
```

## Engineering Position

The central design choice is to make every performance claim traceable: quantization semantics live in a reference contract, kernel behavior is compared with an unfused oracle, and reports are derived from structured evidence. The goal is a credible path to FlashInfer and vLLM contributions, not a benchmark screenshot detached from its environment.

## Status

The reference layer, portable validation suite, public audit, and initial SM120 kernel-lab scaffolding are available now. The fused row-scale kernel, full vLLM dense online path, rented-GPU quality/performance evidence, and upstream submissions are intentionally not represented as complete.

## License

TokenScaleFP4 is released under the [Apache License 2.0](LICENSE).
