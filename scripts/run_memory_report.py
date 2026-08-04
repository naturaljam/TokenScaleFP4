# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "evals" / "quality.json"


class MemoryResult(NamedTuple):
    model: str
    model_revision: str
    eligible_weight_count: int
    bf16_bytes: int
    packed_bytes: int
    block_scale_bytes: int
    global_scale_bytes: int
    padding_bytes: int
    peak_cuda_bytes: int

    @property
    def quantized_allocated_bytes(self) -> int:
        return (
            self.packed_bytes
            + self.block_scale_bytes
            + self.global_scale_bytes
            + self.padding_bytes
        )

    @property
    def reduction(self) -> float:
        if self.bf16_bytes <= 0:
            raise ValueError("BF16 eligible-weight bytes must be positive")
        return 1.0 - (self.quantized_allocated_bytes / self.bf16_bytes)

    def require(self, *, minimum_reduction: float) -> None:
        if self.reduction < minimum_reduction:
            raise SystemExit(
                "eligible quantized weights must reduce allocated bytes by at least "
                f"{minimum_reduction:.2%}; got {self.reduction:.2%}"
            )


class Nvfp4Storage(NamedTuple):
    packed_bytes: int
    block_scale_bytes: int
    global_scale_bytes: int
    padding_bytes: int

    @property
    def allocated_bytes(self) -> int:
        return (
            self.packed_bytes
            + self.block_scale_bytes
            + self.global_scale_bytes
            + self.padding_bytes
        )


def nvfp4_storage_breakdown(*, rows: int, columns: int) -> Nvfp4Storage:
    if rows <= 0 or columns <= 0 or columns % 32 != 0:
        raise ValueError("NVFP4 weights require positive rows and K divisible by 32")
    packed_bytes = rows * (columns // 2)
    logical_scale_columns = columns // 16
    block_scale_bytes = rows * logical_scale_columns
    padded_rows = ((rows + 127) // 128) * 128
    padded_scale_columns = ((logical_scale_columns + 3) // 4) * 4
    allocated_scale_bytes = padded_rows * padded_scale_columns
    return Nvfp4Storage(
        packed_bytes=packed_bytes,
        block_scale_bytes=block_scale_bytes,
        global_scale_bytes=4,
        padding_bytes=allocated_scale_bytes - block_scale_bytes,
    )


def build_memory_payload(result: MemoryResult) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": result.model,
        "model_revision": result.model_revision,
        "eligible_weight_count": result.eligible_weight_count,
        "bf16_bytes": result.bf16_bytes,
        "packed_bytes": result.packed_bytes,
        "block_scale_bytes": result.block_scale_bytes,
        "global_scale_bytes": result.global_scale_bytes,
        "padding_bytes": result.padding_bytes,
        "quantized_allocated_bytes": result.quantized_allocated_bytes,
        "reduction": result.reduction,
        "peak_cuda_bytes": result.peak_cuda_bytes,
    }


def run_memory_report(
    model_name: str,
    model_revision: str,
    *,
    device: str = "cuda",
) -> MemoryResult:
    import torch
    from transformers import AutoModelForCausalLM

    from tokenscalefp4.kernel_lab.flashinfer_ops import Nvfp4Linear, quantize_weight

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.to(device)

    import run_quality_eval

    eligible = [
        (name, module)
        for name, module in model.named_modules()
        if name
        and isinstance(module, torch.nn.Linear)
        and run_quality_eval.is_eligible_linear(name)
    ]
    if not eligible:
        raise RuntimeError("no eligible dense projection layers were found")

    bf16_bytes = 0
    packed_bytes = 0
    block_scale_bytes = 0
    global_scale_bytes = 0
    padding_bytes = 0
    for name, linear in eligible:
        bf16_bytes += linear.weight.numel() * linear.weight.element_size()
        quantized = quantize_weight(linear.weight.detach())
        rows, columns = quantized.logical_shape
        block_scale_bytes += rows * (columns // 16)
        padding_bytes += quantized.block_scales.numel() - rows * (columns // 16)
        packed_bytes += quantized.packed.numel() * quantized.packed.element_size()
        global_scale_bytes += quantized.alpha.numel() * quantized.alpha.element_size()
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        parent_name, child_name = (
            name.rsplit(".", maxsplit=1) if "." in name else ("", name)
        )
        parent = model.get_submodule(parent_name) if parent_name else model
        parent.add_module(child_name, Nvfp4Linear(quantized, bias))

    peak_cuda_bytes = (
        torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else 0
    )
    return MemoryResult(
        model=model_name,
        model_revision=model_revision,
        eligible_weight_count=len(eligible),
        bf16_bytes=bf16_bytes,
        packed_bytes=packed_bytes,
        block_scale_bytes=block_scale_bytes,
        global_scale_bytes=global_scale_bytes,
        padding_bytes=padding_bytes,
        peak_cuda_bytes=int(peak_cuda_bytes),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report dense NVFP4 memory accounting")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import run_quality_eval

    settings = run_quality_eval.load_quality_settings(args.config, args.model)
    result = run_memory_report(
        settings.model,
        settings.model_revision,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_memory_payload(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(build_memory_payload(result), indent=2, sort_keys=True))
    return 0 if result.reduction >= 0.65 else 1


if __name__ == "__main__":
    raise SystemExit(main())
