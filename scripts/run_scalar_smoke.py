# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from check_environment import collect_environment

from tokenscalefp4.reference.metrics import cosine_similarity

MINIMUM_COSINE_SIMILARITY = 0.999
DEFAULT_FLASHINFER_SOURCE_ALIAS = Path("/opt/tokenscalefp4/flashinfer")
DEFAULT_FLASHINFER_WORKSPACE_BASE = Path("/opt/tokenscalefp4/runtime")


@dataclass(frozen=True, slots=True)
class SmokeResult:
    output_dtype: str
    all_finite: bool
    cosine_similarity: float

    def require(self) -> None:
        if self.output_dtype != "bfloat16":
            raise SystemExit(
                f"scalar b12x smoke requires BF16 output; got {self.output_dtype}"
            )
        if not self.all_finite:
            raise SystemExit("scalar b12x smoke output must contain only finite values")
        if self.cosine_similarity < MINIMUM_COSINE_SIMILARITY:
            raise SystemExit(
                "scalar b12x smoke cosine similarity must be at least "
                f"{MINIMUM_COSINE_SIMILARITY}; got {self.cosine_similarity:.6f}"
            )


def prepare_runtime_environment(
    *,
    executable: Path = Path(sys.executable),
    flashinfer_source: Path | None = DEFAULT_FLASHINFER_SOURCE_ALIAS,
    workspace_base: Path | None = DEFAULT_FLASHINFER_WORKSPACE_BASE,
) -> None:
    interpreter_bin = str(executable.parent)
    existing = [
        entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry
    ]
    os.environ["PATH"] = os.pathsep.join(
        [interpreter_bin, *(entry for entry in existing if entry != interpreter_bin)]
    )
    if flashinfer_source is not None and flashinfer_source.is_dir():
        source = str(flashinfer_source)
        sys.path[:] = [source, *(entry for entry in sys.path if entry != source)]
        if workspace_base is not None:
            workspace_base.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(workspace_base))


def run_scalar_smoke(*, m: int, n: int, k: int, seed: int) -> SmokeResult:
    prepare_runtime_environment()

    import torch
    from flashinfer import (
        SfLayout,
        e2m1_and_ufp8sf_scale_to_float,
        mm_fp4,
        nvfp4_quantize,
    )

    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("M, N, and K must be positive")
    if n % 8 != 0:
        raise ValueError("N must be divisible by 8 for the b12x backend")
    if k % 32 != 0:
        raise ValueError("K must be divisible by 32 for the b12x backend")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    a_encode_scale = (448.0 * 6.0) / a.float().abs().nan_to_num().max()
    b_encode_scale = (448.0 * 6.0) / b.float().abs().nan_to_num().max()

    a_packed, a_block_scales = nvfp4_quantize(
        a,
        a_encode_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    b_packed, b_block_scales = nvfp4_quantize(
        b,
        b_encode_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    a_decode_scale = a_encode_scale.reciprocal()
    b_decode_scale = b_encode_scale.reciprocal()
    alpha = a_decode_scale * b_decode_scale

    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    mm_fp4(
        a_packed,
        b_packed.T,
        a_block_scales,
        b_block_scales.T,
        alpha,
        torch.bfloat16,
        out,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="b12x",
        use_nvfp4=True,
        skip_check=False,
    )

    a_dequantized = e2m1_and_ufp8sf_scale_to_float(
        a_packed,
        a_block_scales,
        a_decode_scale.reshape(1),
        sf_vec_size=16,
        ufp8_type=1,
        is_sf_swizzled_layout=True,
    )
    b_dequantized = e2m1_and_ufp8sf_scale_to_float(
        b_packed,
        b_block_scales,
        b_decode_scale.reshape(1),
        sf_vec_size=16,
        ufp8_type=1,
        is_sf_swizzled_layout=True,
    )
    reference = (a_dequantized.float() @ b_dequantized.float().T).to(torch.bfloat16)
    torch.cuda.synchronize()

    return SmokeResult(
        output_dtype=str(out.dtype).removeprefix("torch."),
        all_finite=bool(torch.isfinite(out).all().item()),
        cosine_similarity=cosine_similarity(out.cpu(), reference.cpu()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the scalar SM120 b12x FP4 gate")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    facts = collect_environment()
    facts.require_sm120_b12x()
    result = run_scalar_smoke(m=args.m, n=args.n, k=args.k, seed=args.seed)
    result.require()
    payload = {
        "backend": "b12x",
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "seed": args.seed,
        **asdict(result),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
