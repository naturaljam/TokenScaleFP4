# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest
import torch

from tokenscalefp4.kernel_lab.flashinfer_ops import (
    mm_fp4_unfused,
    quantize_activation_per_token,
)
from tokenscalefp4.reference import (
    assert_all_finite,
    cosine_similarity,
    normalized_rmse,
    reference_mm,
)

REPO_ROOT = Path(__file__).parents[2]


def load_script(name: str) -> dict[str, Any]:
    return runpy.run_path(str(REPO_ROOT / "scripts" / name))


def adversarial_activation(*, m: int, k: int) -> torch.Tensor:
    value = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    value[0].zero_()
    value[1].mul_(1e-3)
    value[2].mul_(1e2)
    value[3].zero_()
    value[3, 0] = 256.0
    boundaries = torch.linspace(-6.0, 6.0, k, device="cuda")
    value[4].copy_(boundaries.to(torch.bfloat16))
    return value


@pytest.mark.sm120
def test_unfused_oracle_matches_fp32_dequantized_reference() -> None:
    environment = load_script("check_environment.py")
    facts = environment["collect_environment"]()
    try:
        facts.require_sm120_b12x()
    except SystemExit as error:
        pytest.skip(str(error))

    smoke = load_script("run_scalar_smoke.py")
    smoke["prepare_runtime_environment"]()

    from flashinfer import SfLayout, nvfp4_quantize

    m, n, k = 8, 256, 1536
    torch.manual_seed(20260803)
    torch.cuda.manual_seed_all(20260803)
    activation = adversarial_activation(m=m, k=k)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    activation_q = quantize_activation_per_token(activation)
    weight_encode_scale = (448.0 * 6.0) / weight.float().abs().max()
    weight_packed, weight_block_scales = nvfp4_quantize(
        weight,
        weight_encode_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    alpha = weight_encode_scale.reciprocal().reshape(1)

    actual = mm_fp4_unfused(
        activation_q.packed,
        weight_packed.T,
        activation_q.block_scales,
        weight_block_scales.T,
        alpha,
        activation_q.row_scales,
    )
    expected = reference_mm(
        activation_q.packed,
        weight_packed,
        activation_q.block_scales,
        weight_block_scales,
        alpha,
        activation_q.row_scales,
        m=m,
        n=n,
        k=k,
    )
    torch.cuda.synchronize()

    assert_all_finite(actual, name="unfused oracle output")
    cosine = cosine_similarity(actual.cpu(), expected.cpu())
    nrmse = normalized_rmse(actual.cpu(), expected.cpu())
    assert cosine >= 0.999, f"cosine={cosine:.9f}, normalized_rmse={nrmse:.9f}"
    assert nrmse >= 0.0
