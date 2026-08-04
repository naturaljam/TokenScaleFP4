# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from tokenscalefp4.contracts import (
    Nvfp4Problem,
    validate_output_dtype,
    validate_per_token_scale,
)


@pytest.mark.parametrize("field", ["m", "n"])
@pytest.mark.parametrize("value", [0, -1])
def test_problem_requires_positive_output_dimensions(field: str, value: int) -> None:
    dimensions = {"m": 4, "n": 8, "k": 32}
    dimensions[field] = value

    with pytest.raises(ValueError, match=rf"{field.upper()} must be positive"):
        Nvfp4Problem(**dimensions)


@pytest.mark.parametrize("k", [0, 16, 33])
def test_problem_requires_k_divisible_by_32(k: int) -> None:
    with pytest.raises(ValueError, match="K must be positive and divisible by 32"):
        Nvfp4Problem(m=4, n=8, k=k)


def test_problem_requires_nvfp4_block_size() -> None:
    with pytest.raises(ValueError, match="block_size must be 16"):
        Nvfp4Problem(m=4, n=8, k=32, block_size=32)


def test_problem_accepts_the_nvfp4_contract() -> None:
    assert Nvfp4Problem(m=4, n=8, k=64) == Nvfp4Problem(
        m=4,
        n=8,
        k=64,
        block_size=16,
    )


@pytest.mark.parametrize("shape", [(4, 1), (3,), (4, 1, 1)])
def test_row_scale_requires_exact_vector(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"exact shape \[4\]"):
        validate_per_token_scale(
            torch.ones(shape),
            m=4,
            device=torch.device("cpu"),
        )


def test_row_scale_requires_fp32() -> None:
    with pytest.raises(ValueError, match="FP32"):
        validate_per_token_scale(
            torch.ones(4, dtype=torch.bfloat16),
            m=4,
            device=torch.device("cpu"),
        )


def test_row_scale_requires_contiguous_storage() -> None:
    scale = torch.ones(8, dtype=torch.float32)[::2]
    assert not scale.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        validate_per_token_scale(scale, m=4, device=torch.device("cpu"))


def test_row_scale_must_share_the_operand_device() -> None:
    with pytest.raises(ValueError, match="same device"):
        validate_per_token_scale(
            torch.ones(4, dtype=torch.float32),
            m=4,
            device=torch.device("cuda", 0),
        )


def test_row_scale_accepts_a_matching_cpu_fixture() -> None:
    validate_per_token_scale(
        torch.ones(4, dtype=torch.float32),
        m=4,
        device=torch.device("cpu"),
    )


def test_output_requires_bfloat16() -> None:
    with pytest.raises(ValueError, match="BF16"):
        validate_output_dtype(torch.float16)


def test_output_accepts_bfloat16() -> None:
    validate_output_dtype(torch.bfloat16)
