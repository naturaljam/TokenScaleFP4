# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Nvfp4Problem:
    """Logical dimensions supported by the NVFP4 GEMM contract."""

    m: int
    n: int
    k: int
    block_size: int = 16

    def __post_init__(self) -> None:
        if self.m <= 0:
            raise ValueError("M must be positive")
        if self.n <= 0:
            raise ValueError("N must be positive")
        if self.k <= 0 or self.k % 32 != 0:
            raise ValueError("K must be positive and divisible by 32")
        if self.block_size != 16:
            raise ValueError("block_size must be 16 for NVFP4")


def validate_per_token_scale(
    scale: torch.Tensor,
    *,
    m: int,
    device: torch.device,
) -> None:
    """Validate row-scale metadata without scanning tensor values."""

    if tuple(scale.shape) != (m,):
        raise ValueError(f"per_token_scale must have exact shape [{m}]")
    if scale.dtype != torch.float32:
        raise ValueError("per_token_scale must use FP32 dtype")
    if not scale.is_contiguous():
        raise ValueError("per_token_scale must be contiguous")
    if scale.device != device:
        raise ValueError("per_token_scale must be on the same device as the operands")


def validate_output_dtype(out_dtype: torch.dtype) -> None:
    """Reject output types outside the BF16-only MVP contract."""

    if out_dtype != torch.bfloat16:
        raise ValueError("NVFP4 output must use BF16 dtype")
