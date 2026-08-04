# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import nn

from tokenscalefp4.contracts import validate_per_token_scale

_NVFP4_PER_TOKEN_SCALE_INV = 1.0 / (448.0 * 6.0)


class _SfLayout(Protocol):
    layout_128x4: object


class _FlashInfer(Protocol):
    SfLayout: _SfLayout
    mm_fp4: Callable[..., torch.Tensor]
    nvfp4_quantize: Callable[..., tuple[torch.Tensor, ...]]


def _flashinfer() -> _FlashInfer:
    return cast(_FlashInfer, importlib.import_module("flashinfer"))


@dataclass(frozen=True, slots=True)
class Nvfp4Activation:
    packed: torch.Tensor
    block_scales: torch.Tensor
    row_scales: torch.Tensor


@dataclass(frozen=True, slots=True)
class Nvfp4Weight:
    packed: torch.Tensor
    block_scales: torch.Tensor
    alpha: torch.Tensor
    logical_shape: tuple[int, int]


class Nvfp4Linear(nn.Module):
    """Dense linear layer backed by the Task 6 unfused NVFP4 oracle."""

    def __init__(
        self,
        weight: Nvfp4Weight,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.packed_weight: torch.Tensor
        self.block_scales: torch.Tensor
        self.alpha: torch.Tensor
        self.bias: torch.Tensor | None
        self.register_buffer("packed_weight", weight.packed)
        self.register_buffer("block_scales", weight.block_scales)
        self.register_buffer("alpha", weight.alpha)
        self.register_buffer("bias", bias)
        self.out_features, self.in_features = weight.logical_shape

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> Nvfp4Linear:
        weight = quantize_weight(linear.weight.detach())
        linear_bias = cast(torch.Tensor | None, linear.bias)
        bias = linear_bias.detach().clone() if linear_bias is not None else None
        return cls(weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"NVFP4 linear expected input K={self.in_features}; got {x.shape[-1]}"
            )
        leading_shape = x.shape[:-1]
        activation = quantize_activation_per_token(x.reshape(-1, self.in_features))
        output = mm_fp4_unfused(
            activation.packed,
            self.packed_weight.T,
            activation.block_scales,
            self.block_scales.T,
            self.alpha,
            activation.row_scales,
        )
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*leading_shape, self.out_features)


def quantize_activation_per_token(x: torch.Tensor) -> Nvfp4Activation:
    flashinfer = _flashinfer()
    packed, block_scales, row_scales = flashinfer.nvfp4_quantize(
        x,
        _NVFP4_PER_TOKEN_SCALE_INV,
        sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False,
        sf_vec_size=16,
        per_token_activation=True,
        backend="cute-dsl",
    )
    validate_per_token_scale(row_scales, m=x.shape[0], device=x.device)
    return Nvfp4Activation(
        packed=packed,
        block_scales=block_scales,
        row_scales=row_scales,
    )


def quantize_weight(weight: torch.Tensor) -> Nvfp4Weight:
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError("NVFP4 weights must be two-dimensional BF16 tensors")
    if weight.shape[1] % 32 != 0:
        raise ValueError("NVFP4 weight K must be divisible by 32")
    amax = weight.float().abs().max()
    encode_scale = torch.where(
        amax == 0,
        torch.full_like(amax, torch.finfo(torch.float32).max),
        (448.0 * 6.0) / amax,
    )
    flashinfer = _flashinfer()
    packed, block_scales = flashinfer.nvfp4_quantize(
        weight,
        encode_scale,
        sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False,
        sf_vec_size=16,
        backend="cuda",
    )
    return Nvfp4Weight(
        packed=packed,
        block_scales=block_scales,
        alpha=encode_scale.reciprocal().reshape(1),
        logical_shape=(weight.shape[0], weight.shape[1]),
    )


def mm_fp4_unfused(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: torch.Tensor,
    row_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    validate_per_token_scale(row_scale, m=a.shape[0], device=a.device)
    y = _flashinfer().mm_fp4(
        a,
        b,
        a_descale,
        b_descale,
        alpha=alpha,
        out_dtype=torch.bfloat16,
        out=out,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="b12x",
        use_nvfp4=True,
    )
    return y.mul_(row_scale[:, None])
