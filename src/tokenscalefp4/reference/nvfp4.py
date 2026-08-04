# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from tokenscalefp4.contracts import Nvfp4Problem, validate_per_token_scale

_E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _validate_scalar(scale: torch.Tensor, *, name: str, device: torch.device) -> None:
    if scale.dtype != torch.float32 or scale.numel() != 1:
        raise ValueError(f"{name} must contain exactly one FP32 element")
    if scale.device != device:
        raise ValueError(f"{name} must be on the same device as the packed values")


def dequantize_nvfp4(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    logical_shape: tuple[int, int],
) -> torch.Tensor:
    """Dequantize E2M1 values and canonical 128x4 E4M3 scale storage."""

    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError("packed NVFP4 values must be a two-dimensional uint8 tensor")
    if block_scales.ndim != 2 or block_scales.dtype != torch.uint8:
        raise ValueError("block_scales must be two-dimensional raw E4M3 bytes")
    if not block_scales.is_contiguous():
        raise ValueError("block_scales must use contiguous 128x4 storage")
    if block_scales.device != packed.device:
        raise ValueError("block_scales must share the packed values' device")
    _validate_scalar(global_scale, name="global_scale", device=packed.device)

    logical_rows, logical_columns = logical_shape
    if logical_rows <= 0 or logical_columns <= 0:
        raise ValueError("logical_shape dimensions must be positive")
    if packed.shape[0] < logical_rows or packed.shape[1] * 2 < logical_columns:
        raise ValueError("packed storage is smaller than logical_shape")

    required_blocks = (logical_columns + 15) // 16
    minimum_scale_rows = ((logical_rows + 127) // 128) * 128
    minimum_scale_columns = ((required_blocks + 3) // 4) * 4
    storage_rows, storage_columns = block_scales.shape
    if (
        storage_rows < minimum_scale_rows
        or storage_rows % 128 != 0
        or storage_columns < minimum_scale_columns
        or storage_columns % 4 != 0
    ):
        raise ValueError(
            "block_scales must use canonical 128x4 padded storage with at least "
            f"shape ({minimum_scale_rows}, {minimum_scale_columns})"
        )

    codes = torch.empty(
        (packed.shape[0], packed.shape[1] * 2),
        dtype=torch.long,
        device=packed.device,
    )
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = (packed >> 4) & 0x0F

    values = _E2M1.to(device=packed.device)[
        codes[:logical_rows, :logical_columns]
    ]

    rows = torch.arange(logical_rows, dtype=torch.long, device=packed.device)[:, None]
    columns = torch.arange(
        required_blocks,
        dtype=torch.long,
        device=packed.device,
    )[None, :]
    # This index mapping follows FlashInfer's Apache-2.0 128x4 layout contract.
    scale_offsets = (
        columns % 4
        + (columns // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + (rows // 128) * 128 * storage_columns
    )
    logical_scales = block_scales.reshape(-1)[scale_offsets]
    logical_scales = logical_scales.view(torch.float8_e4m3fn).float()
    expanded_scales = logical_scales.repeat_interleave(16, dim=1)

    dequantized = values * expanded_scales[:, :logical_columns]
    dequantized = dequantized * global_scale.reshape(())
    return dequantized


def reference_mm(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    a_block_scales: torch.Tensor,
    b_block_scales: torch.Tensor,
    alpha: torch.Tensor,
    per_token_scale: torch.Tensor | None,
    *,
    m: int,
    n: int,
    k: int,
) -> torch.Tensor:
    """Evaluate the token-scaled NVFP4 GEMM contract on dequantized FP32 data."""

    Nvfp4Problem(m=m, n=n, k=k)
    if a_packed.device != b_packed.device:
        raise ValueError("packed operands must be on the same device")
    _validate_scalar(alpha, name="alpha", device=a_packed.device)
    if per_token_scale is not None:
        validate_per_token_scale(per_token_scale, m=m, device=a_packed.device)

    one = torch.ones((), dtype=torch.float32, device=a_packed.device)
    a_fp32 = dequantize_nvfp4(
        a_packed,
        a_block_scales,
        one,
        logical_shape=(m, k),
    )
    b_fp32 = dequantize_nvfp4(
        b_packed,
        b_block_scales,
        one,
        logical_shape=(n, k),
    )
    out = alpha.float().reshape(()) * (a_fp32 @ b_fp32.T)
    if per_token_scale is not None:
        out = out * per_token_scale.float()[:, None]
    return out.to(torch.bfloat16)
