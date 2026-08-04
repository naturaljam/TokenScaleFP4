# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from tokenscalefp4.reference import dequantize_nvfp4, reference_mm

E2M1_VALUES = torch.tensor(
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


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    codes = codes.to(torch.uint8)
    assert codes.ndim == 2
    assert codes.shape[1] % 2 == 0
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def _swizzle_block_scales(scales: torch.Tensor) -> torch.Tensor:
    scales = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    rows, columns = scales.shape
    padded_rows = ((rows + 127) // 128) * 128
    padded_columns = ((columns + 3) // 4) * 4
    storage = torch.zeros(padded_rows * padded_columns, dtype=torch.uint8)
    for row in range(rows):
        for column in range(columns):
            offset = (
                column % 4
                + (column // 4) * 512
                + (row % 32) * 16
                + ((row % 128) // 32) * 4
                + (row // 128) * 128 * padded_columns
            )
            storage[offset] = scales[row, column]
    return storage.reshape(padded_rows, padded_columns)


def _unit_block_scales(rows: int, values: int) -> torch.Tensor:
    return _swizzle_block_scales(
        torch.ones((rows, (values + 15) // 16), dtype=torch.float32)
    )


def test_dequantize_unpacks_low_nibble_first_for_every_e2m1_code() -> None:
    codes = torch.arange(16, dtype=torch.uint8).reshape(1, 16)

    result = dequantize_nvfp4(
        _pack_codes(codes),
        _unit_block_scales(1, 16),
        torch.tensor([1.0], dtype=torch.float32),
        logical_shape=(1, 16),
    )

    torch.testing.assert_close(result, E2M1_VALUES.reshape(1, 16))


def test_dequantize_applies_one_e4m3_scale_per_16_values() -> None:
    codes = torch.full((1, 32), 2, dtype=torch.uint8)
    scales = _swizzle_block_scales(torch.tensor([[0.5, 2.0]]))

    result = dequantize_nvfp4(
        _pack_codes(codes),
        scales,
        torch.tensor(3.0, dtype=torch.float32),
        logical_shape=(1, 32),
    )

    expected = torch.cat((torch.full((16,), 1.5), torch.full((16,), 6.0)))
    torch.testing.assert_close(result, expected.reshape(1, 32))


def test_dequantize_reads_the_canonical_128x4_scale_layout() -> None:
    codes = torch.full((33, 80), 2, dtype=torch.uint8)
    logical_scales = torch.ones((33, 5))
    logical_scales[0, 4] = 4.0
    logical_scales[1, 0] = 2.0
    logical_scales[32, 3] = 3.0

    result = dequantize_nvfp4(
        _pack_codes(codes),
        _swizzle_block_scales(logical_scales),
        torch.tensor(1.0, dtype=torch.float32),
        logical_shape=(33, 80),
    )

    expected = logical_scales.repeat_interleave(16, dim=1)
    torch.testing.assert_close(result, expected)


def test_dequantize_requires_raw_e4m3_scale_bytes() -> None:
    scales = torch.ones((128, 4), dtype=torch.float8_e4m3fn)

    with pytest.raises(ValueError, match="raw E4M3 bytes"):
        dequantize_nvfp4(
            _pack_codes(torch.full((1, 16), 2, dtype=torch.uint8)),
            scales,
            torch.tensor(1.0, dtype=torch.float32),
            logical_shape=(1, 16),
        )


def test_dequantize_crops_physical_row_and_column_padding() -> None:
    codes = torch.full((4, 64), 2, dtype=torch.uint8)
    codes[3, :] = 7
    codes[:, 32:] = 7
    scales = _unit_block_scales(4, 64)

    result = dequantize_nvfp4(
        _pack_codes(codes),
        scales,
        torch.tensor(1.0, dtype=torch.float32),
        logical_shape=(3, 32),
    )

    assert result.shape == (3, 32)
    torch.testing.assert_close(result, torch.ones((3, 32)))


def test_dequantize_crops_padding_beyond_the_minimum_128x4_tiles() -> None:
    codes = torch.full((256, 128), 2, dtype=torch.uint8)
    codes[120:, :] = 7
    codes[:, 64:] = 7
    physical_scales = torch.ones((256, 8))
    physical_scales[120:, :] = 6.0
    physical_scales[:, 4:] = 6.0

    result = dequantize_nvfp4(
        _pack_codes(codes),
        _swizzle_block_scales(physical_scales),
        torch.tensor(1.0, dtype=torch.float32),
        logical_shape=(120, 64),
    )

    assert result.shape == (120, 64)
    torch.testing.assert_close(result, torch.ones((120, 64)))


def test_reference_mm_handles_zero_rows_alternating_signs_and_one_outlier() -> None:
    a_codes = torch.tensor(
        [
            [0] * 32,
            [2, 10] * 16,
            [2] * 31 + [7],
        ],
        dtype=torch.uint8,
    )
    b_codes = torch.tensor(
        [
            [2] * 32,
            [10] * 32,
            [2, 10] * 16,
            [3] * 32,
            [11] * 32,
            [7] + [0] * 31,
            [15] + [0] * 31,
            [2] * 16 + [10] * 16,
        ],
        dtype=torch.uint8,
    )
    row_scale = torch.tensor([2.0**-12, 1.0, 2.0**12], dtype=torch.float32)

    result = reference_mm(
        _pack_codes(a_codes),
        _pack_codes(b_codes),
        _unit_block_scales(3, 32),
        _unit_block_scales(8, 32),
        torch.tensor([0.5], dtype=torch.float32),
        row_scale,
        m=3,
        n=8,
        k=32,
    )

    a = E2M1_VALUES[a_codes.long()]
    b = E2M1_VALUES[b_codes.long()]
    expected = (0.5 * (a @ b.T) * row_scale[:, None]).to(torch.bfloat16)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.count_nonzero(result[0]) == 0


def test_reference_mm_crops_padded_m_and_n() -> None:
    a_codes = torch.full((4, 32), 2, dtype=torch.uint8)
    b_codes = torch.full((8, 32), 2, dtype=torch.uint8)

    result = reference_mm(
        _pack_codes(a_codes),
        _pack_codes(b_codes),
        _unit_block_scales(4, 32),
        _unit_block_scales(8, 32),
        torch.tensor(1.0, dtype=torch.float32),
        None,
        m=3,
        n=5,
        k=32,
    )

    assert result.shape == (3, 5)
    torch.testing.assert_close(result, torch.full((3, 5), 32.0).to(torch.bfloat16))


def test_reference_mm_ones_row_scale_is_bitwise_equal_to_scalar_path() -> None:
    generator = torch.Generator().manual_seed(20260803)
    a_codes = torch.randint(0, 16, (3, 32), generator=generator, dtype=torch.uint8)
    b_codes = torch.randint(0, 16, (5, 32), generator=generator, dtype=torch.uint8)
    arguments = (
        _pack_codes(a_codes),
        _pack_codes(b_codes),
        _unit_block_scales(3, 32),
        _unit_block_scales(5, 32),
        torch.tensor([0.75], dtype=torch.float32),
    )

    scalar = reference_mm(*arguments, None, m=3, n=5, k=32)
    row_scaled = reference_mm(
        *arguments,
        torch.ones(3, dtype=torch.float32),
        m=3,
        n=5,
        k=32,
    )

    assert torch.equal(row_scaled, scalar)
