# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn


def load_flashinfer_ops() -> Any:
    try:
        return importlib.import_module("tokenscalefp4.kernel_lab.flashinfer_ops")
    except ModuleNotFoundError as error:
        pytest.fail(f"Task 6 flashinfer wrapper is missing: {error}")


class FakeOutput:
    def __init__(self) -> None:
        self.mul_calls: list[torch.Tensor] = []

    def mul_(self, scale: torch.Tensor) -> FakeOutput:
        self.mul_calls.append(scale)
        return self


def test_mm_fp4_unfused_uses_b12x_and_one_device_side_row_multiply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    output = FakeOutput()

    def mm_fp4(*args: object, **kwargs: object) -> FakeOutput:
        calls.append((args, kwargs))
        return output

    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(mm_fp4=mm_fp4))
    ops = load_flashinfer_ops()
    a = SimpleNamespace(shape=(3, 16), device=torch.device("cpu"))
    b = object()
    a_descale = object()
    b_descale = object()
    alpha = object()
    row_scale = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    original_row_scale = row_scale.clone()

    actual = ops.mm_fp4_unfused(
        a,
        b,
        a_descale,
        b_descale,
        alpha,
        row_scale,
        out=output,
    )

    assert actual is output
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (a, b, a_descale, b_descale)
    assert kwargs == {
        "alpha": alpha,
        "out_dtype": torch.bfloat16,
        "out": output,
        "block_size": 16,
        "use_8x4_sf_layout": False,
        "backend": "b12x",
        "use_nvfp4": True,
    }
    assert len(output.mul_calls) == 1
    assert output.mul_calls[0].shape == (3, 1)
    assert output.mul_calls[0].data_ptr() == row_scale.data_ptr()
    torch.testing.assert_close(row_scale, original_row_scale)


def test_quantize_activation_per_token_preserves_all_quantizer_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed = object()
    block_scales = object()
    row_scales = torch.tensor([0.25, 0.5], dtype=torch.float32)
    calls: list[tuple[object, object, dict[str, object]]] = []
    layout_128x4 = object()

    def nvfp4_quantize(
        value: object,
        global_scale: object,
        **kwargs: object,
    ) -> tuple[object, object, torch.Tensor]:
        calls.append((value, global_scale, kwargs))
        return packed, block_scales, row_scales

    fake_flashinfer = SimpleNamespace(
        SfLayout=SimpleNamespace(layout_128x4=layout_128x4),
        nvfp4_quantize=nvfp4_quantize,
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    ops = load_flashinfer_ops()
    x = torch.ones((2, 32), dtype=torch.bfloat16)

    result = ops.quantize_activation_per_token(x)

    assert result.packed is packed
    assert result.block_scales is block_scales
    assert result.row_scales is row_scales
    assert len(calls) == 1
    value, global_scale, kwargs = calls[0]
    assert value is x
    assert global_scale == pytest.approx(1.0 / (448.0 * 6.0))
    assert kwargs == {
        "sfLayout": layout_128x4,
        "do_shuffle": False,
        "sf_vec_size": 16,
        "per_token_activation": True,
        "backend": "cute-dsl",
    }


def test_quantize_weight_uses_one_static_global_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed = torch.zeros((8, 16), dtype=torch.uint8)
    block_scales = torch.zeros((128, 4), dtype=torch.uint8)
    calls: list[tuple[torch.Tensor, torch.Tensor, dict[str, object]]] = []
    layout_128x4 = object()

    def nvfp4_quantize(
        value: torch.Tensor,
        global_scale: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((value, global_scale, kwargs))
        return packed, block_scales

    fake_flashinfer = SimpleNamespace(
        SfLayout=SimpleNamespace(layout_128x4=layout_128x4),
        nvfp4_quantize=nvfp4_quantize,
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    ops = load_flashinfer_ops()
    weight = torch.ones((8, 32), dtype=torch.bfloat16)

    result = ops.quantize_weight(weight)

    assert result.packed is packed
    assert result.block_scales is block_scales
    assert result.logical_shape == (8, 32)
    torch.testing.assert_close(
        result.alpha,
        torch.tensor([1.0 / (448.0 * 6.0)], dtype=torch.float32),
    )
    assert len(calls) == 1
    value, global_scale, kwargs = calls[0]
    assert value is weight
    torch.testing.assert_close(
        global_scale,
        torch.tensor(448.0 * 6.0, dtype=torch.float32),
    )
    assert kwargs == {
        "sfLayout": layout_128x4,
        "do_shuffle": False,
        "sf_vec_size": 16,
        "backend": "cuda",
    }


def test_nvfp4_linear_flattens_restores_shape_and_applies_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout_128x4 = object()

    def nvfp4_quantize(
        value: torch.Tensor,
        global_scale: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, ...]:
        if kwargs.get("per_token_activation"):
            return (
                torch.zeros(
                    (value.shape[0], value.shape[1] // 2),
                    dtype=torch.uint8,
                ),
                torch.zeros((128, 4), dtype=torch.uint8),
                torch.ones((value.shape[0],), dtype=torch.float32),
            )
        return (
            torch.zeros((value.shape[0], value.shape[1] // 2), dtype=torch.uint8),
            torch.zeros((128, 4), dtype=torch.uint8),
        )

    def mm_fp4(
        a: torch.Tensor,
        b: torch.Tensor,
        *_args: object,
        **_kwargs: object,
    ) -> torch.Tensor:
        return torch.zeros((a.shape[0], b.shape[1]), dtype=torch.bfloat16)

    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(
            SfLayout=SimpleNamespace(layout_128x4=layout_128x4),
            mm_fp4=mm_fp4,
            nvfp4_quantize=nvfp4_quantize,
        ),
    )
    ops = load_flashinfer_ops()
    source = nn.Linear(32, 8, bias=True, dtype=torch.bfloat16)
    source.weight.data.fill_(1.0)
    source.bias.data.fill_(1.0)

    layer = ops.Nvfp4Linear.from_linear(source)
    output = layer(torch.ones((2, 3, 32), dtype=torch.bfloat16))

    assert output.shape == (2, 3, 8)
    torch.testing.assert_close(output, torch.ones_like(output))
    assert layer.packed_weight.shape == (8, 16)
    assert layer.block_scales.shape == (128, 4)
    assert layer.alpha.shape == (1,)


def test_kernel_lab_exports_task6_primitives() -> None:
    from tokenscalefp4 import kernel_lab

    assert kernel_lab.Nvfp4Activation is not None
    assert kernel_lab.Nvfp4Weight is not None
    assert kernel_lab.mm_fp4_unfused is not None
    assert kernel_lab.quantize_activation_per_token is not None
