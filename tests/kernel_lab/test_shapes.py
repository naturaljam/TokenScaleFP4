# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from tokenscalefp4.kernel_lab.shapes import GemmShape, load_shape_suite

M_VALUES = (1, 2, 4, 8, 16, 32, 128, 256, 512, 1024)


@pytest.mark.parametrize(
    ("model", "projection_pairs"),
    [
        (
            "Qwen/Qwen2.5-1.5B",
            {(1536, 1536), (256, 1536), (8960, 1536), (1536, 8960)},
        ),
        (
            "Qwen/Qwen2.5-7B",
            {(3584, 3584), (512, 3584), (18944, 3584), (3584, 18944)},
        ),
    ],
)
def test_qwen_shape_suite_expands_unique_projection_pairs(
    model: str, projection_pairs: set[tuple[int, int]]
) -> None:
    shapes = load_shape_suite(model)

    assert set(shapes) == {
        GemmShape(m=m, n=n, k=k)
        for n, k in projection_pairs
        for m in M_VALUES
    }
    assert len(shapes) == len(set(shapes))


def test_shape_suite_accepts_stable_short_model_names() -> None:
    assert load_shape_suite("qwen2.5-1.5b") == load_shape_suite(
        "Qwen/Qwen2.5-1.5B"
    )


def test_shape_suite_rejects_unknown_models() -> None:
    with pytest.raises(ValueError, match="Unknown shape suite"):
        load_shape_suite("Qwen/unknown")
