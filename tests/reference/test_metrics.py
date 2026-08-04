# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch

from tokenscalefp4.reference import (
    assert_all_finite,
    cosine_similarity,
    normalized_rmse,
)


def test_cosine_similarity_flattens_inputs() -> None:
    actual = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    expected = torch.tensor([1.0, 0.0, 0.0, 1.0])

    assert cosine_similarity(actual, expected) == pytest.approx(1.0)


def test_cosine_similarity_handles_zero_vectors() -> None:
    zeros = torch.zeros(4)

    assert cosine_similarity(zeros, zeros) == 1.0
    assert cosine_similarity(zeros, torch.ones(4)) == 0.0


def test_normalized_rmse_uses_reference_rms() -> None:
    actual = torch.tensor([2.0, 0.0])
    expected = torch.tensor([1.0, 1.0])

    assert normalized_rmse(actual, expected) == pytest.approx(1.0)


def test_normalized_rmse_handles_a_zero_reference() -> None:
    zeros = torch.zeros(4)

    assert normalized_rmse(zeros, zeros) == 0.0
    assert math.isinf(normalized_rmse(torch.ones(4), zeros))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_assert_all_finite_names_non_finite_tensors(bad_value: float) -> None:
    with pytest.raises(ValueError, match="kernel output contains non-finite values"):
        assert_all_finite(torch.tensor([1.0, bad_value]), name="kernel output")


def test_metrics_reject_shape_mismatches() -> None:
    with pytest.raises(ValueError, match="same number of elements"):
        cosine_similarity(torch.ones(2), torch.ones(3))

    with pytest.raises(ValueError, match="same number of elements"):
        normalized_rmse(torch.ones(2), torch.ones(3))
