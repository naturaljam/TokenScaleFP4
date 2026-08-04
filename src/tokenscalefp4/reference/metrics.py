# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch


def assert_all_finite(value: torch.Tensor, *, name: str = "tensor") -> None:
    """Raise when a debug or evidence tensor contains NaN or infinity."""

    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains non-finite values")


def _metric_inputs(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if actual.numel() != expected.numel():
        raise ValueError("metric inputs must have the same number of elements")
    if actual.numel() == 0:
        raise ValueError("metric inputs must not be empty")
    assert_all_finite(actual, name="actual")
    assert_all_finite(expected, name="expected")
    return actual.detach().to(torch.float64).reshape(-1), expected.detach().to(
        torch.float64
    ).reshape(-1)


def cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Compute flattened cosine similarity with defined zero-vector behavior."""

    actual_fp64, expected_fp64 = _metric_inputs(actual, expected)
    actual_norm = float(actual_fp64.square().sum().sqrt().item())
    expected_norm = float(expected_fp64.square().sum().sqrt().item())
    if actual_norm == 0.0 or expected_norm == 0.0:
        return 1.0 if actual_norm == expected_norm else 0.0

    similarity = float(torch.dot(actual_fp64, expected_fp64).item())
    similarity /= actual_norm * expected_norm
    return min(1.0, max(-1.0, similarity))


def normalized_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Compute RMSE normalized by the root-mean-square reference magnitude."""

    actual_fp64, expected_fp64 = _metric_inputs(actual, expected)
    rmse = float(torch.sqrt(torch.mean((actual_fp64 - expected_fp64) ** 2)).item())
    reference_rms = float(torch.sqrt(torch.mean(expected_fp64**2)).item())
    if reference_rms == 0.0:
        return 0.0 if rmse == 0.0 else math.inf
    return rmse / reference_rms
