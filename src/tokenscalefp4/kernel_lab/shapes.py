# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import cast

SHAPE_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs" / "shapes" / "qwen2_5.json"
)


@dataclass(frozen=True, order=True, slots=True)
class GemmShape:
    m: int
    n: int
    k: int

    def __post_init__(self) -> None:
        if self.m <= 0 or self.n <= 0 or self.k <= 0:
            raise ValueError("GEMM dimensions must be positive")


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be an object")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise TypeError(f"{path} must use string keys")
    return cast(dict[str, object], mapping)


def _load_config() -> dict[str, object]:
    if SHAPE_CONFIG.is_file():
        config_text = SHAPE_CONFIG.read_text(encoding="utf-8")
        config_location = str(SHAPE_CONFIG)
    else:
        packaged_config = files("tokenscalefp4").joinpath(
            "configs", "shapes", "qwen2_5.json"
        )
        config_text = packaged_config.read_text(encoding="utf-8")
        config_location = str(packaged_config)
    decoded: object = json.loads(config_text)
    payload = _object(decoded, config_location)
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported shape configuration: {SHAPE_CONFIG}")
    return payload


def load_shape_suite(model: str) -> tuple[GemmShape, ...]:
    """Expand a configured model's unique projection dimensions over M values."""
    config = _load_config()
    models = _object(config.get("models"), f"{SHAPE_CONFIG}: models")

    selected: object | None = models.get(model)
    if selected is None:
        normalized = model.casefold()
        for model_id, candidate_value in models.items():
            candidate = _object(
                candidate_value, f"{SHAPE_CONFIG}: models.{model_id}"
            )
            aliases_value = candidate.get("aliases", [])
            if not isinstance(aliases_value, list):
                raise TypeError(f"Aliases for {model_id} must be an array")
            aliases = cast(list[object], aliases_value)
            names = [model_id, *aliases]
            if any(isinstance(name, str) and name.casefold() == normalized for name in names):
                selected = candidate
                break
    if selected is None:
        known = ", ".join(sorted(models))
        raise ValueError(f"Unknown shape suite {model!r}; expected one of: {known}")
    selected_model = _object(selected, f"{SHAPE_CONFIG}: models.{model}")

    m_values = config.get("m_values")
    projections = selected_model.get("projections")
    if not isinstance(m_values, list) or not isinstance(projections, list):
        raise TypeError(f"Shape entries must be arrays in {SHAPE_CONFIG}")

    shapes: dict[GemmShape, None] = {}
    for m in cast(list[object], m_values):
        for projection_value in cast(list[object], projections):
            projection = _object(
                projection_value, f"{SHAPE_CONFIG}: projection"
            )
            n = projection.get("n")
            k = projection.get("k")
            if not isinstance(m, int) or not isinstance(n, int) or not isinstance(k, int):
                raise TypeError(f"Shape dimensions must be integers in {SHAPE_CONFIG}")
            shapes[GemmShape(m=m, n=n, k=k)] = None
    return tuple(shapes)
