# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMPUTE_CAPABILITY = re.compile(r"^[0-9]+\.[0-9]+$")
GateValue: TypeAlias = int | float | str


class EvidenceValidationError(ValueError):
    """Raised when an evidence document violates the frozen public contract."""


def _object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"{path} must be an object")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise EvidenceValidationError(f"{path} must use string keys")
    return cast(dict[str, Any], mapping)


def _required(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise EvidenceValidationError(f"{path}.{key} is required")
    return mapping[key]


def _only(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unexpected = sorted(set(mapping) - allowed)
    if unexpected:
        raise EvidenceValidationError(
            f"{path} contains unsupported fields: {', '.join(unexpected)}"
        )


def _string(value: object, path: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceValidationError(f"{path} must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise EvidenceValidationError(f"{path} has an invalid format")
    return value


def _gate_value(value: object, path: str) -> GateValue:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise EvidenceValidationError(f"{path} must be a number or string")
    return value


@dataclass(frozen=True, slots=True)
class UpstreamRevisions:
    flashinfer_sha: str
    vllm_sha: str


@dataclass(frozen=True, slots=True)
class EnvironmentRecord:
    gpu_name: str
    compute_capability: str
    cuda_version: str
    pytorch_version: str
    flashinfer_version: str
    vllm_version: str


@dataclass(frozen=True, slots=True)
class RawSamples:
    reference: str
    sha256: str


@dataclass(frozen=True, slots=True)
class GateRecord:
    name: str
    threshold: GateValue
    observed: GateValue
    passed: bool


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    schema_version: int
    upstream: UpstreamRevisions
    environment: EnvironmentRecord
    seed: int
    command: str
    raw_samples: RawSamples
    gate: GateRecord

    @classmethod
    def from_json(cls, path: Path) -> EvidenceRecord:
        try:
            payload = _object(json.loads(path.read_text(encoding="utf-8")), "evidence")
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise EvidenceValidationError(f"Invalid evidence JSON: {error}") from error

        root_fields = {
            "schema_version",
            "upstream",
            "environment",
            "seed",
            "command",
            "raw_samples",
            "gate",
        }
        _only(payload, root_fields, "evidence")

        schema_version = _required(payload, "schema_version", "evidence")
        if isinstance(schema_version, bool) or schema_version != 1:
            raise EvidenceValidationError("evidence.schema_version must equal 1")

        upstream_data = _object(
            _required(payload, "upstream", "evidence"), "evidence.upstream"
        )
        _only(upstream_data, {"flashinfer_sha", "vllm_sha"}, "evidence.upstream")
        upstream = UpstreamRevisions(
            flashinfer_sha=_string(
                _required(upstream_data, "flashinfer_sha", "evidence.upstream"),
                "evidence.upstream.flashinfer_sha",
                FULL_SHA,
            ),
            vllm_sha=_string(
                _required(upstream_data, "vllm_sha", "evidence.upstream"),
                "evidence.upstream.vllm_sha",
                FULL_SHA,
            ),
        )

        environment_data = _object(
            _required(payload, "environment", "evidence"), "evidence.environment"
        )
        environment_fields = {
            "gpu_name",
            "compute_capability",
            "cuda_version",
            "pytorch_version",
            "flashinfer_version",
            "vllm_version",
        }
        _only(environment_data, environment_fields, "evidence.environment")
        environment = EnvironmentRecord(
            gpu_name=_string(
                _required(environment_data, "gpu_name", "evidence.environment"),
                "evidence.environment.gpu_name",
            ),
            compute_capability=_string(
                _required(
                    environment_data, "compute_capability", "evidence.environment"
                ),
                "evidence.environment.compute_capability",
                COMPUTE_CAPABILITY,
            ),
            cuda_version=_string(
                _required(environment_data, "cuda_version", "evidence.environment"),
                "evidence.environment.cuda_version",
            ),
            pytorch_version=_string(
                _required(environment_data, "pytorch_version", "evidence.environment"),
                "evidence.environment.pytorch_version",
            ),
            flashinfer_version=_string(
                _required(
                    environment_data, "flashinfer_version", "evidence.environment"
                ),
                "evidence.environment.flashinfer_version",
            ),
            vllm_version=_string(
                _required(environment_data, "vllm_version", "evidence.environment"),
                "evidence.environment.vllm_version",
            ),
        )

        seed = _required(payload, "seed", "evidence")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise EvidenceValidationError("evidence.seed must be an integer")

        raw_data = _object(
            _required(payload, "raw_samples", "evidence"), "evidence.raw_samples"
        )
        _only(raw_data, {"reference", "sha256"}, "evidence.raw_samples")
        raw_samples = RawSamples(
            reference=_string(
                _required(raw_data, "reference", "evidence.raw_samples"),
                "evidence.raw_samples.reference",
            ),
            sha256=_string(
                _required(raw_data, "sha256", "evidence.raw_samples"),
                "evidence.raw_samples.sha256",
                SHA256,
            ),
        )

        gate_data = _object(_required(payload, "gate", "evidence"), "evidence.gate")
        _only(gate_data, {"name", "threshold", "observed", "passed"}, "evidence.gate")
        passed = _required(gate_data, "passed", "evidence.gate")
        if not isinstance(passed, bool):
            raise EvidenceValidationError("evidence.gate.passed must be a boolean")
        gate = GateRecord(
            name=_string(
                _required(gate_data, "name", "evidence.gate"), "evidence.gate.name"
            ),
            threshold=_gate_value(
                _required(gate_data, "threshold", "evidence.gate"),
                "evidence.gate.threshold",
            ),
            observed=_gate_value(
                _required(gate_data, "observed", "evidence.gate"),
                "evidence.gate.observed",
            ),
            passed=passed,
        )

        return cls(
            schema_version=1,
            upstream=upstream,
            environment=environment,
            seed=seed,
            command=_string(
                _required(payload, "command", "evidence"), "evidence.command"
            ),
            raw_samples=raw_samples,
            gate=gate,
        )
