# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema.validators import Draft202012Validator

from tokenscalefp4.reporting.schema import EvidenceRecord, EvidenceValidationError

REPO_ROOT = Path(__file__).parents[2]


def valid_evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "upstream": {
            "flashinfer_sha": "1" * 40,
            "vllm_sha": "2" * 40,
        },
        "environment": {
            "gpu_name": "NVIDIA GeForce RTX 5070 Laptop GPU",
            "compute_capability": "12.0",
            "cuda_version": "13.0",
            "pytorch_version": "2.11.0",
            "flashinfer_version": "0.6.4",
            "vllm_version": "0.14.0",
        },
        "seed": 20260803,
        "command": "python scripts/run_quality_eval.py --mode bf16",
        "raw_samples": {
            "reference": "external://quality/qwen1.5b-bf16.json",
            "sha256": "a" * 64,
        },
        "gate": {
            "name": "perplexity_relative_increase",
            "threshold": 0.05,
            "observed": 0.03,
            "passed": True,
        },
    }


def write_record(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def remove_path(payload: dict[str, Any], dotted_path: str) -> None:
    target = payload
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        target = target[part]
    del target[parts[-1]]


def test_evidence_schema_is_a_valid_draft_2020_12_schema() -> None:
    schema = json.loads(
        (REPO_ROOT / "schemas" / "evidence.schema.json").read_text(
            encoding="utf-8"
        )
    )

    Draft202012Validator.check_schema(schema)


def test_evidence_record_loads_all_frozen_metadata(tmp_path: Path) -> None:
    record = EvidenceRecord.from_json(write_record(tmp_path, valid_evidence()))

    assert record.schema_version == 1
    assert record.upstream.flashinfer_sha == "1" * 40
    assert record.environment.compute_capability == "12.0"
    assert record.raw_samples.sha256 == "a" * 64
    assert record.gate.passed is True


@pytest.mark.parametrize(
    "missing",
    [
        "schema_version",
        "upstream.flashinfer_sha",
        "upstream.vllm_sha",
        "environment.gpu_name",
        "environment.compute_capability",
        "environment.cuda_version",
        "environment.pytorch_version",
        "environment.flashinfer_version",
        "environment.vllm_version",
        "seed",
        "command",
        "raw_samples.reference",
        "raw_samples.sha256",
        "gate.name",
        "gate.threshold",
        "gate.observed",
        "gate.passed",
    ],
)
def test_evidence_record_rejects_missing_required_metadata(
    tmp_path: Path, missing: str
) -> None:
    payload = copy.deepcopy(valid_evidence())
    remove_path(payload, missing)

    with pytest.raises(EvidenceValidationError, match="required"):
        EvidenceRecord.from_json(write_record(tmp_path, payload))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("upstream.flashinfer_sha", "main"),
        ("raw_samples.sha256", "not-a-checksum"),
        ("environment.compute_capability", "sm120"),
    ],
)
def test_evidence_record_rejects_malformed_identifiers(
    tmp_path: Path, path: str, value: str
) -> None:
    payload = valid_evidence()
    target = payload
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value

    with pytest.raises(EvidenceValidationError):
        EvidenceRecord.from_json(write_record(tmp_path, payload))
