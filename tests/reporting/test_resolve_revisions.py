# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parents[2]
RESOLVER = runpy.run_path(str(REPO_ROOT / "scripts" / "resolve_revisions.py"))
HttpResponse = RESOLVER["HttpResponse"]
resolve_config = RESOLVER["resolve_config"]
resolve_revision = RESOLVER["resolve_revision"]

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "89abcdef0123456789abcdef0123456789abcdef"


class FixtureHttpClient:
    def __init__(
        self,
        revisions: dict[str, str] | None = None,
        *,
        fail_after: int | None = None,
    ) -> None:
        self.revisions = revisions or {}
        self.fail_after = fail_after
        self.urls: list[str] = []

    def get(self, url: str) -> Any:
        self.urls.append(url)
        if self.fail_after is not None and len(self.urls) > self.fail_after:
            raise OSError("fixture network unavailable")
        repository_id = url.split("/revision/", maxsplit=1)[0].split("/api/", maxsplit=1)[1]
        sha = self.revisions.get(repository_id, SHA)
        return HttpResponse(
            body=json.dumps({"sha": sha, "ignored": "body-secret"}).encode(),
            headers={"authorization": "Bearer header-secret", "etag": "private"},
        )


def quality_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "perplexity": {
            "dataset": "Salesforce/wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "test",
            "seed": 20260803,
            "max_relative_increase": 0.05,
        },
        "gsm8k": {
            "dataset": "openai/gsm8k",
            "subset": "main",
            "split": "test",
            "seed": 20260803,
            "max_absolute_drop": 0.02,
            "decoding": "greedy",
        },
        "finite_required": True,
        "local_model": "Qwen/Qwen2.5-1.5B",
        "final_model": "Qwen/Qwen2.5-7B",
    }


def write_config(tmp_path: Path, payload: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "quality.json"
    path.write_text(
        json.dumps(payload or quality_config(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def test_symbolic_revision_resolves_through_public_api_shape() -> None:
    client = FixtureHttpClient()

    revision = resolve_revision(
        repository_type="models",
        repository_id="Qwen/Qwen2.5-1.5B",
        revision="main",
        http_client=client,
    )

    assert revision == SHA
    assert client.urls == [
        "https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B/revision/main"
    ]


def test_full_sha_is_immutable_without_an_http_request() -> None:
    client = FixtureHttpClient(fail_after=0)

    assert (
        resolve_revision(
            repository_type="datasets",
            repository_id="openai/gsm8k",
            revision=SHA,
            http_client=client,
        )
        == SHA
    )
    assert client.urls == []


def test_config_resolution_writes_only_revision_fields(tmp_path: Path) -> None:
    path = write_config(tmp_path)
    client = FixtureHttpClient()

    changed = resolve_config(path, http_client=client)
    serialized = path.read_text(encoding="utf-8")
    payload = json.loads(serialized)

    assert changed is True
    assert payload["perplexity"]["revision"] == SHA
    assert payload["gsm8k"]["revision"] == SHA
    assert payload["local_model_revision"] == SHA
    assert payload["final_model_revision"] == SHA
    assert "header-secret" not in serialized
    assert "body-secret" not in serialized
    assert "authorization" not in serialized.lower()
    assert "headers" not in serialized.lower()


def test_network_failure_leaves_config_byte_for_byte_unchanged(tmp_path: Path) -> None:
    path = write_config(tmp_path)
    original = path.read_bytes()

    with pytest.raises(OSError, match="network unavailable"):
        resolve_config(path, http_client=FixtureHttpClient(fail_after=1))

    assert path.read_bytes() == original


def test_existing_revisions_are_not_replaced_without_refresh(tmp_path: Path) -> None:
    payload = quality_config()
    payload["perplexity"]["revision"] = SHA
    payload["gsm8k"]["revision"] = SHA
    payload["local_model_revision"] = SHA
    payload["final_model_revision"] = SHA
    path = write_config(tmp_path, payload)
    original = path.read_bytes()
    client = FixtureHttpClient(
        {
            "datasets/Salesforce/wikitext": OTHER_SHA,
            "datasets/openai/gsm8k": OTHER_SHA,
            "models/Qwen/Qwen2.5-1.5B": OTHER_SHA,
            "models/Qwen/Qwen2.5-7B": OTHER_SHA,
        }
    )

    changed = resolve_config(path, http_client=client)

    assert changed is False
    assert client.urls == []
    assert path.read_bytes() == original


def test_refresh_is_rejected_after_evidence_exists(tmp_path: Path) -> None:
    path = write_config(tmp_path)
    results_dir = tmp_path / "reports"
    (results_dir / "local").mkdir(parents=True)
    (results_dir / "local" / "quality.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="evaluation evidence already exists"):
        resolve_config(
            path,
            http_client=FixtureHttpClient(),
            refresh=True,
            results_dir=results_dir,
        )
