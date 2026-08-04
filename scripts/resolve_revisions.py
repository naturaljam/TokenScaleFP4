# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, NamedTuple, Protocol, cast
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "evals" / "quality.json"
DEFAULT_RESULTS_DIR = REPO_ROOT / "reports"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
HF_API_ROOT = "https://huggingface.co/api"


class HttpResponse(NamedTuple):
    body: bytes
    headers: Mapping[str, str]


class HttpClient(Protocol):
    def get(self, url: str) -> HttpResponse: ...


class PublicHttpClient:
    def get(self, url: str) -> HttpResponse:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "TokenScaleFP4-revision-resolver/1",
            },
        )
        with urlopen(request, timeout=30) as response:
            return HttpResponse(
                body=response.read(),
                headers=dict(response.headers.items()),
            )


def _object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be an object")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise TypeError(f"{path} must use string keys")
    return cast(dict[str, Any], mapping)


def resolve_revision(
    *,
    repository_type: str,
    repository_id: str,
    revision: str = "main",
    http_client: HttpClient,
) -> str:
    if FULL_SHA.fullmatch(revision):
        return revision
    if repository_type not in {"models", "datasets"}:
        raise ValueError("repository_type must be 'models' or 'datasets'")
    if not repository_id or not revision:
        raise ValueError("repository_id and revision must be non-empty")

    encoded_id = quote(repository_id, safe="/")
    encoded_revision = quote(revision, safe="")
    url = f"{HF_API_ROOT}/{repository_type}/{encoded_id}/revision/{encoded_revision}"
    response = http_client.get(url)
    decoded: object = json.loads(response.body)
    payload = _object(decoded, f"Hugging Face response for {repository_id}")
    sha = payload.get("sha")
    if not isinstance(sha, str) or FULL_SHA.fullmatch(sha) is None:
        raise ValueError(
            f"Hugging Face returned no immutable SHA for {repository_id}@{revision}"
        )
    return sha


def _has_evaluation_results(results_dir: Path) -> bool:
    return any(
        path.is_file()
        for directory in ("raw", "local", "generated")
        for path in (results_dir / directory).glob("**/*")
    )


def _resolve_entry(
    entry: dict[str, Any],
    *,
    id_key: str,
    repository_type: str,
    http_client: HttpClient,
    refresh: bool,
) -> str:
    repository_id = entry.get(id_key)
    if not isinstance(repository_id, str) or not repository_id:
        raise ValueError(f"Quality configuration field {id_key!r} must be a string")
    current = entry.get("revision", "main")
    if not isinstance(current, str):
        raise TypeError(f"Revision for {repository_id} must be a string")
    if FULL_SHA.fullmatch(current) and not refresh:
        return current
    requested = "main" if refresh and FULL_SHA.fullmatch(current) else current
    return resolve_revision(
        repository_type=repository_type,
        repository_id=repository_id,
        revision=requested,
        http_client=http_client,
    )


def resolve_config(
    config_path: Path,
    *,
    http_client: HttpClient,
    refresh: bool = False,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> bool:
    if refresh and _has_evaluation_results(results_dir):
        raise RuntimeError("Cannot refresh revisions: evaluation evidence already exists")

    original = config_path.read_bytes()
    decoded: object = json.loads(original)
    source = _object(decoded, "Quality configuration")
    if source.get("schema_version") != 1:
        raise ValueError("Quality configuration must use schema_version 1")
    updated: dict[str, Any] = deepcopy(source)

    for section_name in ("perplexity", "gsm8k"):
        section = _object(
            updated.get(section_name),
            f"Quality configuration field {section_name!r}",
        )
        section["revision"] = _resolve_entry(
            section,
            id_key="dataset",
            repository_type="datasets",
            http_client=http_client,
            refresh=refresh,
        )

    for model_key in ("local_model", "final_model"):
        repository_id = updated.get(model_key)
        if not isinstance(repository_id, str) or not repository_id:
            raise ValueError(f"Quality configuration field {model_key!r} must be a string")
        revision_key = f"{model_key}_revision"
        current = updated.get(revision_key, "main")
        if not isinstance(current, str):
            raise TypeError(
                f"Quality configuration field {revision_key!r} must be a string"
            )
        if FULL_SHA.fullmatch(current) and not refresh:
            resolved = current
        else:
            requested = "main" if refresh and FULL_SHA.fullmatch(current) else current
            resolved = resolve_revision(
                repository_type="models",
                repository_id=repository_id,
                revision=requested,
                http_client=http_client,
            )
        updated[revision_key] = resolved

    if updated == source:
        return False

    serialized = (json.dumps(updated, indent=2, ensure_ascii=True) + "\n").encode()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=config_path.parent, delete=False
        ) as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, config_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve quality datasets and models to immutable Hugging Face SHAs"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    changed = resolve_config(
        args.config,
        http_client=PublicHttpClient(),
        refresh=args.refresh,
    )
    print("updated" if changed else "already pinned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
