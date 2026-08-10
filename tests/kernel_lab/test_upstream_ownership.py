# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parents[2]
MODULE = runpy.run_path(str(REPO_ROOT / "scripts" / "check_upstream_ownership.py"))
HttpResponse = MODULE["HttpResponse"]
check_ownership = MODULE["check_ownership"]


class FixtureHttpClient:
    def __init__(self, responses: dict[str, object], *, error: Exception | None = None):
        self.responses = responses
        self.error = error
        self.urls: list[str] = []

    def get(self, url: str) -> HttpResponse:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        payload = self.responses.get(url)
        if payload is None:
            raise AssertionError(f"unexpected URL: {url}")
        return HttpResponse(json.dumps(payload).encode(), {})


ROOT = "https://api.github.com/repos/flashinfer-ai/flashinfer"


def fixture(issue: dict[str, Any], *, timeline: list[Any] | None = None, searches: list[Any] | None = None) -> FixtureHttpClient:
    return FixtureHttpClient(
        {
            f"{ROOT}/issues/4300": issue,
            f"{ROOT}/issues/4300/timeline": timeline or [],
            "https://api.github.com/search/issues?q=repo%3Aflashinfer-ai%2Fflashinfer%20in%3Atitle%20%22Support%20per-token%20activation%20scales%20in%20mm_fp4%22": searches or [],
            "https://api.github.com/search/code?q=repo%3Aflashinfer-ai%2Fflashinfer%20mm_fp4%20per_token_scale": {"items": []},
        }
    )


def issue(*, assignee: str | None = None, comments: int = 0) -> dict[str, Any]:
    return {
        "number": 4300,
        "state": "open",
        "assignee": None if assignee is None else {"login": assignee},
        "comments": comments,
        "html_url": "https://github.com/flashinfer-ai/flashinfer/issues/4300",
        "title": "Support per-token activation scales in mm_fp4",
    }


def test_open_unowned_is_clear() -> None:
    result = check_ownership(4300, http_client=fixture(issue()))
    assert result["issue_state"] == "open"
    assert result["assignee"] is None
    assert result["stop_condition"] is None


def test_assigned_issue_reports_coordination_gate() -> None:
    result = check_ownership(4300, http_client=fixture(issue(assignee="dhiraj113")))
    assert result["assignee"] == "dhiraj113"
    assert result["ownership_gate"] == "assigned"


def test_linked_draft_pr_stops() -> None:
    timeline = [{"event": "cross-referenced", "source": {"issue": {"number": 501, "state": "open", "html_url": "https://github.com/flashinfer-ai/flashinfer/pull/501", "pull_request": {"html_url": "https://github.com/flashinfer-ai/flashinfer/pull/501"}}}}]
    result = check_ownership(4300, http_client=fixture(issue(), timeline=timeline))
    assert result["stop_condition"] == "STOP_CONDITION_1"
    assert result["pull_requests"][0]["state"] == "draft"


def test_merged_equivalent_search_stops() -> None:
    searches = [{"number": 502, "state": "closed", "pull_request": {"merged_at": "2026-08-05T00:00:00Z"}, "html_url": "https://github.com/flashinfer-ai/flashinfer/pull/502", "title": "mm_fp4 per-token activation scales"}]
    result = check_ownership(4300, http_client=fixture(issue(), searches=searches))
    assert result["stop_condition"] == "STOP_CONDITION_1"


def test_rate_limit_is_sanitized() -> None:
    result = check_ownership(4300, http_client=FixtureHttpClient({}, error=RuntimeError("API rate limit exceeded token=secret")))
    assert result["error"] == "api_request_failed"
    assert "secret" not in json.dumps(result)


def test_malformed_response_is_rejected() -> None:
    client = fixture(issue())
    client.responses[f"{ROOT}/issues/4300"] = {"state": "open"}
    with pytest.raises(ValueError, match="malformed"):
        check_ownership(4300, http_client=client)
