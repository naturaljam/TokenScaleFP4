# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

API_ROOT = "https://api.github.com"
REPOSITORY = "flashinfer-ai/flashinfer"
ISSUE_TITLE = "Support per-token activation scales in mm_fp4"
CODE_QUERY = "mm_fp4 per_token_scale"


class HttpResponse(NamedTuple):
    body: bytes
    headers: dict[str, str]


class HttpClient(Protocol):
    def get(self, url: str) -> HttpResponse: ...


class PublicHttpClient:
    def get(self, url: str) -> HttpResponse:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "TokenScaleFP4-ownership/1"}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=30) as response:
                return HttpResponse(response.read(), dict(response.headers.items()))
        except (HTTPError, URLError) as exc:
            raise RuntimeError("GitHub API request failed") from exc


def _decode(body: bytes, label: str) -> Any:
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed {label} response") from exc


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"malformed {label} response")  # noqa: TRY004
    return value


def _items(value: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, dict) and isinstance(value.get("items"), list):
        raw = value["items"]
    else:
        raise ValueError(f"malformed {label} response")  # noqa: TRY004
    return [_object(item, label) for item in raw]


def _pr_record(item: dict[str, Any]) -> dict[str, Any]:
    url = item.get("html_url")
    if not isinstance(url, str) or not url.startswith("https://github.com/"):
        raise ValueError("malformed pull request response")
    merged_at = item.get("merged_at")
    state = "merged" if merged_at else ("draft" if item.get("draft", True) else str(item.get("state", "open")))
    return {"url": url, "state": state}


def check_ownership(issue_number: int, *, http_client: HttpClient) -> dict[str, Any]:
    try:
        return _check_ownership(issue_number, http_client=http_client)
    except (OSError, RuntimeError):
        return {"issue_number": issue_number, "error": "api_request_failed"}


def _check_ownership(issue_number: int, *, http_client: HttpClient) -> dict[str, Any]:
    if issue_number <= 0:
        raise ValueError("issue number must be positive")
    base = f"{API_ROOT}/repos/{REPOSITORY}/issues/{issue_number}"
    issue = _object(_decode(http_client.get(base).body, "issue"), "issue")
    if issue.get("number") != issue_number or issue.get("state") not in {"open", "closed"}:
        raise ValueError("malformed issue response")
    assignee = issue.get("assignee")
    assignee_login = None
    if assignee is not None:
        if not isinstance(assignee, dict) or not isinstance(assignee.get("login"), str):
            raise ValueError("malformed issue response")
        assignee_login = assignee["login"]

    timeline = _items(_decode(http_client.get(f"{base}/timeline").body, "timeline"), "timeline")
    prs: list[dict[str, str]] = []
    for event in timeline:
        source = event.get("source")
        source_issue = source.get("issue") if isinstance(source, dict) else None
        if isinstance(source_issue, dict) and isinstance(source_issue.get("pull_request"), dict):
            try:
                prs.append(_pr_record(source_issue))
            except ValueError:
                continue

    title_query = quote(f'repo:{REPOSITORY} in:title "{ISSUE_TITLE}"', safe="")
    search_issue_url = f"{API_ROOT}/search/issues?q={title_query}"
    search_code_url = f"{API_ROOT}/search/code?q=repo%3A{REPOSITORY.replace('/', '%2F')}%20{CODE_QUERY.replace(' ', '%20')}"
    issue_hits = _items(_decode(http_client.get(search_issue_url).body, "issue search"), "issue search")
    code_hits = _items(_decode(http_client.get(search_code_url).body, "code search"), "code search")
    for hit in issue_hits:
        if hit.get("number") == issue_number:
            continue
        title = hit.get("title")
        if isinstance(title, str) and re.search(
            r"per.?token.*scale|activation.*scale", title, re.IGNORECASE
        ):
            try:
                prs.append(_pr_record(hit))
            except ValueError:
                pass

    stop = "STOP_CONDITION_1" if any(pr["state"] in {"draft", "open", "merged", "closed"} for pr in prs) else None
    return {
        "issue_number": issue_number,
        "issue_state": issue["state"],
        "assignee": assignee_login,
        "comments": issue.get("comments", 0) if isinstance(issue.get("comments", 0), int) else 0,
        "issue_url": issue.get("html_url") if isinstance(issue.get("html_url"), str) else f"https://github.com/{REPOSITORY}/issues/{issue_number}",
        "pull_requests": prs,
        "exact_title_results": [{"url": x.get("html_url")} for x in issue_hits if isinstance(x.get("html_url"), str)],
        "code_search_results": [{"url": x.get("html_url")} for x in code_hits if isinstance(x.get("html_url"), str)],
        "ownership_gate": "assigned" if assignee_login else "unowned",
        "stop_condition": stop,
        "checked_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only FlashInfer upstream ownership check")
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exit_code: int | None = None
    try:
        result = check_ownership(args.issue, http_client=PublicHttpClient())
    except ValueError:
        result = {"issue_number": args.issue, "error": "malformed_response"}
        print("malformed_response")
        exit_code = 1
    if result.get("error") and exit_code is None:
        print("api_request_failed")
        exit_code = 1
    elif exit_code is None:
        exit_code = 2 if result["stop_condition"] else 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if exit_code:
        return exit_code
    if result["stop_condition"]:
        print(result["stop_condition"])
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
