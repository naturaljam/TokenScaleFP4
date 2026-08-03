# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tokenscalefp4.integrations.patches import (
    PROJECTS,
    apply_patch_stack,
    read_base_revision,
)

CANONICAL_URLS = {
    "flashinfer": "https://github.com/flashinfer-ai/flashinfer.git",
    "vllm": "https://github.com/vllm-project/vllm.git",
}


def run_git(checkout: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )


def require_clean_checkout(project: str, checkout: Path) -> None:
    result = run_git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    if result.returncode != 0:
        raise RuntimeError(f"{project} checkout is not a Git working tree: {checkout}")
    if result.stdout:
        paths = [line[3:] for line in result.stdout.splitlines()]
        raise RuntimeError(
            f"Refusing dirty {project} checkout; modified paths: {', '.join(paths)}"
        )


def synchronize_checkout(
    project: str,
    checkout: Path,
    *,
    repository_url: str | None = None,
    revision: str | None = None,
) -> str:
    """Clone if needed, then fetch and detach exactly at an upstream revision."""
    if project not in PROJECTS:
        choices = ", ".join(PROJECTS)
        raise ValueError(f"Unknown upstream project {project!r}; expected one of {choices}")
    source_url = repository_url or CANONICAL_URLS[project]
    pinned_revision = revision or read_base_revision(project)

    if checkout.exists():
        require_clean_checkout(project, checkout)
    else:
        clone = subprocess.run(
            ["git", "clone", "--filter=blob:none", source_url, str(checkout)],
            check=False,
            capture_output=True,
            text=True,
        )
        if clone.returncode != 0:
            raise RuntimeError(f"Failed to clone {project}: {clone.stderr.strip()}")

    fetch = run_git(checkout, "fetch", "origin", pinned_revision)
    if fetch.returncode != 0:
        raise RuntimeError(f"Failed to fetch {project} revision: {fetch.stderr.strip()}")
    switch = run_git(checkout, "switch", "--detach", pinned_revision)
    if switch.returncode != 0:
        raise RuntimeError(f"Failed to detach {project} at revision: {switch.stderr.strip()}")
    head = run_git(checkout, "rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != pinned_revision:
        raise RuntimeError(f"{project} checkout did not reach the pinned revision")
    return pinned_revision


def synchronize(project: str, checkout_only: bool) -> tuple[str, list[Path]]:
    checkout = PROJECT_ROOT / ".third_party" / project
    revision = synchronize_checkout(project, checkout)
    applied = [] if checkout_only else apply_patch_stack(project, checkout)
    return revision, applied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce pinned upstream checkouts")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--project", choices=PROJECTS)
    selection.add_argument("--all", action="store_true")
    parser.add_argument(
        "--checkout-only",
        action="store_true",
        help="clone, fetch, and detach without applying local patches",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    projects = PROJECTS if args.all else (args.project,)
    for project in projects:
        revision, applied = synchronize(project, args.checkout_only)
        print(f"{project}: {revision}")
        for patch in applied:
            print(f"{project}: applied {patch.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
