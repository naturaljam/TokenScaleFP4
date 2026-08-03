# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from tokenscalefp4.integrations import patches
from tokenscalefp4.integrations.patches import (
    apply_patch_stack,
    discover_patches,
    read_base_revision,
)

REPO_ROOT = Path(__file__).parents[2]
SYNC_UPSTREAMS = runpy.run_path(str(REPO_ROOT / "scripts" / "sync_upstreams.py"))
CANONICAL_URLS = SYNC_UPSTREAMS["CANONICAL_URLS"]
synchronize_checkout = SYNC_UPSTREAMS["synchronize_checkout"]


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def write_file(repo: Path, relative_path: str, content: str) -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def commit(repo: Path, message: str) -> str:
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-qm", message)
    return run_git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture
def upstream_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "upstream-source"
    repo.mkdir()
    run_git(repo, "init", "-q")
    run_git(repo, "config", "user.email", "test@example.invalid")
    run_git(repo, "config", "user.name", "Test User")
    write_file(repo, "tracked.txt", "base\n")
    commit(repo, "base")
    return repo


def test_pinned_revisions_are_full_sha() -> None:
    assert read_base_revision("flashinfer") == (
        "08ddfbcd2e89b2f4b68391825817909e30d445e2"
    )
    assert read_base_revision("vllm") == (
        "0a6446005d51c9e6bfa09352f7f288ddeff17c77"
    )


def test_sync_uses_canonical_upstream_urls() -> None:
    assert CANONICAL_URLS == {
        "flashinfer": "https://github.com/flashinfer-ai/flashinfer.git",
        "vllm": "https://github.com/vllm-project/vllm.git",
    }


def test_patch_stack_is_lexically_ordered(tmp_path: Path) -> None:
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    for name in ("0002-b.patch", "0001-a.patch"):
        (patch_dir / name).write_text("", encoding="utf-8")

    assert discover_patches(patch_dir) == [
        patch_dir / "0001-a.patch",
        patch_dir / "0002-b.patch",
    ]


def test_synchronize_checkout_clones_clean_detached_revision(
    upstream_repo: Path, tmp_path: Path
) -> None:
    revision = run_git(upstream_repo, "rev-parse", "HEAD").stdout.strip()
    checkout = tmp_path / "checkout"

    synchronize_checkout(
        "flashinfer",
        checkout,
        repository_url=str(upstream_repo),
        revision=revision,
    )

    assert run_git(checkout, "rev-parse", "HEAD").stdout.strip() == revision
    detached_head = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )
    assert detached_head.returncode == 1
    assert run_git(checkout, "status", "--porcelain").stdout == ""


def test_apply_patch_stack_rejects_dirty_checkout_with_project_and_paths(
    upstream_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    run_git(tmp_path, "clone", "-q", str(upstream_repo), str(checkout))
    write_file(checkout, "tracked.txt", "modified\n")
    monkeypatch.setattr(patches, "REPO_ROOT", tmp_path)
    (tmp_path / "upstream" / "flashinfer" / "patches").mkdir(parents=True)

    with pytest.raises(RuntimeError, match=r"flashinfer.*tracked\.txt"):
        apply_patch_stack("flashinfer", checkout)


def test_apply_patch_stack_applies_patches_in_lexical_order(
    upstream_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_revision = run_git(upstream_repo, "rev-parse", "HEAD").stdout.strip()
    write_file(upstream_repo, "first.txt", "first\n")
    commit(upstream_repo, "first")
    write_file(upstream_repo, "second.txt", "second\n")
    commit(upstream_repo, "second")

    patch_dir = tmp_path / "upstream" / "flashinfer" / "patches"
    patch_dir.mkdir(parents=True)
    patches_to_export = run_git(
        upstream_repo,
        "format-patch",
        "--output-directory",
        str(patch_dir),
        f"{base_revision}..HEAD",
    )
    assert patches_to_export.returncode == 0
    patch_names = sorted(path.name for path in patch_dir.glob("*.patch"))
    temporary_name = patch_dir / "temporary.patch"
    (patch_dir / patch_names[0]).rename(temporary_name)
    (patch_dir / patch_names[1]).rename(patch_dir / "0001-first.patch")
    temporary_name.rename(patch_dir / "0002-second.patch")

    checkout = tmp_path / "checkout"
    run_git(tmp_path, "clone", "-q", str(upstream_repo), str(checkout))
    run_git(checkout, "reset", "--hard", "-q", base_revision)
    monkeypatch.setattr(patches, "REPO_ROOT", tmp_path)

    applied = apply_patch_stack("flashinfer", checkout)

    assert [path.name for path in applied] == [
        "0001-first.patch",
        "0002-second.patch",
    ]
    assert (checkout / "first.txt").read_text(encoding="utf-8") == "first\n"
    assert (checkout / "second.txt").read_text(encoding="utf-8") == "second\n"
    assert run_git(checkout, "status", "--porcelain").stdout == ""


def test_failing_patch_stack_aborts_git_am(
    upstream_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_revision = run_git(upstream_repo, "rev-parse", "HEAD").stdout.strip()
    write_file(upstream_repo, "temporary.txt", "must be rolled back\n")
    commit(upstream_repo, "temporary change")

    checkout = tmp_path / "checkout"
    run_git(tmp_path, "clone", "-q", str(upstream_repo), str(checkout))
    patch_dir = tmp_path / "upstream" / "flashinfer" / "patches"
    patch_dir.mkdir(parents=True)
    run_git(
        upstream_repo,
        "format-patch",
        "--output-directory",
        str(patch_dir),
        f"{base_revision}..HEAD",
    )
    (patch_dir / "0001-broken.patch").write_text(
        "This is not a mail-formatted patch.\n", encoding="utf-8"
    )
    (patch_dir / "0001-broken.patch").rename(patch_dir / "0002-broken.patch")
    run_git(checkout, "reset", "--hard", "-q", base_revision)
    monkeypatch.setattr(patches, "REPO_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match=r"flashinfer.*0001-.*\.patch"):
        apply_patch_stack("flashinfer", checkout)

    assert run_git(checkout, "status", "--porcelain").stdout == ""
    assert not (checkout / ".git" / "rebase-apply").exists()
    assert not (checkout / "temporary.txt").exists()
    assert run_git(checkout, "rev-parse", "HEAD").stdout.strip() == base_revision


def test_checkout_only_command_accepts_one_project() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/sync_upstreams.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--checkout-only" in result.stdout
