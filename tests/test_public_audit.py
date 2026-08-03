# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

AUDIT_SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_public_repo.py"


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def run_audit(repo: Path, mode: str = "--tracked") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), mode],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    run_git(tmp_path, "init", "-q")
    run_git(tmp_path, "config", "user.email", "test@example.invalid")
    run_git(tmp_path, "config", "user.name", "Test User")
    return tmp_path


def write_and_track(repo: Path, relative_path: str, content: str | bytes) -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    run_git(repo, "add", relative_path)


def test_safe_markdown_passes_tracked_audit(git_repo: Path) -> None:
    write_and_track(git_repo, "README.md", "# Safe public project\n")
    run_git(git_repo, "commit", "-qm", "safe fixture")

    result = run_audit(git_repo)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "content",
    [
        "docs/setup.md describes repo/relative/example.txt\n",
        "route = /api/v1/inference\n",
        "artifact = workspace/checkpoints/model.pt\n",
    ],
)
def test_repo_relative_text_passes_tracked_audit(
    git_repo: Path, content: str
) -> None:
    write_and_track(git_repo, "notes.txt", content)
    run_git(git_repo, "commit", "-qm", "safe relative-path fixture")

    result = run_audit(git_repo)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("relative_path", "content", "rule"),
    [
        (
            "config.txt",
            "AWS_SECRET_ACCESS" + "_KEY=not-a-real-secret",
            "secret pattern",
        ),
        (
            "private-key.txt",
            "-----BEGIN "
            + "PRIVATE KEY-----\nnot-a-real-key\n-----END PRIVATE KEY-----",
            "secret pattern",
        ),
        (
            "notes.txt",
            "/" + "home/alice/private/checkpoint.bin",
            "machine-specific absolute path",
        ),
        (
            "windows-path.txt",
            "D:" + r"\research\private\checkpoint.bin",
            "machine-specific absolute path",
        ),
        (
            "root-path.txt",
            "/" + "root/private/checkpoint.bin",
            "machine-specific absolute path",
        ),
        (
            "tmp-path.txt",
            "/" + "tmp/private/checkpoint.bin",
            "machine-specific absolute path",
        ),
        (
            "workspace-path.txt",
            "/" + "workspace/private/checkpoint.bin",
            "machine-specific absolute path",
        ),
        ("payload.dat", b"0" * (6 * 1024 * 1024), "file size exceeds 5 MiB"),
        ("extension.pt", b"synthetic weights", "forbidden extension"),
    ],
    ids=[
        "credential",
        "private-key",
        "absolute-home",
        "absolute-windows-drive",
        "absolute-root",
        "absolute-tmp",
        "absolute-workspace",
        "oversized",
        "extension",
    ],
)
def test_tracked_audit_rejects_unsafe_files_without_leaking_content(
    git_repo: Path,
    relative_path: str,
    content: str | bytes,
    rule: str,
) -> None:
    write_and_track(git_repo, relative_path, content)
    run_git(git_repo, "commit", "-qm", "unsafe fixture")

    result = run_audit(git_repo)
    output = result.stdout + result.stderr

    assert result.returncode == 1
    assert output.strip() == f"{relative_path}: {rule}"
    if isinstance(content, str):
        assert content not in output


def test_staged_audit_reads_index_blob_not_worktree(git_repo: Path) -> None:
    credential = "AWS_SECRET_ACCESS" + "_KEY=staged-only-secret"
    write_and_track(git_repo, "config.txt", credential)
    (git_repo / "config.txt").write_text("safe working tree contents\n", encoding="utf-8")

    result = run_audit(git_repo, "--staged")
    output = result.stdout + result.stderr

    assert result.returncode == 1
    assert output.strip() == "config.txt: secret pattern"
    assert credential not in output


def test_detect_secrets_scans_staged_blob_without_leaking_match(
    git_repo: Path,
) -> None:
    credential = "auth_token = " + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"
    write_and_track(git_repo, "detector-only.txt", credential)
    (git_repo / "detector-only.txt").write_text(
        "safe working tree contents\n", encoding="utf-8"
    )

    result = run_audit(git_repo, "--staged")
    output = result.stdout + result.stderr

    assert result.returncode == 1
    assert output.strip() == "detector-only.txt: secret pattern"
    assert credential not in output


@pytest.mark.parametrize(
    "relative_path",
    [
        "program.exe",
        "extension.pyd",
        "cache.pyc",
        "package.whl",
        "bundle.zip",
        "bundle.tar",
        "bundle.tar.gz",
        "bundle.tar.bz2",
        "bundle.tgz",
        "bundle.tar.zst",
        "bundle.7z",
    ],
)
def test_tracked_audit_rejects_build_products_and_archives(
    git_repo: Path, relative_path: str
) -> None:
    write_and_track(git_repo, relative_path, b"synthetic artifact")
    run_git(git_repo, "commit", "-qm", "artifact fixture")

    result = run_audit(git_repo)

    assert result.returncode == 1
    assert (result.stdout + result.stderr).strip() == (
        f"{relative_path}: forbidden extension"
    )
