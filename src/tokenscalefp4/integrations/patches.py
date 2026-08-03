# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Literal

Project = Literal["flashinfer", "vllm"]
PROJECTS: tuple[Project, ...] = ("flashinfer", "vllm")
REPO_ROOT = Path(__file__).resolve().parents[3]
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def _validate_project(project: str) -> Project:
    if project not in PROJECTS:
        choices = ", ".join(PROJECTS)
        raise ValueError(f"Unknown upstream project {project!r}; expected one of {choices}")
    return project


def _run_git(checkout: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )


def read_base_revision(project: Project) -> str:
    """Return the validated immutable revision recorded for an upstream project."""
    validated_project = _validate_project(project)
    revision_path = REPO_ROOT / "upstream" / validated_project / "base-revision.txt"
    revision = revision_path.read_text(encoding="utf-8").strip()
    if not FULL_SHA.fullmatch(revision):
        raise ValueError(f"{revision_path} must contain one full lowercase Git SHA")
    return revision


def discover_patches(patch_dir: Path) -> list[Path]:
    """Return regular patch files in deterministic lexical filename order."""
    if not patch_dir.exists():
        return []
    return sorted(
        (path for path in patch_dir.glob("*.patch") if path.is_file()),
        key=lambda path: path.name,
    )


def _reject_dirty_checkout(project: str, checkout: Path) -> None:
    status = _run_git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode != 0:
        raise RuntimeError(f"{project} checkout is not a Git working tree: {checkout}")
    if status.stdout:
        modified_paths = [line[3:] for line in status.stdout.splitlines()]
        raise RuntimeError(
            f"Refusing dirty {project} checkout; modified paths: "
            f"{', '.join(modified_paths)}"
        )


def apply_patch_stack(project: str, checkout: Path) -> list[Path]:
    """Apply the project's ordered patch stack to a clean Git checkout."""
    validated_project = _validate_project(project)
    _reject_dirty_checkout(validated_project, checkout)
    patch_dir = REPO_ROOT / "upstream" / validated_project / "patches"
    patch_stack = discover_patches(patch_dir)
    if not patch_stack:
        return []

    result = _run_git(checkout, "am", "--3way", *(str(path) for path in patch_stack))
    if result.returncode == 0:
        return patch_stack

    _run_git(checkout, "am", "--abort")
    failed_patch = patch_stack[0].name
    raise RuntimeError(
        f"Failed to apply patch stack for {validated_project} at {failed_patch}: "
        f"{result.stderr.strip()}"
    )
