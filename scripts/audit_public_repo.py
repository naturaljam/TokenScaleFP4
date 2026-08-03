# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

MAX_FILE_SIZE = 5 * 1024 * 1024
FORBIDDEN_EXTENSIONS = {
    ".a",
    ".bin",
    ".bz2",
    ".ckpt",
    ".cubin",
    ".dll",
    ".dylib",
    ".exe",
    ".gz",
    ".jar",
    ".lib",
    ".npy",
    ".npz",
    ".o",
    ".obj",
    ".onnx",
    ".pt",
    ".pth",
    ".pyc",
    ".pyd",
    ".rar",
    ".safetensors",
    ".so",
    ".tar",
    ".tgz",
    ".txz",
    ".whl",
    ".xz",
    ".zip",
    ".zst",
    ".7z",
}
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    re.compile(
        rb"(?i)(?:aws_secret_access_key|api[_-]?key|client[_-]?secret|password)"
        rb"\s*[:=]\s*[^\s]+"
    ),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
)
ABSOLUTE_PATH_PATTERNS = (
    re.compile(
        rb"(?<![A-Za-z0-9_])/(?:"
        rb"(?:home|Users)/[^/\s]+|"
        rb"(?:private/)?tmp|root|var/tmp|workspace"
        rb")(?:[/\s]|$)"
    ),
    re.compile(rb"(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/]"),
)


def run_git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
    ).stdout


def selected_paths(mode: str) -> list[str]:
    if mode == "--staged":
        output = run_git(
            "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"
        )
    else:
        output = run_git("ls-files", "-z")
    return [item.decode("utf-8") for item in output.split(b"\0") if item]


def read_blob(path: str, mode: str) -> bytes:
    if mode == "--staged":
        return run_git("show", f":{path}")
    return Path(path).read_bytes()


def detect_secret_paths(blobs: dict[str, bytes]) -> set[str]:
    with tempfile.TemporaryDirectory(prefix="public-repo-audit-") as directory:
        scan_root = Path(directory)
        for path, content in blobs.items():
            destination = scan_root.joinpath(*PurePosixPath(path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "detect_secrets",
                "-C",
                str(scan_root),
                "scan",
                "--all-files",
                "--slim",
                "--no-verify",
                ".",
            ],
            check=True,
            capture_output=True,
        )
        baseline: dict[str, Any] = json.loads(result.stdout)

    results: dict[str, list[object]] = baseline.get("results", {})
    return {PurePosixPath(path).as_posix() for path, matches in results.items() if matches}


def audit_blob(path: str, content: bytes) -> list[str]:
    violations: list[str] = []
    suffix = PurePosixPath(path).suffix.lower()
    if len(content) > MAX_FILE_SIZE:
        violations.append("file size exceeds 5 MiB")
    if suffix in FORBIDDEN_EXTENSIONS:
        violations.append("forbidden extension")
    if any(pattern.search(content) for pattern in SECRET_PATTERNS):
        violations.append("secret pattern")
    if any(pattern.search(content) for pattern in ABSOLUTE_PATH_PATTERNS):
        violations.append("machine-specific absolute path")
    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit files before publication")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--staged", action="store_true")
    selection.add_argument("--tracked", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = "--staged" if args.staged else "--tracked"
    blobs = {path: read_blob(path, mode) for path in selected_paths(mode)}
    detected_secret_paths = detect_secret_paths(blobs)
    violations: list[tuple[str, str]] = []
    for path, content in blobs.items():
        rules = audit_blob(path, content)
        violations.extend((path, rule) for rule in rules)
        if path in detected_secret_paths and "secret pattern" not in rules:
            violations.append((path, "secret pattern"))
    for path, rule in violations:
        print(f"{path}: {rule}", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
