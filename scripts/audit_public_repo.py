# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

MAX_FILE_SIZE = 5 * 1024 * 1024
FORBIDDEN_EXTENSIONS = {
    ".a",
    ".bin",
    ".ckpt",
    ".cubin",
    ".dll",
    ".dylib",
    ".lib",
    ".npy",
    ".npz",
    ".o",
    ".obj",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".so",
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
    re.compile(rb"(?<![A-Za-z0-9_])/(?:home|Users)/[^/\s]+/"),
    re.compile(rb"(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/](?:Users|Documents and Settings)[\\/]"),
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
    violations = [
        (path, rule)
        for path in selected_paths(mode)
        for rule in audit_blob(path, read_blob(path, mode))
    ]
    for path, rule in violations:
        print(f"{path}: {rule}", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
