# SPDX-License-Identifier: Apache-2.0

import tomllib
from pathlib import Path

import tokenscalefp4

REPO_ROOT = Path(__file__).parents[1]


def test_package_exports_development_version() -> None:
    assert tokenscalefp4.__version__ == "0.1.0.dev0"


def test_pytest_enforces_diagnostic_and_marker_options() -> None:
    configuration = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert configuration["tool"]["pytest"]["ini_options"]["addopts"] == (
        "-ra --strict-markers"
    )


def test_notice_names_upstream_projects() -> None:
    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")

    assert "FlashInfer" in notice
    assert "vLLM" in notice
