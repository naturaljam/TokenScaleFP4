# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parents[2]


def load_script(name: str) -> dict[str, Any]:
    return runpy.run_path(str(REPO_ROOT / "scripts" / name))


def test_environment_rejects_cuda_12_8() -> None:
    environment = load_script("check_environment.py")
    facts = environment["EnvironmentFacts"](
        compute_capability=(12, 0),
        torch_cuda="12.8",
    )

    with pytest.raises(SystemExit, match="CUDA 13"):
        facts.require_sm120_b12x()


def test_environment_rejects_non_sm120() -> None:
    environment = load_script("check_environment.py")
    facts = environment["EnvironmentFacts"](
        compute_capability=(12, 1),
        torch_cuda="13.0",
    )

    with pytest.raises(SystemExit, match="SM120"):
        facts.require_sm120_b12x()


def test_environment_rejects_missing_cute_dsl() -> None:
    environment = load_script("check_environment.py")
    facts = environment["EnvironmentFacts"](
        compute_capability=(12, 0),
        torch_cuda="13.0",
        cute_dsl_available=False,
    )

    with pytest.raises(SystemExit, match="CuTe DSL"):
        facts.require_sm120_b12x()


def test_nvidia_smi_parser_accepts_injected_csv_output() -> None:
    environment = load_script("check_environment.py")

    parsed = environment["parse_nvidia_smi_csv"](
        "NVIDIA GeForce RTX 5070 Laptop GPU, 581.42, 8151 MiB, 12.0\n"
    )

    assert parsed == {
        "gpu_product_name": "NVIDIA GeForce RTX 5070 Laptop GPU",
        "driver": "581.42",
        "vram_total_mib": 8151,
        "compute_capability": (12, 0),
    }


def test_environment_records_only_relevant_package_versions() -> None:
    environment = load_script("check_environment.py")

    assert environment["RELEVANT_PACKAGES"] == (
        "cuda-bindings",
        "cuda-python",
        "cuda-toolkit",
        "flashinfer-python",
        "ninja",
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-cu13",
        "nvidia-nccl-cu13",
    )


@pytest.mark.parametrize(
    ("return_codes", "untracked", "expected"),
    [
        ((0, 0), "", True),
        ((1, 0), "", False),
        ((0, 1), "", False),
        ((0, 0), "generated.txt\n", False),
    ],
)
def test_checkout_clean_gate_ignores_only_eol_representation_changes(
    return_codes: tuple[int, int],
    untracked: str,
    expected: bool,
    tmp_path: Path,
) -> None:
    environment = load_script("check_environment.py")
    results = iter(
        [
            subprocess.CompletedProcess([], return_codes[0], "", ""),
            subprocess.CompletedProcess([], return_codes[1], "", ""),
            subprocess.CompletedProcess([], 0, untracked, ""),
        ]
    )

    def injected_runner(
        checkout: Path,
        *args: str,
    ) -> subprocess.CompletedProcess[str]:
        assert checkout == tmp_path
        return next(results)

    assert (
        environment["git_checkout_is_clean"](tmp_path, runner=injected_runner)
        is expected
    )


def test_checkout_clean_gate_rejects_ls_files_command_failure(tmp_path: Path) -> None:
    environment = load_script("check_environment.py")
    results = iter(
        [
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 1, "", "ls-files failed"),
        ]
    )

    def injected_runner(
        checkout: Path,
        *args: str,
    ) -> subprocess.CompletedProcess[str]:
        assert checkout == tmp_path
        return next(results)

    with pytest.raises(RuntimeError, match="ls-files failed"):
        environment["git_checkout_is_clean"](tmp_path, runner=injected_runner)


def test_manifest_uses_only_public_safe_allowlisted_fields() -> None:
    environment = load_script("check_environment.py")
    facts = environment["EnvironmentFacts"](
        os_distribution="Ubuntu 24.04.3 LTS",
        kernel="6.6.87.2-microsoft-standard-WSL2",
        python="3.12.11",
        pytorch="2.11.0+cu130",
        torch_cuda="13.0",
        driver="581.42",
        gpu_product_name="NVIDIA GeForce RTX 5070 Laptop GPU",
        compute_capability=(12, 0),
        vram_total_mib=8151,
        flashinfer_sha="08ddfbcd2e89b2f4b68391825817909e30d445e2",
        vllm_sha="0a6446005d51c9e6bfa09352f7f288ddeff17c77",
        packages={"flashinfer-python": "0.6.4"},
    )

    manifest = facts.as_manifest()

    assert set(manifest) == {
        "os_distribution",
        "kernel",
        "python",
        "pytorch",
        "torch_cuda",
        "driver",
        "gpu_product_name",
        "compute_capability",
        "vram_total_mib",
        "flashinfer_sha",
        "vllm_sha",
        "packages",
    }
    assert manifest["compute_capability"] == "12.0"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"output_dtype": "float16"}, "BF16"),
        ({"all_finite": False}, "finite"),
        ({"cosine_similarity": 0.998}, "0.999"),
    ],
)
def test_scalar_smoke_result_rejects_failed_gate(
    overrides: dict[str, object],
    message: str,
) -> None:
    smoke = load_script("run_scalar_smoke.py")
    values: dict[str, object] = {
        "output_dtype": "bfloat16",
        "all_finite": True,
        "cosine_similarity": 0.999,
    }
    values.update(overrides)
    result = smoke["SmokeResult"](**values)

    with pytest.raises(SystemExit, match=message):
        result.require()


def test_scalar_smoke_result_accepts_threshold() -> None:
    smoke = load_script("run_scalar_smoke.py")
    result = smoke["SmokeResult"](
        output_dtype="bfloat16",
        all_finite=True,
        cosine_similarity=0.999,
    )

    result.require()


def test_smoke_prepends_interpreter_bin_to_subprocess_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke = load_script("run_scalar_smoke.py")
    interpreter = tmp_path / "venv" / "bin" / "python"
    monkeypatch.setenv("PATH", os.pathsep.join(("system-bin", "other-bin")))

    smoke["prepare_runtime_environment"](executable=interpreter)
    smoke["prepare_runtime_environment"](executable=interpreter)

    path_entries = os.environ["PATH"].split(os.pathsep)
    assert path_entries[0] == str(interpreter.parent)
    assert path_entries.count(str(interpreter.parent)) == 1


def test_smoke_prefers_space_free_flashinfer_source_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke = load_script("run_scalar_smoke.py")
    alias = tmp_path / "flashinfer"
    alias.mkdir()
    workspace_base = tmp_path / "runtime"
    monkeypatch.setattr(sys, "path", ["existing-import-path"])
    monkeypatch.delenv("FLASHINFER_WORKSPACE_BASE", raising=False)

    smoke["prepare_runtime_environment"](
        executable=tmp_path / "venv" / "bin" / "python",
        flashinfer_source=alias,
        workspace_base=workspace_base,
    )

    assert sys.path[0] == str(alias)
    assert os.environ["FLASHINFER_WORKSPACE_BASE"] == str(workspace_base)


@pytest.mark.sm120
def test_scalar_b12x_smoke() -> None:
    environment = load_script("check_environment.py")
    facts = environment["collect_environment"]()
    try:
        facts.require_sm120_b12x()
    except SystemExit as error:
        pytest.skip(str(error))

    smoke = load_script("run_scalar_smoke.py")
    result = smoke["run_scalar_smoke"](m=16, n=1536, k=1536, seed=20260803)

    result.require()
