# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEVANT_PACKAGES = (
    "cuda-bindings",
    "cuda-python",
    "cuda-toolkit",
    "flashinfer-python",
    "ninja",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-cu13",
    "nvidia-nccl-cu13",
)
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class EnvironmentFacts:
    os_distribution: str = "unknown"
    kernel: str = "unknown"
    python: str = "unknown"
    pytorch: str = "unknown"
    torch_cuda: str | None = None
    driver: str = "unknown"
    gpu_product_name: str = "unknown"
    compute_capability: tuple[int, int] = (0, 0)
    vram_total_mib: int = 0
    flashinfer_sha: str = "unknown"
    vllm_sha: str = "unknown"
    packages: dict[str, str] = field(default_factory=dict)
    cute_dsl_available: bool = True

    def require_sm120_b12x(self) -> None:
        if self.compute_capability != (12, 0):
            detected = ".".join(str(part) for part in self.compute_capability)
            raise SystemExit(
                f"SM120 is required for the b12x smoke gate; detected {detected}"
            )
        if self.torch_cuda is None:
            raise SystemExit(
                "CUDA 13 or later is required; PyTorch reports no CUDA runtime"
            )
        try:
            cuda_major = int(self.torch_cuda.split(".", maxsplit=1)[0])
        except ValueError as error:
            raise SystemExit(
                f"CUDA 13 or later is required; PyTorch reports {self.torch_cuda!r}"
            ) from error
        if cuda_major < 13:
            raise SystemExit(
                f"CUDA 13 or later is required; PyTorch reports {self.torch_cuda}"
            )
        if not self.cute_dsl_available:
            raise SystemExit("CuTe DSL is required for the SM120 b12x backend")

    def as_manifest(self) -> dict[str, Any]:
        manifest = asdict(self)
        manifest.pop("cute_dsl_available")
        manifest["compute_capability"] = ".".join(
            str(part) for part in self.compute_capability
        )
        return manifest


def parse_nvidia_smi_csv(output: str) -> dict[str, object]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("expected nvidia-smi output for exactly one GPU")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 4:
        raise ValueError("unexpected nvidia-smi CSV field count")
    product_name, driver, memory_text, capability_text = fields
    memory_match = re.fullmatch(r"([0-9]+)\s+MiB", memory_text)
    capability_match = re.fullmatch(r"([0-9]+)\.([0-9]+)", capability_text)
    if memory_match is None or capability_match is None:
        raise ValueError("unexpected nvidia-smi memory or compute capability value")
    return {
        "gpu_product_name": product_name,
        "driver": driver,
        "vram_total_mib": int(memory_match.group(1)),
        "compute_capability": (
            int(capability_match.group(1)),
            int(capability_match.group(2)),
        ),
    }


def _run(*command: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RuntimeError(f"{command[0]} failed: {detail}")
    return result.stdout.strip()


def _run_git(checkout: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )


def git_checkout_is_clean(
    checkout: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = _run_git,
) -> bool:
    staged = runner(checkout, "diff", "--cached", "--quiet")
    working = runner(
        checkout,
        "-c",
        "core.filemode=false",
        "diff",
        "--quiet",
        "--ignore-space-at-eol",
    )
    untracked = runner(checkout, "ls-files", "--others", "--exclude-standard")
    for result in (staged, working):
        if result.returncode not in (0, 1):
            detail = result.stderr.strip() or "no diagnostic"
            raise RuntimeError(f"git checkout inspection failed: {detail}")
    if untracked.returncode != 0:
        detail = untracked.stderr.strip() or "no diagnostic"
        raise RuntimeError(f"git checkout inspection failed: {detail}")
    return (
        staged.returncode == 0
        and working.returncode == 0
        and not untracked.stdout.strip()
    )


def _checkout_revision(project: str) -> str:
    checkout = PROJECT_ROOT / ".third_party" / project
    revision_path = PROJECT_ROOT / "upstream" / project / "base-revision.txt"
    expected = revision_path.read_text(encoding="utf-8").strip()
    if FULL_SHA.fullmatch(expected) is None:
        raise RuntimeError(f"{project} base revision is not a full lowercase SHA")
    actual = _run("git", "rev-parse", "HEAD", cwd=checkout)
    if actual != expected:
        raise RuntimeError(f"{project} checkout does not match its pinned revision")
    if not git_checkout_is_clean(checkout):
        raise RuntimeError(f"{project} checkout must be clean with no patch applied")
    return actual


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in RELEVANT_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def collect_environment() -> EnvironmentFacts:
    import torch

    gpu = parse_nvidia_smi_csv(
        _run(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        )
    )
    os_release = platform.freedesktop_os_release()
    return EnvironmentFacts(
        os_distribution=os_release.get("PRETTY_NAME", "unknown"),
        kernel=platform.release(),
        python=platform.python_version(),
        pytorch=torch.__version__,
        torch_cuda=torch.version.cuda,
        driver=str(gpu["driver"]),
        gpu_product_name=str(gpu["gpu_product_name"]),
        compute_capability=tuple(gpu["compute_capability"]),
        vram_total_mib=int(gpu["vram_total_mib"]),
        flashinfer_sha=_checkout_revision("flashinfer"),
        vllm_sha=_checkout_revision("vllm"),
        packages=_package_versions(),
        cute_dsl_available=(
            importlib.util.find_spec("cutlass") is not None
            and importlib.util.find_spec("cutlass.cute") is not None
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and record the public-safe SM120 environment"
    )
    parser.add_argument("--write", type=Path, help="write the sanitized JSON manifest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    facts = collect_environment()
    facts.require_sm120_b12x()
    manifest = facts.as_manifest()
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
