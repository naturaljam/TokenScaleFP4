# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import runpy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple, cast


class QualityGateResult(NamedTuple):
    perplexity_relative_increase: float
    gsm8k_absolute_drop: float
    perplexity_passed: bool
    gsm8k_passed: bool
    finite_passed: bool

    @property
    def passed(self) -> bool:
        return self.perplexity_passed and self.gsm8k_passed and self.finite_passed


class EvaluationSettings(NamedTuple):
    dataset: str
    subset: str
    split: str
    revision: str
    seed: int


class QualitySettings(NamedTuple):
    model: str
    model_revision: str
    seed: int
    perplexity: EvaluationSettings
    gsm8k: EvaluationSettings
    max_relative_perplexity_increase: float
    max_absolute_gsm8k_drop: float
    finite_required: bool


FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
GSM8K_ANSWER = re.compile(r"####\s*(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "evals" / "quality.json"
QUALITY_WINDOW_TOKENS = 2048
GSM8K_FEWSHOT_COUNT = 5
GSM8K_MAX_NEW_TOKENS = 256
ELIGIBLE_LINEAR_SUFFIXES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")  # noqa: TRY004
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{path} must use string keys")
    return cast(Mapping[str, Any], mapping)


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")  # noqa: TRY004
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")  # noqa: TRY004
    return value


def _revision(value: object, path: str) -> str:
    revision = _string(value, path)
    if FULL_SHA.fullmatch(revision) is None:
        raise ValueError(f"{path} must be an immutable 40-character SHA")
    return revision


def is_eligible_linear(name: str) -> bool:
    return name.rsplit(".", maxsplit=1)[-1] in ELIGIBLE_LINEAR_SUFFIXES


def replace_eligible_linears(
    model: Any,
    *,
    factory: Callable[[Any], Any],
) -> tuple[str, ...]:
    import torch

    names = [
        name
        for name, module in model.named_modules()
        if name and isinstance(module, torch.nn.Linear) and is_eligible_linear(name)
    ]
    for name in names:
        parent_name, child_name = name.rsplit(".", maxsplit=1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        parent.add_module(child_name, factory(getattr(parent, child_name)))
    return tuple(names)


def _evaluation_settings(value: object, path: str) -> EvaluationSettings:
    section = _mapping(value, path)
    return EvaluationSettings(
        dataset=_string(section.get("dataset"), f"{path}.dataset"),
        subset=_string(section.get("subset"), f"{path}.subset"),
        split=_string(section.get("split"), f"{path}.split"),
        revision=_revision(section.get("revision"), f"{path}.revision"),
        seed=_integer(section.get("seed"), f"{path}.seed"),
    )


def load_quality_settings(path: Path, model: str) -> QualitySettings:
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "quality")
    if payload.get("schema_version") != 1:
        raise ValueError("quality.schema_version must equal 1")
    local_model = _string(payload.get("local_model"), "quality.local_model")
    final_model = _string(payload.get("final_model"), "quality.final_model")
    if model == local_model:
        revision_key = "local_model_revision"
    elif model == final_model:
        revision_key = "final_model_revision"
    else:
        raise ValueError(f"model {model!r} is not a configured quality target")

    perplexity = _evaluation_settings(payload.get("perplexity"), "quality.perplexity")
    gsm8k = _evaluation_settings(payload.get("gsm8k"), "quality.gsm8k")
    if perplexity.seed != gsm8k.seed:
        raise ValueError("quality evaluation seeds must match")
    finite_required = payload.get("finite_required")
    if not isinstance(finite_required, bool):
        raise ValueError("quality.finite_required must be a boolean")  # noqa: TRY004
    return QualitySettings(
        model=model,
        model_revision=_revision(payload.get(revision_key), f"quality.{revision_key}"),
        seed=perplexity.seed,
        perplexity=perplexity,
        gsm8k=gsm8k,
        max_relative_perplexity_increase=_number(
            _mapping(payload.get("perplexity"), "quality.perplexity").get(
                "max_relative_increase"
            ),
            "quality.perplexity.max_relative_increase",
        ),
        max_absolute_gsm8k_drop=_number(
            _mapping(payload.get("gsm8k"), "quality.gsm8k").get(
                "max_absolute_drop"
            ),
            "quality.gsm8k.max_absolute_drop",
        ),
        finite_required=finite_required,
    )


def build_quality_payload(
    settings: QualitySettings,
    *,
    mode: str,
    perplexity: float,
    perplexity_sample_count: int,
    gsm8k_accuracy: float,
    gsm8k_sample_count: int,
    all_finite: bool,
    greedy_token_ids: list[int] | None = None,
    target_logprobs: list[float] | None = None,
) -> dict[str, Any]:
    if mode not in {"bf16", "nvfp4-unfused"}:
        raise ValueError("mode must be 'bf16' or 'nvfp4-unfused'")
    payload = {
        "schema_version": 1,
        "model": settings.model,
        "model_revision": settings.model_revision,
        "mode": mode,
        "seed": settings.seed,
        "perplexity": {
            "dataset": settings.perplexity.dataset,
            "revision": settings.perplexity.revision,
            "split": settings.perplexity.split,
            "sample_count": perplexity_sample_count,
            "value": perplexity,
        },
        "gsm8k": {
            "dataset": settings.gsm8k.dataset,
            "revision": settings.gsm8k.revision,
            "split": settings.gsm8k.split,
            "sample_count": gsm8k_sample_count,
            "accuracy": gsm8k_accuracy,
        },
        "all_finite": all_finite,
    }
    if greedy_token_ids is not None or target_logprobs is not None:
        if greedy_token_ids is None or target_logprobs is None:
            raise ValueError("both token trace fields must be provided")
        if len(greedy_token_ids) != len(target_logprobs):
            raise ValueError("token trace fields must have equal lengths")
        payload["token_trace"] = {
            "greedy_token_ids": greedy_token_ids,
            "target_logprobs": target_logprobs,
        }
    return payload


def perplexity_from_totals(*, total_nll: float, token_count: int) -> float:
    if not math.isfinite(total_nll) or token_count <= 0:
        raise ValueError("perplexity totals must be finite with positive token count")
    return math.exp(total_nll / token_count)


def extract_gsm8k_answer(generated: str) -> str | None:
    match = GSM8K_ANSWER.search(generated)
    if match is None:
        return None
    return match.group(1).replace("$", "").replace(",", "")


def build_gsm8k_prompt(
    fewshot_examples: list[Mapping[str, object]],
    sample: Mapping[str, object],
) -> str:
    parts: list[str] = []
    for example in fewshot_examples:
        parts.append(
            "Question: "
            + _string(example.get("question"), "gsm8k.question")
            + "\nAnswer: "
            + _string(example.get("answer"), "gsm8k.answer")
        )
    parts.append(
        "Question: "
        + _string(sample.get("question"), "gsm8k.question")
        + "\nAnswer:"
    )
    return "\n\n".join(parts)


def prepare_runtime_environment() -> None:
    smoke = runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_scalar_smoke.py"))
    smoke["prepare_runtime_environment"]()


def _load_model_and_tokenizer(
    settings: QualitySettings,
    *,
    mode: str,
    device: str,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        settings.model,
        revision=settings.model_revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.to(device)
    if mode == "nvfp4-unfused":
        from tokenscalefp4.kernel_lab.flashinfer_ops import Nvfp4Linear

        replaced = replace_eligible_linears(model, factory=Nvfp4Linear.from_linear)
        if not replaced:
            raise RuntimeError("no eligible dense projection layers were quantized")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model,
        revision=settings.model_revision,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device


def load_dataset_split(
    settings: EvaluationSettings,
    *,
    split: str,
    loader: Callable[..., Any] | None = None,
    snapshot_resolver: Callable[..., str] | None = None,
) -> Any:
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        return loader(
            settings.dataset,
            name=settings.subset,
            split=split,
            revision=settings.revision,
        )

    if snapshot_resolver is None:
        from huggingface_hub import snapshot_download

        snapshot_resolver = snapshot_download
    snapshot = Path(
        snapshot_resolver(
            repo_id=settings.dataset,
            repo_type="dataset",
            revision=settings.revision,
            local_files_only=True,
        )
    )
    parquet_files = sorted(
        str(path)
        for path in (snapshot / settings.subset).glob(f"{split}-*.parquet")
        if path.is_file()
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"offline snapshot has no {settings.subset}/{split} parquet files"
        )
    return loader("parquet", data_files={split: parquet_files}, split=split)


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    settings: QualitySettings,
) -> tuple[float, int, bool, list[int], list[float]]:
    import torch
    from torch.nn import functional

    dataset = load_dataset_split(
        settings.perplexity,
        split=settings.perplexity.split,
    )
    text = "\n\n".join(
        _string(row.get("text"), "wikitext.text")
        for row in dataset
        if isinstance(row, Mapping) and row.get("text")
    )
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    tokens = encoded["input_ids"][0]
    device = _model_device(model)
    total_nll = 0.0
    token_count = 0
    all_finite = True
    greedy_token_ids: list[int] = []
    target_logprobs: list[float] = []
    with torch.inference_mode():
        for start in range(0, max(tokens.numel() - 1, 0), QUALITY_WINDOW_TOKENS):
            window = tokens[start : start + QUALITY_WINDOW_TOKENS]
            if window.numel() < 2:
                continue
            input_ids = window.unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits[:, :-1].float()
            labels = input_ids[:, 1:]
            all_finite = all_finite and bool(torch.isfinite(logits).all().item())
            logprobs = functional.log_softmax(logits, dim=-1)
            selected_logprobs = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            all_finite = all_finite and bool(
                torch.isfinite(selected_logprobs).all().item()
            )
            greedy_token_ids.extend(logits.argmax(dim=-1).reshape(-1).cpu().tolist())
            target_logprobs.extend(selected_logprobs.reshape(-1).cpu().tolist())
            nll = -selected_logprobs.sum()
            all_finite = all_finite and bool(torch.isfinite(nll).item())
            total_nll += float(nll.item())
            token_count += labels.numel()
    return (
        perplexity_from_totals(total_nll=total_nll, token_count=token_count),
        token_count,
        all_finite,
        greedy_token_ids,
        target_logprobs,
    )


def evaluate_gsm8k(
    model: Any,
    tokenizer: Any,
    settings: QualitySettings,
) -> tuple[float, int, bool]:
    import torch

    train = load_dataset_split(
        settings.gsm8k,
        split="train",
    )
    test = load_dataset_split(
        settings.gsm8k,
        split=settings.gsm8k.split,
    )
    rng = random.Random(settings.seed)
    fewshot_indices = rng.sample(range(len(train)), GSM8K_FEWSHOT_COUNT)
    fewshot = [cast(Mapping[str, object], train[index]) for index in fewshot_indices]
    device = _model_device(model)
    correct = 0
    all_finite = True
    for row in test:
        sample = cast(Mapping[str, object], row)
        prompt = build_gsm8k_prompt(fewshot, sample)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=GSM8K_MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        scores = generated.scores or ()
        all_finite = all_finite and all(
            bool(torch.isfinite(score).all().item()) for score in scores
        )
        continuation = generated.sequences[0, input_ids.shape[1] :]
        decoded = tokenizer.decode(continuation, skip_special_tokens=True)
        predicted = extract_gsm8k_answer(decoded)
        target = extract_gsm8k_answer(
            _string(sample.get("answer"), "gsm8k.answer")
        )
        if predicted is not None and predicted == target:
            correct += 1
    count = len(test)
    return correct / count, count, all_finite


def run_quality_evaluation(
    settings: QualitySettings,
    *,
    mode: str,
    device: str = "cuda",
) -> dict[str, Any]:
    if mode not in {"bf16", "nvfp4-unfused"}:
        raise ValueError("mode must be 'bf16' or 'nvfp4-unfused'")
    if mode == "nvfp4-unfused":
        prepare_runtime_environment()
    model, tokenizer = _load_model_and_tokenizer(
        settings,
        mode=mode,
        device=device,
    )
    (
        perplexity,
        perplexity_count,
        ppl_finite,
        greedy_token_ids,
        target_logprobs,
    ) = evaluate_perplexity(model, tokenizer, settings)
    gsm8k_accuracy, gsm8k_count, gsm_finite = evaluate_gsm8k(
        model, tokenizer, settings
    )
    return build_quality_payload(
        settings,
        mode=mode,
        perplexity=perplexity,
        perplexity_sample_count=perplexity_count,
        gsm8k_accuracy=gsm8k_accuracy,
        gsm8k_sample_count=gsm8k_count,
        all_finite=ppl_finite and gsm_finite,
        greedy_token_ids=greedy_token_ids,
        target_logprobs=target_logprobs,
    )


def write_quality_result(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _trace_values(
    payload: Mapping[str, object],
    *,
    label: str,
) -> tuple[list[int], list[float]]:
    trace = _mapping(payload.get("token_trace"), f"{label}.token_trace")
    greedy_value = trace.get("greedy_token_ids")
    logprob_value = trace.get("target_logprobs")
    if not isinstance(greedy_value, list) or not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in greedy_value
    ):
        raise ValueError(f"{label}.token_trace.greedy_token_ids must be integers")
    if not isinstance(logprob_value, list):
        raise TypeError(f"{label}.token_trace.target_logprobs must be a list")
    logprobs = [
        _number(value, f"{label}.token_trace.target_logprobs")
        for value in logprob_value
    ]
    if not greedy_value or len(greedy_value) != len(logprobs):
        raise ValueError(f"{label} token trace must be non-empty with equal lengths")
    return greedy_value, logprobs


def compare_token_traces(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, int | float]:
    baseline_greedy, baseline_logprobs = _trace_values(baseline, label="baseline")
    candidate_greedy, candidate_logprobs = _trace_values(
        candidate, label="candidate"
    )
    if len(baseline_greedy) != len(candidate_greedy):
        raise ValueError("quality token traces use different sample counts")
    sample_count = len(baseline_greedy)
    agreements = sum(
        baseline_id == candidate_id
        for baseline_id, candidate_id in zip(
            baseline_greedy, candidate_greedy, strict=True
        )
    )
    absolute_drift = sum(
        abs(baseline_value - candidate_value)
        for baseline_value, candidate_value in zip(
            baseline_logprobs, candidate_logprobs, strict=True
        )
    )
    return {
        "sample_count": sample_count,
        "greedy_token_agreement": agreements / sample_count,
        "mean_absolute_target_logprob_drift": absolute_drift / sample_count,
    }


def _at_most(observed: float, threshold: float) -> bool:
    return observed <= threshold or math.isclose(
        observed,
        threshold,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def evaluate_quality_gates(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    max_relative_perplexity_increase: float,
    max_absolute_gsm8k_drop: float,
    finite_required: bool,
) -> QualityGateResult:
    if baseline.get("mode") != "bf16":
        raise ValueError("quality baseline mode must be 'bf16'")
    if candidate.get("mode") != "nvfp4-unfused":
        raise ValueError("quality candidate mode must be 'nvfp4-unfused'")
    for field in ("model", "model_revision", "seed"):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"quality results use different {field}")

    baseline_perplexity = _mapping(baseline.get("perplexity"), "baseline.perplexity")
    candidate_perplexity = _mapping(
        candidate.get("perplexity"), "candidate.perplexity"
    )
    baseline_gsm8k = _mapping(baseline.get("gsm8k"), "baseline.gsm8k")
    candidate_gsm8k = _mapping(candidate.get("gsm8k"), "candidate.gsm8k")
    for field in ("dataset", "revision", "split", "sample_count"):
        if baseline_perplexity.get(field) != candidate_perplexity.get(field):
            raise ValueError(f"perplexity results use different {field}")
        if baseline_gsm8k.get(field) != candidate_gsm8k.get(field):
            raise ValueError(f"gsm8k results use different {field}")

    baseline_ppl = _number(baseline_perplexity.get("value"), "baseline perplexity")
    candidate_ppl = _number(
        candidate_perplexity.get("value"), "candidate perplexity"
    )
    if baseline_ppl <= 0:
        raise ValueError("baseline perplexity must be positive")
    baseline_accuracy = _number(
        baseline_gsm8k.get("accuracy"), "baseline gsm8k accuracy"
    )
    candidate_accuracy = _number(
        candidate_gsm8k.get("accuracy"), "candidate gsm8k accuracy"
    )

    ppl_increase = (candidate_ppl - baseline_ppl) / baseline_ppl
    accuracy_drop = baseline_accuracy - candidate_accuracy
    baseline_finite = baseline.get("all_finite") is True
    candidate_finite = candidate.get("all_finite") is True
    finite_passed = not finite_required or (baseline_finite and candidate_finite)
    return QualityGateResult(
        perplexity_relative_increase=ppl_increase,
        gsm8k_absolute_drop=accuracy_drop,
        perplexity_passed=_at_most(
            ppl_increase, max_relative_perplexity_increase
        ),
        gsm8k_passed=_at_most(accuracy_drop, max_absolute_gsm8k_drop),
        finite_passed=finite_passed,
    )


def build_quality_comparison(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    settings: QualitySettings,
) -> tuple[dict[str, object], QualityGateResult]:
    gate = evaluate_quality_gates(
        baseline,
        candidate,
        max_relative_perplexity_increase=(
            settings.max_relative_perplexity_increase
        ),
        max_absolute_gsm8k_drop=settings.max_absolute_gsm8k_drop,
        finite_required=settings.finite_required,
    )
    comparison: dict[str, object] = {
        "perplexity_relative_increase": gate.perplexity_relative_increase,
        "gsm8k_absolute_drop": gate.gsm8k_absolute_drop,
        **compare_token_traces(baseline, candidate),
        "gates": {
            "perplexity_passed": gate.perplexity_passed,
            "gsm8k_passed": gate.gsm8k_passed,
            "finite_passed": gate.finite_passed,
            "passed": gate.passed,
        },
    }
    return comparison, gate


def _collect_evidence_environment() -> Mapping[str, object]:
    environment_script = runpy.run_path(
        str(PROJECT_ROOT / "scripts" / "check_environment.py")
    )
    facts = environment_script["collect_environment"]()
    facts.require_sm120_b12x()
    return cast(Mapping[str, object], facts.as_manifest())


def _environment_package(
    environment: Mapping[str, object],
    package: str,
    *,
    default: str,
) -> str:
    packages = _mapping(environment.get("packages"), "environment.packages")
    value = packages.get(package, default)
    return _string(value, f"environment.packages.{package}")


def write_quality_evidence(
    path: Path,
    *,
    raw_path: Path,
    settings: QualitySettings,
    command: str,
    gate: QualityGateResult | None,
    environment: Mapping[str, object] | None = None,
) -> None:
    manifest = environment or _collect_evidence_environment()
    compute_capability = manifest.get("compute_capability")
    if isinstance(compute_capability, (tuple, list)):
        compute_capability = ".".join(str(part) for part in compute_capability)
    if gate is None:
        raw_payload = _mapping(
            json.loads(raw_path.read_text(encoding="utf-8")), "quality"
        )
        finite_passed = raw_payload.get("all_finite") is True
        gate_payload: dict[str, object] = {
            "name": "finite_values",
            "threshold": "all finite",
            "observed": "all finite" if finite_passed else "non-finite detected",
            "passed": finite_passed,
        }
    else:
        gate_payload = {
            "name": "unfused_quality_feasibility",
            "threshold": (
                f"ppl<={settings.max_relative_perplexity_increase},"
                f"gsm8k_drop<={settings.max_absolute_gsm8k_drop},finite"
            ),
            "observed": (
                f"ppl={gate.perplexity_relative_increase},"
                f"gsm8k_drop={gate.gsm8k_absolute_drop},"
                f"finite={gate.finite_passed}"
            ),
            "passed": gate.passed,
        }
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = "not-installed"
    payload = {
        "schema_version": 1,
        "upstream": {
            "flashinfer_sha": _string(
                manifest.get("flashinfer_sha"), "environment.flashinfer_sha"
            ),
            "vllm_sha": _string(
                manifest.get("vllm_sha"), "environment.vllm_sha"
            ),
        },
        "environment": {
            "gpu_name": _string(
                manifest.get("gpu_product_name"), "environment.gpu_product_name"
            ),
            "compute_capability": _string(
                compute_capability, "environment.compute_capability"
            ),
            "cuda_version": _string(
                manifest.get("torch_cuda"), "environment.torch_cuda"
            ),
            "pytorch_version": _string(
                manifest.get("pytorch"), "environment.pytorch"
            ),
            "flashinfer_version": _environment_package(
                manifest, "flashinfer-python", default="not-installed"
            ),
            "vllm_version": vllm_version,
        },
        "seed": settings.seed,
        "command": command,
        "raw_samples": {
            "reference": f"external://quality/{raw_path.name}",
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        },
        "gate": gate_payload,
    }
    write_quality_result(path, payload)


def _quality_command(args: argparse.Namespace) -> str:
    parts = [
        "python scripts/run_quality_eval.py",
        f"--model {args.model}",
        f"--mode {args.mode}",
        f"--output external://quality/{args.output.name}",
    ]
    if args.baseline is not None:
        parts.append(f"--baseline external://quality/{args.baseline.name}")
    return " ".join(parts)


def _baseline_path(args: argparse.Namespace) -> Path:
    if args.baseline is not None:
        return cast(Path, args.baseline)
    suffix = "-unfused"
    if args.output.stem.endswith(suffix):
        baseline_stem = args.output.stem[: -len(suffix)] + "-bf16"
        inferred = args.output.with_name(baseline_stem + args.output.suffix)
        if inferred.is_file():
            return cast(Path, inferred)
    raise SystemExit(
        "nvfp4-unfused mode requires --baseline or a sibling *-bf16.json result"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed Qwen quality evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("bf16", "nvfp4-unfused"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evidence-output",
        type=Path,
        help="EvidenceRecord wrapper path (defaults beside --output)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help=(
            "BF16 raw result for nvfp4-unfused mode; inferred from a sibling "
            "*-bf16.json when omitted"
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_quality_settings(args.config, args.model)
    payload = run_quality_evaluation(settings, mode=args.mode, device=args.device)
    gate: QualityGateResult | None = None
    if args.mode == "nvfp4-unfused":
        baseline_path = _baseline_path(args)
        baseline = _mapping(
            json.loads(baseline_path.read_text(encoding="utf-8")), "baseline"
        )
        comparison, gate = build_quality_comparison(baseline, payload, settings)
        payload["comparison"] = comparison
    elif args.baseline is not None:
        raise SystemExit("bf16 mode does not accept --baseline")
    write_quality_result(args.output, payload)
    evidence_output = args.evidence_output or args.output.with_name(
        f"{args.output.stem}-evidence.json"
    )
    write_quality_evidence(
        evidence_output,
        raw_path=args.output,
        settings=settings,
        command=_quality_command(args),
        gate=gate,
    )
    console_payload = dict(payload)
    console_payload.pop("token_trace", None)
    print(json.dumps(console_payload, indent=2, sort_keys=True))
    if gate is not None:
        return 0 if gate.passed else 1
    return 0 if not settings.finite_required or payload["all_finite"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
