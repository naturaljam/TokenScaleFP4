# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
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
) -> dict[str, Any]:
    if mode not in {"bf16", "nvfp4-unfused"}:
        raise ValueError("mode must be 'bf16' or 'nvfp4-unfused'")
    return {
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


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    settings: QualitySettings,
) -> tuple[float, int, bool]:
    import torch
    from datasets import load_dataset
    from torch.nn import functional

    dataset = load_dataset(
        settings.perplexity.dataset,
        name=settings.perplexity.subset,
        split=settings.perplexity.split,
        revision=settings.perplexity.revision,
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
            nll = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            )
            all_finite = all_finite and bool(torch.isfinite(nll).item())
            total_nll += float(nll.item())
            token_count += labels.numel()
    return (
        perplexity_from_totals(total_nll=total_nll, token_count=token_count),
        token_count,
        all_finite,
    )


def evaluate_gsm8k(
    model: Any,
    tokenizer: Any,
    settings: QualitySettings,
) -> tuple[float, int, bool]:
    import torch
    from datasets import load_dataset

    train = load_dataset(
        settings.gsm8k.dataset,
        name=settings.gsm8k.subset,
        split="train",
        revision=settings.gsm8k.revision,
    )
    test = load_dataset(
        settings.gsm8k.dataset,
        name=settings.gsm8k.subset,
        split=settings.gsm8k.split,
        revision=settings.gsm8k.revision,
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
    perplexity, perplexity_count, ppl_finite = evaluate_perplexity(
        model, tokenizer, settings
    )
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
    )


def write_quality_result(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed Qwen quality evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("bf16", "nvfp4-unfused"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_quality_settings(args.config, args.model)
    payload = run_quality_evaluation(settings, mode=args.mode, device=args.device)
    write_quality_result(args.output, payload)
    if settings.finite_required and payload["all_finite"] is not True:
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
