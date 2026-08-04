# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import nn

from tokenscalefp4.reporting import EvidenceRecord

REPO_ROOT = Path(__file__).parents[2]


def load_script(name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / name
    if not path.is_file():
        pytest.fail(f"Task 6 reporting script is missing: {path.name}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quality_payload(
    *,
    mode: str,
    perplexity: float,
    gsm8k_accuracy: float,
    all_finite: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": "Qwen/Qwen2.5-1.5B",
        "model_revision": "1" * 40,
        "mode": mode,
        "seed": 20260803,
        "perplexity": {
            "dataset": "Salesforce/wikitext",
            "revision": "2" * 40,
            "split": "test",
            "sample_count": 245569,
            "value": perplexity,
        },
        "gsm8k": {
            "dataset": "openai/gsm8k",
            "revision": "3" * 40,
            "split": "test",
            "sample_count": 1319,
            "accuracy": gsm8k_accuracy,
        },
        "all_finite": all_finite,
    }


def add_token_trace(
    payload: dict[str, object],
    *,
    greedy_token_ids: list[int],
    target_logprobs: list[float],
) -> dict[str, object]:
    payload["token_trace"] = {
        "greedy_token_ids": greedy_token_ids,
        "target_logprobs": target_logprobs,
    }
    return payload


def test_quality_gate_accepts_exact_thresholds() -> None:
    quality = load_script("run_quality_eval.py")
    baseline = quality_payload(mode="bf16", perplexity=10.0, gsm8k_accuracy=0.50)
    candidate = quality_payload(
        mode="nvfp4-unfused",
        perplexity=10.5,
        gsm8k_accuracy=0.48,
    )

    result = quality.evaluate_quality_gates(
        baseline,
        candidate,
        max_relative_perplexity_increase=0.05,
        max_absolute_gsm8k_drop=0.02,
        finite_required=True,
    )

    assert result.perplexity_relative_increase == pytest.approx(0.05)
    assert result.gsm8k_absolute_drop == pytest.approx(0.02)
    assert result.perplexity_passed is True
    assert result.gsm8k_passed is True
    assert result.finite_passed is True
    assert result.passed is True


@pytest.mark.parametrize(
    ("candidate_overrides", "failed_gate"),
    [
        ({"perplexity": 10.5001}, "perplexity_passed"),
        ({"gsm8k_accuracy": 0.4799}, "gsm8k_passed"),
        ({"all_finite": False}, "finite_passed"),
    ],
)
def test_quality_gate_rejects_threshold_and_finite_failures(
    candidate_overrides: dict[str, object],
    failed_gate: str,
) -> None:
    quality = load_script("run_quality_eval.py")
    baseline = quality_payload(mode="bf16", perplexity=10.0, gsm8k_accuracy=0.50)
    values: dict[str, object] = {
        "mode": "nvfp4-unfused",
        "perplexity": 10.5,
        "gsm8k_accuracy": 0.48,
        "all_finite": True,
    }
    values.update(candidate_overrides)
    candidate = quality_payload(**values)

    result = quality.evaluate_quality_gates(
        baseline,
        candidate,
        max_relative_perplexity_increase=0.05,
        max_absolute_gsm8k_drop=0.02,
        finite_required=True,
    )

    assert getattr(result, failed_gate) is False
    assert result.passed is False


def test_quality_gate_rejects_mismatched_evaluation_metadata() -> None:
    quality = load_script("run_quality_eval.py")
    baseline = quality_payload(mode="bf16", perplexity=10.0, gsm8k_accuracy=0.50)
    candidate = quality_payload(
        mode="nvfp4-unfused",
        perplexity=10.2,
        gsm8k_accuracy=0.49,
    )
    candidate["model_revision"] = "f" * 40

    with pytest.raises(ValueError, match="model_revision"):
        quality.evaluate_quality_gates(
            baseline,
            candidate,
            max_relative_perplexity_increase=0.05,
            max_absolute_gsm8k_drop=0.02,
            finite_required=True,
        )


def test_token_comparison_reports_agreement_and_logprob_drift() -> None:
    quality = load_script("run_quality_eval.py")
    baseline = add_token_trace(
        quality_payload(mode="bf16", perplexity=10.0, gsm8k_accuracy=0.50),
        greedy_token_ids=[2, 4, 6],
        target_logprobs=[-1.0, -2.0, -3.0],
    )
    candidate = add_token_trace(
        quality_payload(
            mode="nvfp4-unfused", perplexity=10.2, gsm8k_accuracy=0.49
        ),
        greedy_token_ids=[2, 5, 6],
        target_logprobs=[-1.1, -1.8, -3.3],
    )

    comparison = quality.compare_token_traces(baseline, candidate)

    assert comparison == {
        "sample_count": 3,
        "greedy_token_agreement": pytest.approx(2 / 3),
        "mean_absolute_target_logprob_drift": pytest.approx(0.2),
    }


def test_bad_candidate_makes_cli_write_verdict_and_exit_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = load_script("run_quality_eval.py")
    baseline = add_token_trace(
        quality_payload(mode="bf16", perplexity=10.0, gsm8k_accuracy=0.50),
        greedy_token_ids=[2],
        target_logprobs=[-1.0],
    )
    candidate = add_token_trace(
        quality_payload(
            mode="nvfp4-unfused", perplexity=11.0, gsm8k_accuracy=0.40
        ),
        greedy_token_ids=[3],
        target_logprobs=[-2.0],
    )
    baseline_path = tmp_path / "bf16.json"
    output_path = tmp_path / "unfused.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    settings = quality.load_quality_settings(
        REPO_ROOT / "configs" / "evals" / "quality.json",
        "Qwen/Qwen2.5-1.5B",
    )
    monkeypatch.setattr(
        quality,
        "parse_args",
        lambda: Namespace(
            model=settings.model,
            mode="nvfp4-unfused",
            output=output_path,
            evidence_output=tmp_path / "evidence.json",
            baseline=baseline_path,
            config=REPO_ROOT / "configs" / "evals" / "quality.json",
            device="cuda",
        ),
    )
    monkeypatch.setattr(quality, "run_quality_evaluation", lambda *_a, **_k: candidate)
    monkeypatch.setattr(quality, "write_quality_evidence", lambda *_a, **_k: None)

    assert quality.main() == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["comparison"]["gates"] == {
        "perplexity_passed": False,
        "gsm8k_passed": False,
        "finite_passed": True,
        "passed": False,
    }


def test_quality_evidence_wrapper_round_trips_frozen_schema(tmp_path: Path) -> None:
    quality = load_script("run_quality_eval.py")
    settings = quality.load_quality_settings(
        REPO_ROOT / "configs" / "evals" / "quality.json",
        "Qwen/Qwen2.5-1.5B",
    )
    raw_path = tmp_path / "quality.json"
    raw_path.write_text('{"all_finite": true}\n', encoding="utf-8")
    evidence_path = tmp_path / "quality-evidence.json"
    environment = {
        "flashinfer_sha": "1" * 40,
        "vllm_sha": "2" * 40,
        "gpu_product_name": "NVIDIA GeForce RTX 5070 Laptop GPU",
        "compute_capability": "12.0",
        "torch_cuda": "13.0",
        "pytorch": "2.11.0+cu130",
        "packages": {"flashinfer-python": "0.6.17"},
    }
    gate = quality.QualityGateResult(0.03, 0.01, True, True, True)

    quality.write_quality_evidence(
        evidence_path,
        raw_path=raw_path,
        settings=settings,
        command="python scripts/run_quality_eval.py --mode nvfp4-unfused",
        environment=environment,
        gate=gate,
    )

    record = EvidenceRecord.from_json(evidence_path)
    assert record.raw_samples.reference == "external://quality/quality.json"
    assert record.gate.name == "unfused_quality_feasibility"
    assert record.gate.passed is True


def test_memory_gate_uses_actual_allocated_quantized_bytes() -> None:
    memory = load_script("run_memory_report.py")

    result = memory.MemoryResult(
        model="Qwen/Qwen2.5-1.5B",
        model_revision="1" * 40,
        eligible_weight_count=14,
        bf16_bytes=1000,
        packed_bytes=250,
        block_scale_bytes=30,
        global_scale_bytes=10,
        padding_bytes=20,
        steady_state_cuda_bytes=500,
        peak_cuda_bytes=700,
    )

    assert result.quantized_allocated_bytes == 310
    assert result.reduction == pytest.approx(0.69)
    result.require(minimum_reduction=0.65)


def test_memory_gate_rejects_reduction_below_threshold() -> None:
    memory = load_script("run_memory_report.py")
    result = memory.MemoryResult(
        model="Qwen/Qwen2.5-1.5B",
        model_revision="1" * 40,
        eligible_weight_count=1,
        bf16_bytes=1000,
        packed_bytes=300,
        block_scale_bytes=40,
        global_scale_bytes=10,
        padding_bytes=10,
        steady_state_cuda_bytes=500,
        peak_cuda_bytes=700,
    )

    with pytest.raises(SystemExit, match="65.00%"):
        result.require(minimum_reduction=0.65)


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    ],
)
def test_qwen_dense_projection_names_are_eligible(name: str) -> None:
    quality = load_script("run_quality_eval.py")

    assert quality.is_eligible_linear(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "model.embed_tokens",
        "model.layers.0.input_layernorm",
        "lm_head",
        "model.layers.0.self_attn.rotary_emb",
    ],
)
def test_non_projection_layers_are_not_eligible(name: str) -> None:
    quality = load_script("run_quality_eval.py")

    assert quality.is_eligible_linear(name) is False


def test_quality_settings_require_pinned_model_and_dataset_revisions(
    tmp_path: Path,
) -> None:
    quality = load_script("run_quality_eval.py")
    config = json.loads(
        (REPO_ROOT / "configs" / "evals" / "quality.json").read_text(
            encoding="utf-8"
        )
    )
    config["gsm8k"]["revision"] = "main"
    path = tmp_path / "quality.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable 40-character SHA"):
        quality.load_quality_settings(path, "Qwen/Qwen2.5-1.5B")


def test_offline_dataset_loading_resolves_pinned_snapshot_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = load_script("run_quality_eval.py")
    settings = quality.EvaluationSettings(
        dataset="Salesforce/wikitext",
        subset="wikitext-2-raw-v1",
        split="test",
        revision="2" * 40,
        seed=20260803,
    )
    data_dir = tmp_path / settings.subset
    data_dir.mkdir()
    parquet = data_dir / "test-00000-of-00001.parquet"
    parquet.write_bytes(b"fixture")
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_loader(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        return "dataset"

    def fake_snapshot(**kwargs: object) -> str:
        assert kwargs == {
            "repo_id": settings.dataset,
            "repo_type": "dataset",
            "revision": settings.revision,
            "local_files_only": True,
        }
        return str(tmp_path)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    result = quality.load_dataset_split(
        settings,
        split="test",
        loader=fake_loader,
        snapshot_resolver=fake_snapshot,
    )

    assert result == "dataset"
    assert calls == [
        (
            ("parquet",),
            {"data_files": {"test": [str(parquet)]}, "split": "test"},
        )
    ]


def test_quality_payload_records_pinned_inputs_and_full_sample_counts() -> None:
    quality = load_script("run_quality_eval.py")
    settings = quality.load_quality_settings(
        REPO_ROOT / "configs" / "evals" / "quality.json",
        "Qwen/Qwen2.5-1.5B",
    )

    payload = quality.build_quality_payload(
        settings,
        mode="nvfp4-unfused",
        perplexity=12.5,
        perplexity_sample_count=245569,
        gsm8k_accuracy=0.42,
        gsm8k_sample_count=1319,
        all_finite=True,
    )

    assert payload["model_revision"] == "8faed761d45a263340a0528343f099c05c9a4323"
    assert payload["perplexity"]["revision"] == (
        "b08601e04326c79dfdd32d625aee71d232d685c3"
    )
    assert payload["perplexity"]["sample_count"] == 245569
    assert payload["gsm8k"]["revision"] == (
        "740312add88f781978c0658806c59bc2815b9866"
    )
    assert payload["gsm8k"]["sample_count"] == 1319
    assert payload["all_finite"] is True


def test_nvfp4_storage_breakdown_counts_layout_padding() -> None:
    memory = load_script("run_memory_report.py")

    storage = memory.nvfp4_storage_breakdown(rows=129, columns=96)

    assert storage.packed_bytes == 129 * 48
    assert storage.block_scale_bytes == 129 * 6
    assert storage.global_scale_bytes == 4
    assert storage.padding_bytes == (256 * 8) - (129 * 6)
    assert storage.allocated_bytes == (129 * 48) + (256 * 8) + 4


def test_replace_eligible_linears_preserves_excluded_modules() -> None:
    quality = load_script("run_quality_eval.py")

    class Marker(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value

    model = nn.Module()
    model.layers = nn.ModuleList(
        [
            nn.ModuleDict(
                {
                    "q_proj": nn.Linear(32, 8),
                    "input_layernorm": nn.LayerNorm(32),
                }
            )
        ]
    )
    model.lm_head = nn.Linear(8, 32)

    replaced = quality.replace_eligible_linears(
        model,
        factory=lambda _linear: Marker(),
    )

    assert replaced == ("layers.0.q_proj",)
    assert isinstance(model.layers[0]["q_proj"], Marker)
    assert isinstance(model.layers[0]["input_layernorm"], nn.LayerNorm)
    assert isinstance(model.lm_head, nn.Linear)


def test_perplexity_from_nll_uses_token_count() -> None:
    quality = load_script("run_quality_eval.py")

    assert quality.perplexity_from_totals(total_nll=4.0, token_count=2) == pytest.approx(
        7.38905609893065
    )


def test_target_logprobs_match_log_softmax_without_full_probability_tensor() -> None:
    quality = load_script("run_quality_eval.py")
    logits = torch.tensor(
        [[[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 2]])

    greedy, selected = quality.select_token_metrics(logits, labels)

    expected = torch.log_softmax(logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    assert torch.equal(greedy, torch.tensor([[1, 2]]))
    assert torch.allclose(selected, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("The answer is 1,200. #### 1,200", "1200"),
        ("work... #### -$7.50", "-7.50"),
    ],
)
def test_extract_gsm8k_answer_normalizes_final_marker(
    generated: str,
    expected: str,
) -> None:
    quality = load_script("run_quality_eval.py")

    assert quality.extract_gsm8k_answer(generated) == expected


def test_build_gsm8k_prompt_uses_fixed_five_shot_examples() -> None:
    quality = load_script("run_quality_eval.py")
    examples = [
        {"question": f"example {index}", "answer": f"reasoning #### {index}"}
        for index in range(5)
    ]

    prompt = quality.build_gsm8k_prompt(
        examples,
        {"question": "target", "answer": "unused"},
    )

    assert prompt.count("Question:") == 6
    assert prompt.count("Answer:") == 6
    assert prompt.endswith("Question: target\nAnswer:")


def test_memory_payload_records_steady_state_and_peak_bytes() -> None:
    memory = load_script("run_memory_report.py")
    result = memory.MemoryResult(
        model="Qwen/Qwen2.5-1.5B",
        model_revision="1" * 40,
        eligible_weight_count=2,
        bf16_bytes=1000,
        packed_bytes=250,
        block_scale_bytes=30,
        global_scale_bytes=10,
        padding_bytes=20,
        steady_state_cuda_bytes=500,
        peak_cuda_bytes=700,
    )

    payload = memory.build_memory_payload(result)

    assert payload == {
        "schema_version": 1,
        "model": "Qwen/Qwen2.5-1.5B",
        "model_revision": "1" * 40,
        "eligible_weight_count": 2,
        "bf16_bytes": 1000,
        "packed_bytes": 250,
        "block_scale_bytes": 30,
        "global_scale_bytes": 10,
        "padding_bytes": 20,
        "quantized_allocated_bytes": 310,
        "reduction": pytest.approx(0.69),
        "steady_state_cuda_bytes": 500,
        "peak_cuda_bytes": 700,
    }
