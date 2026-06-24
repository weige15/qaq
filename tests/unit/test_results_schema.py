import json
from dataclasses import replace
from pathlib import Path

import pytest

from qaq.config import RunConfig
from qaq.report import build_report
from qaq.results import (
    COMPARISON_REQUIRED_MODES,
    ResultArtifact,
    ResultValidationError,
    build_report_rows,
    build_result_artifact,
    group_result_artifacts,
    load_result_artifact,
    save_result_artifact,
    validate_comparison,
    validate_paper_reproduction_claim,
    validate_result_artifact,
)
from qaq.runtime.static import run_static_runtime


def _config(tmp_path: Path, mode: str, **overrides: object) -> RunConfig:
    data = {
        "model": "fake-qaq-smoke-model",
        "tokenizer": "fake-qaq-smoke-tokenizer",
        "dataset": "fake_smoke",
        "split": "validation",
        "mode": mode,
        "precision_candidates": [4, 8],
        "max_bit_width": 8,
        "block_granularity": "mha_ffn",
        "device": "cpu",
        "gpu_ids": [],
        "seed": 0,
        "output_dir": str(tmp_path / mode),
        "overwrite": False,
        "logging": {"console": False},
        "prompt_format": "fake_smoke_v1",
        "metric": "exact_match",
    }
    data.update(overrides)
    return RunConfig.from_mapping(data, validate_output=False)


def _artifact(
    mode: str,
    *,
    model: str = "LLaMA-3.1-8B",
    tokenizer: str = "LLaMA-3.1-8B",
    dataset: str = "HellaSwag",
    split: str = "validation",
    prompt_format: str = "hellaswag_v1",
    metric: str = "exact_match",
    seed: int = 7,
    completion_status: str = "completed",
    diagnostic: bool = False,
    routing_summary: dict[str, object] | None = None,
    loader_summary: dict[str, object] | None = None,
    score: float | None = 0.75,
    perplexity: float | None = None,
    latency_seconds: float = 1.25,
    peak_gpu_memory_gb: float = 10.5,
) -> ResultArtifact:
    if mode in {"qaq_on_demand_off", "qaq_on_demand_on"} and routing_summary is None:
        routing_summary = {
            "constant_global_precision": False,
            "constant_precision_flagged": False,
            "diagnostic": False,
            "decision_count": 8,
        }
    if mode == "qaq_on_demand_on" and loader_summary is None:
        loader_summary = {
            "loads": 4,
            "cache_hits": 2,
            "total_bytes_transferred": 128,
            "failures": 0,
        }
    metadata = (
        {"runtime_metadata": {"mixed_precision_forward_applied": True}}
        if mode in {"static_8bit", "static_4bit", "qaq_on_demand_off", "qaq_on_demand_on"}
        else {}
    )
    return ResultArtifact(
        schema_version="qaq.result.v1",
        result_id=f"{model}:{dataset}:{split}:{mode}:seed{seed}",
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        split=split,
        prompt_format=prompt_format,
        mode=mode,
        precision_candidates=(4, 8),
        max_bit_width=8,
        block_granularity="mha_ffn",
        seed=seed,
        gpu_ids=(0,),
        hardware={
            "gpu_model": "NVIDIA GeForce RTX 3090",
            "selected_gpu_ids": [0],
        },
        metric=metric,
        metrics={
            "quality": {
                "metric_name": metric,
                "primary_value": score if score is not None else perplexity,
                "higher_is_better": perplexity is None,
                "num_examples": 2,
                "score": score,
                "perplexity": perplexity,
                "average_loss": None,
                "source": "fixture",
            },
            "latency": {
                "end_to_end_seconds": latency_seconds,
                "event_count": 1,
                "events": [],
            },
            "memory": {
                "peak_gpu_memory_gb": peak_gpu_memory_gb,
                "event_count": 1,
                "events": [],
            },
        },
        score=score,
        perplexity=perplexity,
        latency_seconds=latency_seconds,
        peak_gpu_memory_gb=peak_gpu_memory_gb,
        routing_summary=routing_summary,
        loader_summary=loader_summary,
        log_paths={"eval_log": f"runs/{mode}/eval.jsonl"},
        log_events=(),
        completion_status=completion_status,
        diagnostic=diagnostic,
        constrained=False,
        metadata=metadata,
    )


def _accepted_matrix() -> tuple[ResultArtifact, ...]:
    return tuple(_artifact(mode) for mode in COMPARISON_REQUIRED_MODES)


def test_runtime_output_builds_valid_result_artifact(tmp_path: Path) -> None:
    config = _config(tmp_path, "fp16")
    runtime_output = run_static_runtime(config)

    artifact = build_result_artifact(config, runtime_output)

    validate_result_artifact(artifact)
    payload = artifact.as_dict()
    assert payload["schema_version"] == "qaq.result.v1"
    assert payload["mode"] == "fp16"
    assert payload["metrics"]["quality"]["metric_name"] == "exact_match"
    assert payload["latency_seconds"] >= 0
    assert payload["peak_gpu_memory_gb"] == 0.0
    assert payload["diagnostic"] is True


def test_result_artifact_roundtrip_is_deterministic_json(tmp_path: Path) -> None:
    artifact = _artifact("static_8bit")
    path = save_result_artifact(artifact, tmp_path / "result.json")

    loaded = load_result_artifact(path)

    assert loaded == artifact
    assert json.loads(path.read_text(encoding="utf-8")) == artifact.as_dict()


def test_result_schema_rejects_missing_required_fields() -> None:
    payload = _artifact("fp16").as_dict()
    del payload["latency_seconds"]

    with pytest.raises(ResultValidationError) as exc:
        validate_result_artifact(payload)

    assert exc.value.code == "missing_result_field"
    assert exc.value.field == "latency_seconds"


def test_comparison_accepts_complete_non_diagnostic_matrix() -> None:
    validation = validate_comparison(_accepted_matrix())

    assert validation.state == "accepted"
    assert validation.reasons == ()
    assert validation.missing_modes == ()


def test_comparison_rejects_missing_static_baselines() -> None:
    validation = validate_comparison(
        (
            _artifact("qaq_on_demand_off"),
            _artifact("qaq_on_demand_on"),
        )
    )

    assert validation.state == "invalid"
    assert "missing_required_modes:fp16,static_8bit,static_4bit" in validation.reasons


def test_comparison_rejects_setting_mismatches() -> None:
    artifacts = list(_accepted_matrix())
    artifacts[-1] = replace(artifacts[-1], tokenizer="different-tokenizer")

    validation = validate_comparison(tuple(artifacts))
    groups = group_result_artifacts(tuple(artifacts))

    assert validation.state == "invalid"
    assert "settings_mismatch" in validation.reasons
    assert len(groups) == 2


def test_comparison_rejects_missing_qaq_summaries() -> None:
    artifacts = list(_accepted_matrix())
    artifacts[-1] = replace(artifacts[-1], loader_summary=None)

    validation = validate_comparison(tuple(artifacts))

    assert validation.state == "invalid"
    assert "missing_loader_summary:qaq_on_demand_on" in validation.reasons


def test_comparison_rejects_quantized_results_without_mixed_weight_forward() -> None:
    artifacts = list(_accepted_matrix())
    artifacts[1] = replace(artifacts[1], metadata={})

    validation = validate_comparison(tuple(artifacts))

    assert validation.state == "invalid"
    assert "missing_mixed_weight_forward:static_8bit" in validation.reasons


def test_diagnostic_modes_cannot_satisfy_accepted_comparison() -> None:
    artifacts = tuple(replace(artifact, diagnostic=True) for artifact in _accepted_matrix())

    validation = validate_comparison(artifacts)

    assert validation.state == "diagnostic"
    assert "diagnostic_or_constrained_results" in validation.reasons


def test_full_reproduction_claim_requires_paper_scope_or_deviation_labels() -> None:
    validation = validate_paper_reproduction_claim(_accepted_matrix())
    deviation = validate_paper_reproduction_claim(
        _accepted_matrix(),
        deviation_labels=("single-model-smoke",),
    )

    assert validation.state == "invalid"
    assert any(reason.startswith("missing_paper_models:") for reason in validation.reasons)
    assert any(reason.startswith("missing_paper_datasets:") for reason in validation.reasons)
    assert deviation.state == "diagnostic"
    assert "paper_deviation:single-model-smoke" in deviation.reasons


def test_golden_result_artifact_validates() -> None:
    artifact = load_result_artifact("tests/golden/result_artifact_static.json")

    validate_result_artifact(artifact)
    assert artifact.mode == "static_8bit"
    assert artifact.score == 0.75


def test_golden_report_rows_are_stable() -> None:
    rows = build_report_rows(_accepted_matrix())
    expected = json.loads(Path("tests/golden/report_rows.json").read_text(encoding="utf-8"))

    assert list(rows) == expected


def test_report_cli_payload_groups_result_files(tmp_path: Path) -> None:
    paths = [
        save_result_artifact(artifact, tmp_path / f"{artifact.mode}.json")
        for artifact in _accepted_matrix()
    ]

    report = build_report(paths)

    assert report["schema_version"] == "qaq.report.v1"
    assert report["comparisons"][0]["validation"]["state"] == "accepted"
    assert len(report["rows"]) == 5
