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
    artifact_ref_mode: str | None = None,
    mixed_precision_forward_applied: bool | None = None,
    gpu_selector_record: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
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
    if artifact_ref_mode is None:
        artifact_ref_mode = (
            "full_tensor_index"
            if mode in {"static_8bit", "static_4bit", "qaq_on_demand_off", "qaq_on_demand_on"}
            else "none"
        )
    if mixed_precision_forward_applied is None:
        mixed_precision_forward_applied = mode in {
            "static_8bit",
            "static_4bit",
            "qaq_on_demand_off",
            "qaq_on_demand_on",
        }
    if gpu_selector_record is None:
        gpu_selector_record = {
            "status": "selected",
            "selected_physical_gpu_ids": [0],
            "cuda_visible_devices": "0",
            "pytorch_logical_mapping": {"cuda:0": 0},
            "command": ["python", "-m", "qaq.evaluate"],
        }
    artifact_scope = (
        "full_runtime_tensor_index"
        if artifact_ref_mode == "full_tensor_index"
        else "not_applicable_fp16"
        if mode == "fp16"
        else "partial_or_legacy_runtime_index"
        if artifact_ref_mode in {"partial_tensor_index", "legacy_bit_width_index"}
        else "missing_or_not_used"
    )
    metadata = dict(
        metadata
        or {
            "runtime_metadata": {
                "mixed_precision_forward_applied": mixed_precision_forward_applied,
                "artifact_ref_mode": artifact_ref_mode,
            }
        }
    )
    dataset_is_fake = any(
        token in dataset.lower()
        for token in ("fake", "smoke", "fixture", "synthetic", "toy", "tiny")
    )
    model_is_fake = any(
        token in model.lower()
        for token in ("fake", "smoke", "fixture", "synthetic", "toy", "tiny")
    )
    rejection_reasons: list[str] = []
    if completion_status != "completed":
        rejection_reasons.append("incomplete_result")
    if diagnostic:
        rejection_reasons.append("diagnostic_result")
    if dataset_is_fake:
        rejection_reasons.append("fake_dataset")
    if model_is_fake:
        rejection_reasons.append("fake_model")
    metadata_text = json.dumps(metadata, sort_keys=True).lower()
    if any(
        token in " ".join((dataset, split, artifact_scope, metadata_text)).lower()
        for token in (
            "fake_smoke",
            "health_check",
            "diagnostic_training",
            "router_health_check",
            "fixture",
            "tiny",
            "synthetic",
            "sampled_weight_values",
            "truncated",
        )
    ):
        rejection_reasons.append("smoke_fixture_or_synthetic_data")
    for value in (metadata.get("runtime_metadata"), metadata.get("raw_output_metadata")):
        if isinstance(value, dict) and (
            value.get("subset_run") is True or value.get("max_examples") is not None
        ):
            rejection_reasons.append("benchmark_subset_not_full_acceptance")
            break
    if mode in {"static_8bit", "static_4bit", "qaq_on_demand_off", "qaq_on_demand_on"}:
        if not mixed_precision_forward_applied:
            rejection_reasons.append("mixed_precision_forward_not_applied")
        if artifact_ref_mode in {"partial_tensor_index", "legacy_bit_width_index"}:
            rejection_reasons.append(f"unaccepted_artifact_ref_mode:{artifact_ref_mode}")
        elif artifact_ref_mode != "full_tensor_index":
            rejection_reasons.append(f"missing_full_tensor_artifact_index:{artifact_ref_mode}")
    if "llama" in model.lower() and not gpu_selector_record:
        rejection_reasons.append("missing_gpu_selector_record")
    rejection_reasons = list(dict.fromkeys(rejection_reasons))
    evidence_level = (
        "diagnostic_health_check"
        if any(
            reason
            in {
                "diagnostic_result",
                "fake_dataset",
                "fake_model",
                "smoke_fixture_or_synthetic_data",
            }
            for reason in rejection_reasons
        )
        else "real_path_implemented"
        if rejection_reasons
        else "accepted_experiment_result"
    )
    return ResultArtifact(
        schema_version="qaq.result.v1",
        result_id=f"{model}:{dataset}:{split}:{mode}:seed{seed}",
        evidence_level=evidence_level,
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
            "gpu_selector_record": gpu_selector_record,
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
                "source": "schema_fixture_not_benchmark_claim",
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
        dataset_is_fake=dataset_is_fake,
        model_is_fake=model_is_fake,
        artifact_scope=artifact_scope,
        artifact_ref_mode=artifact_ref_mode,
        mixed_precision_forward_applied=mixed_precision_forward_applied,
        benchmark_name=dataset,
        benchmark_split=split,
        gpu_selector_record=gpu_selector_record,
        accepted_as_qaq_result=not rejection_reasons,
        rejection_reasons=tuple(rejection_reasons),
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
    assert payload["evidence_level"] == "diagnostic_health_check"
    assert payload["accepted_as_qaq_result"] is False
    assert "fake_dataset" in payload["rejection_reasons"]


def test_real_benchmark_subset_is_non_diagnostic_but_not_full_acceptance() -> None:
    artifact = _artifact(
        "fp16",
        metadata={
            "runtime_metadata": {
                "artifact_ref_mode": "none",
                "mixed_precision_forward_applied": False,
                "max_examples": 128,
                "subset_run": True,
            },
            "raw_output_metadata": {
                "max_examples": 128,
                "subset_run": True,
            },
        },
    )

    validate_result_artifact(artifact)

    assert artifact.diagnostic is False
    assert artifact.evidence_level == "real_path_implemented"
    assert artifact.accepted_as_qaq_result is False
    assert "benchmark_subset_not_full_acceptance" in artifact.rejection_reasons
    assert "diagnostic_result" not in artifact.rejection_reasons


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


def test_result_schema_rejects_accepted_artifact_with_extra_rejections() -> None:
    payload = _artifact("fp16").as_dict()
    payload["rejection_reasons"] = ["manual_rejection"]

    with pytest.raises(ResultValidationError) as exc:
        validate_result_artifact(payload)

    assert exc.value.code == "invalid_acceptance_contract"
    assert exc.value.field == "rejection_reasons"


def test_comparison_accepts_only_synthetic_metadata_that_satisfies_contract() -> None:
    artifacts = _accepted_matrix()
    validation = validate_comparison(artifacts)

    assert validation.state == "accepted"
    assert validation.reasons == ()
    assert validation.missing_modes == ()
    assert all(artifact.evidence_level == "accepted_experiment_result" for artifact in artifacts)
    assert all(artifact.accepted_as_qaq_result is True for artifact in artifacts)


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
    artifacts[1] = _artifact("static_8bit", mixed_precision_forward_applied=False)

    validation = validate_comparison(tuple(artifacts))

    assert validation.state == "invalid"
    assert "missing_mixed_weight_forward:static_8bit" in validation.reasons
    assert any(
        reason.startswith("artifact_not_accepted:static_8bit:")
        for reason in validation.reasons
    )


def test_diagnostic_modes_cannot_satisfy_accepted_comparison() -> None:
    artifacts = tuple(
        _artifact(mode, diagnostic=True) for mode in COMPARISON_REQUIRED_MODES
    )

    validation = validate_comparison(artifacts)

    assert validation.state == "diagnostic"
    assert "diagnostic_or_constrained_results" in validation.reasons
    assert all(artifact.accepted_as_qaq_result is False for artifact in artifacts)


def test_router_health_check_result_is_never_accepted() -> None:
    artifact = _artifact(
        "qaq_on_demand_off",
        metadata={
            "runtime_metadata": {
                "mixed_precision_forward_applied": True,
                "artifact_ref_mode": "full_tensor_index",
                "router_health_check": True,
            }
        },
    )

    validate_result_artifact(artifact)

    assert artifact.evidence_level == "diagnostic_health_check"
    assert artifact.accepted_as_qaq_result is False
    assert "smoke_fixture_or_synthetic_data" in artifact.rejection_reasons


def test_partial_tensor_artifact_result_is_never_accepted() -> None:
    artifact = _artifact(
        "qaq_on_demand_off",
        artifact_ref_mode="partial_tensor_index",
        mixed_precision_forward_applied=True,
    )

    validation = validate_comparison(
        (
            _artifact("fp16"),
            _artifact("static_8bit"),
            _artifact("static_4bit"),
            artifact,
            _artifact("qaq_on_demand_on"),
        )
    )

    assert artifact.accepted_as_qaq_result is False
    assert "unaccepted_artifact_ref_mode:partial_tensor_index" in artifact.rejection_reasons
    assert validation.state == "invalid"


def test_report_rejects_mixed_fake_and_real_artifacts(tmp_path: Path) -> None:
    artifacts = list(_accepted_matrix())
    artifacts[0] = _artifact(
        "fp16",
        model="fake-qaq-smoke-model",
        tokenizer="fake-qaq-smoke-tokenizer",
        dataset="fake_smoke",
        prompt_format="fake_smoke_v1",
    )
    paths = [
        save_result_artifact(artifact, tmp_path / f"{index}-{artifact.mode}.json")
        for index, artifact in enumerate(artifacts)
    ]

    report = build_report(paths)

    states = [comparison["validation"]["state"] for comparison in report["comparisons"]]
    reasons = [
        reason
        for comparison in report["comparisons"]
        for reason in comparison["validation"]["reasons"]
    ]
    assert "accepted" not in states
    assert any(reason.startswith("missing_required_modes:") for reason in reasons)


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
