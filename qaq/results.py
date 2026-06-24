"""Result artifact schema and QAQ comparison validation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qaq.config import QAQ_MODES, RunConfig, VALID_MODES
from qaq.manifest import RunManifest
from qaq.metrics import (
    MetricAggregationError,
    compute_quality_metric,
    summarize_latency,
    summarize_memory,
)
from qaq.runtime.common import RuntimeOutputBundle


RESULT_SCHEMA_VERSION = "qaq.result.v1"
EVIDENCE_LEVELS = frozenset(
    {
        "diagnostic_health_check",
        "real_path_implemented",
        "accepted_experiment_result",
    }
)
COMPARISON_REQUIRED_MODES = (
    "fp16",
    "static_8bit",
    "static_4bit",
    "qaq_on_demand_off",
    "qaq_on_demand_on",
)
COMPARISON_STATES = frozenset({"accepted", "diagnostic", "incomplete", "invalid"})
MIXED_PRECISION_REQUIRED_MODES = frozenset(
    {"static_8bit", "static_4bit", "qaq_on_demand_off", "qaq_on_demand_on"}
)
REJECTED_ARTIFACT_REF_MODES = frozenset(
    {"partial_tensor_index", "legacy_bit_width_index"}
)
PAPER_MODELS = frozenset({"Qwen3-4B", "Qwen3-8B", "LLaMA-3.1-8B"})
PAPER_DATASETS = frozenset(
    {"HellaSwag", "PIQA", "ARC-E", "ARC-C", "WinoGrande", "WikiText-2", "PTB"}
)


@dataclass(slots=True)
class ResultValidationError(ValueError):
    """Raised when a result artifact or comparison fails validation."""

    code: str
    message: str
    field: str | None = None

    def __str__(self) -> str:
        if self.field:
            return f"{self.code}: {self.field}: {self.message}"
        return f"{self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class ResultArtifact:
    """Machine-readable result for one runtime mode."""

    schema_version: str
    result_id: str
    evidence_level: str
    model: str
    tokenizer: str
    dataset: str
    split: str
    prompt_format: str
    mode: str
    precision_candidates: tuple[int, ...]
    max_bit_width: int
    block_granularity: str
    seed: int
    gpu_ids: tuple[int, ...]
    hardware: dict[str, Any]
    metric: str
    metrics: dict[str, Any]
    score: float | None
    perplexity: float | None
    latency_seconds: float
    peak_gpu_memory_gb: float
    routing_summary: dict[str, Any] | None
    loader_summary: dict[str, Any] | None
    log_paths: dict[str, str]
    log_events: tuple[dict[str, Any], ...]
    completion_status: str
    diagnostic: bool
    dataset_is_fake: bool
    model_is_fake: bool
    artifact_scope: str
    artifact_ref_mode: str
    mixed_precision_forward_applied: bool
    benchmark_name: str
    benchmark_split: str
    gpu_selector_record: dict[str, Any] | None
    accepted_as_qaq_result: bool
    rejection_reasons: tuple[str, ...]
    constrained: bool = False
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "evidence_level": self.evidence_level,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "dataset": self.dataset,
            "split": self.split,
            "prompt_format": self.prompt_format,
            "mode": self.mode,
            "precision_candidates": list(self.precision_candidates),
            "max_bit_width": self.max_bit_width,
            "block_granularity": self.block_granularity,
            "seed": self.seed,
            "gpu_ids": list(self.gpu_ids),
            "hardware": dict(self.hardware),
            "metric": self.metric,
            "metrics": dict(self.metrics),
            "score": self.score,
            "perplexity": self.perplexity,
            "latency_seconds": self.latency_seconds,
            "peak_gpu_memory_gb": self.peak_gpu_memory_gb,
            "routing_summary": self.routing_summary,
            "loader_summary": self.loader_summary,
            "log_paths": dict(self.log_paths),
            "log_events": list(self.log_events),
            "completion_status": self.completion_status,
            "diagnostic": self.diagnostic,
            "dataset_is_fake": self.dataset_is_fake,
            "model_is_fake": self.model_is_fake,
            "artifact_scope": self.artifact_scope,
            "artifact_ref_mode": self.artifact_ref_mode,
            "mixed_precision_forward_applied": self.mixed_precision_forward_applied,
            "benchmark_name": self.benchmark_name,
            "benchmark_split": self.benchmark_split,
            "gpu_selector_record": (
                dict(self.gpu_selector_record)
                if self.gpu_selector_record is not None
                else None
            ),
            "accepted_as_qaq_result": self.accepted_as_qaq_result,
            "rejection_reasons": list(self.rejection_reasons),
            "constrained": self.constrained,
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ComparisonKey:
    """Fields that must match for a valid cross-mode comparison."""

    model: str
    tokenizer: str
    dataset: str
    split: str
    benchmark_name: str
    benchmark_split: str
    prompt_format: str
    metric: str
    precision_candidates: tuple[int, ...]
    seed: int
    block_granularity: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "dataset": self.dataset,
            "split": self.split,
            "benchmark_name": self.benchmark_name,
            "benchmark_split": self.benchmark_split,
            "prompt_format": self.prompt_format,
            "metric": self.metric,
            "precision_candidates": list(self.precision_candidates),
            "seed": self.seed,
            "block_granularity": self.block_granularity,
        }


@dataclass(frozen=True, slots=True)
class ComparisonValidation:
    """Validation state for one group of comparable result artifacts."""

    key: ComparisonKey | None
    state: str
    reasons: tuple[str, ...]
    present_modes: tuple[str, ...]
    missing_modes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.as_dict() if self.key else None,
            "state": self.state,
            "reasons": list(self.reasons),
            "present_modes": list(self.present_modes),
            "missing_modes": list(self.missing_modes),
        }


def build_result_artifact(
    config: RunConfig,
    output: RuntimeOutputBundle,
    *,
    manifest: RunManifest | None = None,
    result_id: str | None = None,
    log_paths: Mapping[str, str | Path] | None = None,
) -> ResultArtifact:
    """Convert a runtime output bundle into a validated result artifact."""

    try:
        quality = compute_quality_metric(output.raw_output, metric_name=config.metric)
        latency = summarize_latency(output.latency_events)
        memory = summarize_memory(output.memory_events)
    except MetricAggregationError as exc:
        raise ResultValidationError(exc.code, exc.message) from exc

    hardware = (
        manifest.hardware.as_dict()
        if manifest
        else {
            "device": config.device,
            "selected_gpu_ids": list(config.gpu_ids),
            "measurement_source": "runtime_metadata",
        }
    )
    resolved_log_paths = _collect_log_paths(config, manifest=manifest, log_paths=log_paths)
    notes = tuple(note for note in (config.notes,) if note)
    runtime_metadata = dict(output.metadata)
    raw_output_metadata = dict(output.raw_output.metadata)
    metadata = {
        "runtime_metadata": runtime_metadata,
        "raw_output_metadata": raw_output_metadata,
        "runtime_status": output.status,
    }
    artifact_ref_mode = str(runtime_metadata.get("artifact_ref_mode", "none"))
    mixed_precision_forward_applied = (
        runtime_metadata.get("mixed_precision_forward_applied") is True
        or raw_output_metadata.get("mixed_precision_forward_applied") is True
    )
    dataset_is_fake = _is_fake_dataset(config.dataset)
    model_is_fake = _is_fake_model(config.model)
    diagnostic = _is_diagnostic_result(config, output)
    artifact_scope = _artifact_scope(
        config,
        runtime_metadata=runtime_metadata,
        artifact_ref_mode=artifact_ref_mode,
    )
    benchmark_name = str(runtime_metadata.get("benchmark_name", config.dataset))
    benchmark_split = str(runtime_metadata.get("benchmark_split", config.split))
    gpu_selector_record = _gpu_selector_record(
        hardware,
        runtime_metadata=runtime_metadata,
    )
    rejection_reasons = _contract_rejection_reasons(
        mode=output.mode,
        completion_status=output.status,
        diagnostic=diagnostic,
        dataset_is_fake=dataset_is_fake,
        model_is_fake=model_is_fake,
        artifact_scope=artifact_scope,
        artifact_ref_mode=artifact_ref_mode,
        mixed_precision_forward_applied=mixed_precision_forward_applied,
        benchmark_name=benchmark_name,
        benchmark_split=benchmark_split,
        model=config.model,
        gpu_ids=config.gpu_ids,
        gpu_selector_record=gpu_selector_record,
        metadata=metadata,
    )
    artifact = ResultArtifact(
        schema_version=RESULT_SCHEMA_VERSION,
        result_id=result_id or _default_result_id(config),
        evidence_level=_evidence_level_for_rejections(rejection_reasons),
        model=config.model,
        tokenizer=config.tokenizer,
        dataset=config.dataset,
        split=config.split,
        prompt_format=config.prompt_format or "plain",
        mode=output.mode,
        precision_candidates=config.precision_candidates,
        max_bit_width=config.max_bit_width,
        block_granularity=config.block_granularity,
        seed=config.seed,
        gpu_ids=config.gpu_ids,
        hardware=hardware,
        metric=quality.metric_name,
        metrics={
            "quality": quality.as_dict(),
            "latency": latency,
            "memory": memory,
        },
        score=quality.score,
        perplexity=quality.perplexity,
        latency_seconds=float(latency["end_to_end_seconds"]),
        peak_gpu_memory_gb=float(memory["peak_gpu_memory_gb"]),
        routing_summary=_optional_dict(output.metadata.get("routing_summary")),
        loader_summary=_optional_dict(output.metadata.get("loader_summary")),
        log_paths=resolved_log_paths,
        log_events=tuple(dict(event) for event in output.log_events),
        completion_status=output.status,
        diagnostic=diagnostic,
        dataset_is_fake=dataset_is_fake,
        model_is_fake=model_is_fake,
        artifact_scope=artifact_scope,
        artifact_ref_mode=artifact_ref_mode,
        mixed_precision_forward_applied=mixed_precision_forward_applied,
        benchmark_name=benchmark_name,
        benchmark_split=benchmark_split,
        gpu_selector_record=gpu_selector_record,
        accepted_as_qaq_result=not rejection_reasons,
        rejection_reasons=rejection_reasons,
        constrained=config.device == "cpu",
        notes=notes,
        metadata=metadata,
    )
    validate_result_artifact(artifact)
    return artifact


def validate_result_artifact(artifact: ResultArtifact | Mapping[str, Any]) -> None:
    """Validate the result artifact schema without accepting scientific claims."""

    value = artifact.as_dict() if isinstance(artifact, ResultArtifact) else dict(artifact)
    required = (
        "schema_version",
        "result_id",
        "evidence_level",
        "model",
        "tokenizer",
        "dataset",
        "split",
        "prompt_format",
        "mode",
        "precision_candidates",
        "max_bit_width",
        "block_granularity",
        "seed",
        "gpu_ids",
        "hardware",
        "metric",
        "metrics",
        "latency_seconds",
        "peak_gpu_memory_gb",
        "routing_summary",
        "loader_summary",
        "log_paths",
        "completion_status",
        "diagnostic",
        "dataset_is_fake",
        "model_is_fake",
        "artifact_scope",
        "artifact_ref_mode",
        "mixed_precision_forward_applied",
        "benchmark_name",
        "benchmark_split",
        "gpu_selector_record",
        "accepted_as_qaq_result",
        "rejection_reasons",
    )
    for field_name in required:
        if field_name not in value:
            raise ResultValidationError(
                "missing_result_field",
                "required result artifact field is missing",
                field_name,
            )
    if value["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ResultValidationError(
            "unsupported_result_schema",
            f"expected {RESULT_SCHEMA_VERSION}",
            "schema_version",
        )
    if value["evidence_level"] not in EVIDENCE_LEVELS:
        raise ResultValidationError(
            "invalid_evidence_level",
            f"expected one of {sorted(EVIDENCE_LEVELS)}",
            "evidence_level",
        )
    if value["mode"] not in VALID_MODES:
        raise ResultValidationError("invalid_result_mode", "unknown runtime mode", "mode")
    _require_non_empty_string(value, "model")
    _require_non_empty_string(value, "tokenizer")
    _require_non_empty_string(value, "dataset")
    _require_non_empty_string(value, "split")
    _require_non_empty_string(value, "prompt_format")
    _require_non_empty_string(value, "metric")
    _require_int_sequence(value, "precision_candidates")
    if not isinstance(value["gpu_ids"], list):
        raise ResultValidationError("invalid_result_field", "gpu_ids must be a list", "gpu_ids")
    if not isinstance(value["hardware"], dict):
        raise ResultValidationError("invalid_result_field", "hardware must be an object", "hardware")
    if not isinstance(value["metrics"], dict):
        raise ResultValidationError("invalid_result_field", "metrics must be an object", "metrics")
    for metric_field in ("quality", "latency", "memory"):
        if metric_field not in value["metrics"]:
            raise ResultValidationError(
                "missing_metric_field",
                "metrics must include quality, latency, and memory summaries",
                f"metrics.{metric_field}",
            )
    if float(value["latency_seconds"]) < 0:
        raise ResultValidationError("invalid_latency", "latency must be non-negative", "latency_seconds")
    if float(value["peak_gpu_memory_gb"]) < 0:
        raise ResultValidationError(
            "invalid_memory",
            "peak GPU memory must be non-negative",
            "peak_gpu_memory_gb",
        )
    if not isinstance(value["log_paths"], dict):
        raise ResultValidationError("invalid_result_field", "log_paths must be an object", "log_paths")
    if not isinstance(value["diagnostic"], bool):
        raise ResultValidationError("invalid_result_field", "diagnostic must be boolean", "diagnostic")
    for field_name in (
        "dataset_is_fake",
        "model_is_fake",
        "mixed_precision_forward_applied",
        "accepted_as_qaq_result",
    ):
        if not isinstance(value[field_name], bool):
            raise ResultValidationError(
                "invalid_result_field",
                f"{field_name} must be boolean",
                field_name,
            )
    _require_non_empty_string(value, "artifact_scope")
    _require_non_empty_string(value, "artifact_ref_mode")
    _require_non_empty_string(value, "benchmark_name")
    _require_non_empty_string(value, "benchmark_split")
    if value["gpu_selector_record"] is not None and not isinstance(
        value["gpu_selector_record"],
        dict,
    ):
        raise ResultValidationError(
            "invalid_result_field",
            "gpu_selector_record must be an object or null",
            "gpu_selector_record",
        )
    if not isinstance(value["rejection_reasons"], list) or any(
        not isinstance(reason, str) or not reason
        for reason in value["rejection_reasons"]
    ):
        raise ResultValidationError(
            "invalid_result_field",
            "rejection_reasons must be a list of non-empty strings",
            "rejection_reasons",
        )
    _validate_acceptance_contract_fields(value)


def result_artifact_from_mapping(value: Mapping[str, Any]) -> ResultArtifact:
    """Load a result artifact from a JSON-compatible mapping."""

    validate_result_artifact(value)
    return ResultArtifact(
        schema_version=str(value["schema_version"]),
        result_id=str(value["result_id"]),
        evidence_level=str(value["evidence_level"]),
        model=str(value["model"]),
        tokenizer=str(value["tokenizer"]),
        dataset=str(value["dataset"]),
        split=str(value["split"]),
        prompt_format=str(value["prompt_format"]),
        mode=str(value["mode"]),
        precision_candidates=tuple(int(item) for item in value["precision_candidates"]),
        max_bit_width=int(value["max_bit_width"]),
        block_granularity=str(value["block_granularity"]),
        seed=int(value["seed"]),
        gpu_ids=tuple(int(item) for item in value["gpu_ids"]),
        hardware=dict(value["hardware"]),
        metric=str(value["metric"]),
        metrics=dict(value["metrics"]),
        score=_optional_float(value.get("score")),
        perplexity=_optional_float(value.get("perplexity")),
        latency_seconds=float(value["latency_seconds"]),
        peak_gpu_memory_gb=float(value["peak_gpu_memory_gb"]),
        routing_summary=_optional_dict(value.get("routing_summary")),
        loader_summary=_optional_dict(value.get("loader_summary")),
        log_paths={str(key): str(path) for key, path in dict(value["log_paths"]).items()},
        log_events=tuple(dict(event) for event in value.get("log_events", [])),
        completion_status=str(value["completion_status"]),
        diagnostic=bool(value["diagnostic"]),
        dataset_is_fake=bool(value["dataset_is_fake"]),
        model_is_fake=bool(value["model_is_fake"]),
        artifact_scope=str(value["artifact_scope"]),
        artifact_ref_mode=str(value["artifact_ref_mode"]),
        mixed_precision_forward_applied=bool(value["mixed_precision_forward_applied"]),
        benchmark_name=str(value["benchmark_name"]),
        benchmark_split=str(value["benchmark_split"]),
        gpu_selector_record=(
            dict(value["gpu_selector_record"])
            if value["gpu_selector_record"] is not None
            else None
        ),
        accepted_as_qaq_result=bool(value["accepted_as_qaq_result"]),
        rejection_reasons=tuple(str(reason) for reason in value["rejection_reasons"]),
        constrained=bool(value.get("constrained", False)),
        notes=tuple(str(note) for note in value.get("notes", [])),
        metadata=dict(value.get("metadata", {})),
    )


def save_result_artifact(artifact: ResultArtifact, path: str | Path) -> Path:
    """Persist a result artifact as deterministic JSON."""

    validate_result_artifact(artifact)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_result_artifact(path: str | Path) -> ResultArtifact:
    """Read and validate a JSON result artifact."""

    result_path = Path(path)
    try:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ResultValidationError("result_read_failed", str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise ResultValidationError("result_parse_failed", str(exc)) from exc
    if not isinstance(raw, dict):
        raise ResultValidationError("invalid_result_artifact", "result JSON must be an object")
    return result_artifact_from_mapping(raw)


def comparison_key(artifact: ResultArtifact) -> ComparisonKey:
    """Return the reproducibility key used for cross-mode grouping."""

    return ComparisonKey(
        model=artifact.model,
        tokenizer=artifact.tokenizer,
        dataset=artifact.dataset,
        split=artifact.split,
        benchmark_name=artifact.benchmark_name,
        benchmark_split=artifact.benchmark_split,
        prompt_format=artifact.prompt_format,
        metric=artifact.metric,
        precision_candidates=artifact.precision_candidates,
        seed=artifact.seed,
        block_granularity=artifact.block_granularity,
    )


def group_result_artifacts(
    artifacts: Iterable[ResultArtifact],
) -> dict[ComparisonKey, tuple[ResultArtifact, ...]]:
    """Group result artifacts by comparable settings."""

    groups: dict[ComparisonKey, list[ResultArtifact]] = {}
    for artifact in artifacts:
        groups.setdefault(comparison_key(artifact), []).append(artifact)
    return {key: tuple(values) for key, values in groups.items()}


def validate_comparison(
    artifacts: Sequence[ResultArtifact],
    *,
    required_modes: Sequence[str] = COMPARISON_REQUIRED_MODES,
) -> ComparisonValidation:
    """Validate whether a set of results can support a QAQ comparison claim."""

    if not artifacts:
        return ComparisonValidation(
            key=None,
            state="invalid",
            reasons=("no_results",),
            present_modes=(),
            missing_modes=tuple(required_modes),
        )

    reasons: list[str] = []
    for artifact in artifacts:
        try:
            validate_result_artifact(artifact)
        except ResultValidationError as exc:
            reasons.append(f"invalid_result:{artifact.mode}:{exc.code}")

    keys = {comparison_key(artifact) for artifact in artifacts}
    key = next(iter(keys)) if len(keys) == 1 else None
    if len(keys) > 1:
        reasons.append("settings_mismatch")

    by_mode: dict[str, ResultArtifact] = {}
    duplicates: set[str] = set()
    for artifact in artifacts:
        if artifact.mode in by_mode:
            duplicates.add(artifact.mode)
        by_mode[artifact.mode] = artifact
    if duplicates:
        reasons.extend(f"duplicate_mode:{mode}" for mode in sorted(duplicates))

    present_modes = tuple(sorted(by_mode, key=_mode_sort_key))
    missing_modes = tuple(mode for mode in required_modes if mode not in by_mode)
    if missing_modes:
        reasons.append(f"missing_required_modes:{','.join(missing_modes)}")

    incomplete = [
        artifact.mode
        for artifact in artifacts
        if artifact.completion_status != "completed"
    ]
    if incomplete:
        reasons.extend(f"incomplete_result:{mode}" for mode in sorted(incomplete))

    for mode in ("static_8bit", "static_4bit", "qaq_on_demand_off", "qaq_on_demand_on"):
        artifact = by_mode.get(mode)
        if artifact is None or artifact.diagnostic or artifact.constrained:
            continue
        if not _mixed_weight_forward_applied(artifact):
            reasons.append(f"missing_mixed_weight_forward:{mode}")

    for mode in ("qaq_on_demand_off", "qaq_on_demand_on"):
        artifact = by_mode.get(mode)
        if artifact is None:
            continue
        if not isinstance(artifact.routing_summary, dict):
            reasons.append(f"missing_routing_summary:{mode}")
            continue
        constant = artifact.routing_summary.get("constant_global_precision") is True
        flagged = artifact.routing_summary.get("constant_precision_flagged") is True
        diagnostic_router = artifact.routing_summary.get("diagnostic") is True
        if (constant or flagged) and not (artifact.diagnostic or diagnostic_router):
            reasons.append(f"constant_precision_not_adaptive:{mode}")

    on_demand = by_mode.get("qaq_on_demand_on")
    if on_demand is not None:
        loader_summary = on_demand.loader_summary
        if not isinstance(loader_summary, dict):
            reasons.append("missing_loader_summary:qaq_on_demand_on")
        else:
            activity = int(loader_summary.get("loads", 0)) + int(
                loader_summary.get("cache_hits", 0)
            )
            if activity <= 0:
                reasons.append("missing_loader_activity:qaq_on_demand_on")

    fake_model_flags = {artifact.model_is_fake for artifact in artifacts}
    fake_dataset_flags = {artifact.dataset_is_fake for artifact in artifacts}
    if len(fake_model_flags) > 1:
        reasons.append("mixed_fake_real_models")
    if len(fake_dataset_flags) > 1:
        reasons.append("mixed_fake_real_datasets")

    artifact_rejections = [
        f"artifact_not_accepted:{artifact.mode}:{','.join(artifact.rejection_reasons)}"
        for artifact in artifacts
        if not artifact.accepted_as_qaq_result
    ]
    if artifact_rejections:
        reasons.extend(artifact_rejections)

    structural_reasons = [
        reason
        for reason in reasons
        if not reason.startswith("artifact_not_accepted:")
        and reason != "diagnostic_or_constrained_results"
    ]
    all_diagnostic = all(
        artifact.diagnostic
        or artifact.constrained
        or artifact.evidence_level == "diagnostic_health_check"
        for artifact in artifacts
    )
    if structural_reasons:
        state = (
            "incomplete"
            if structural_reasons
            and all(
                reason.startswith("incomplete_result")
                for reason in structural_reasons
            )
            else "invalid"
        )
    elif artifact_rejections and all_diagnostic:
        state = "diagnostic"
        reasons.append("diagnostic_or_constrained_results")
    elif artifact_rejections:
        state = "invalid"
    elif any(artifact.diagnostic or artifact.constrained for artifact in artifacts):
        state = "diagnostic"
        reasons.append("diagnostic_or_constrained_results")
    else:
        state = "accepted"

    return ComparisonValidation(
        key=key,
        state=state,
        reasons=tuple(reasons),
        present_modes=present_modes,
        missing_modes=missing_modes,
    )


def validate_paper_reproduction_claim(
    artifacts: Sequence[ResultArtifact],
    *,
    deviation_labels: Sequence[str] = (),
) -> ComparisonValidation:
    """Validate that a full-paper claim has the required model and benchmark scope."""

    if deviation_labels:
        validation = validate_comparison(artifacts)
        return ComparisonValidation(
            key=validation.key,
            state="diagnostic",
            reasons=validation.reasons + tuple(
                f"paper_deviation:{label}" for label in deviation_labels
            ),
            present_modes=validation.present_modes,
            missing_modes=validation.missing_modes,
        )

    models = {artifact.model for artifact in artifacts}
    datasets = {artifact.dataset for artifact in artifacts}
    reasons: list[str] = []
    missing_models = sorted(PAPER_MODELS - models)
    missing_datasets = sorted(PAPER_DATASETS - datasets)
    if missing_models:
        reasons.append(f"missing_paper_models:{','.join(missing_models)}")
    if missing_datasets:
        reasons.append(f"missing_paper_datasets:{','.join(missing_datasets)}")
    if reasons:
        return ComparisonValidation(
            key=None,
            state="invalid",
            reasons=tuple(reasons),
            present_modes=tuple(sorted({artifact.mode for artifact in artifacts})),
            missing_modes=(),
        )
    return validate_comparison(artifacts)


def build_report_rows(
    artifacts: Sequence[ResultArtifact],
    *,
    validation: ComparisonValidation | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build stable paper-table-shaped rows from result artifacts."""

    validation = validation or validate_comparison(artifacts)
    rows: list[dict[str, Any]] = []
    for artifact in sorted(artifacts, key=lambda item: _mode_sort_key(item.mode)):
        routing = artifact.routing_summary or {}
        loader = artifact.loader_summary or {}
        rows.append(
            {
                "comparison_state": validation.state,
                "model": artifact.model,
                "dataset": artifact.dataset,
                "split": artifact.split,
                "mode": artifact.mode,
                "metric": artifact.metric,
                "score": artifact.score,
                "perplexity": artifact.perplexity,
                "latency_seconds": artifact.latency_seconds,
                "peak_gpu_memory_gb": artifact.peak_gpu_memory_gb,
                "routing_constant": routing.get("constant_global_precision"),
                "loader_loads": loader.get("loads"),
                "diagnostic": artifact.diagnostic,
                "completion_status": artifact.completion_status,
                "evidence_level": artifact.evidence_level,
                "accepted_as_qaq_result": artifact.accepted_as_qaq_result,
                "rejection_reasons": list(artifact.rejection_reasons),
                "artifact_ref_mode": artifact.artifact_ref_mode,
                "artifact_scope": artifact.artifact_scope,
                "mixed_precision_forward_applied": artifact.mixed_precision_forward_applied,
            }
        )
    return tuple(rows)


def _collect_log_paths(
    config: RunConfig,
    *,
    manifest: RunManifest | None,
    log_paths: Mapping[str, str | Path] | None,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    if manifest:
        for name, path in manifest.artifact_paths.items():
            if "log" in name:
                resolved[name] = path
    if config.logging.log_dir is not None:
        resolved.setdefault("log_dir", str(config.logging.log_dir))
    for name, path in dict(log_paths or {}).items():
        resolved[str(name)] = str(path)
    return resolved


def _is_diagnostic_result(config: RunConfig, output: RuntimeOutputBundle) -> bool:
    runtime_impl = str(output.metadata.get("runtime_impl", ""))
    routing_summary = output.metadata.get("routing_summary")
    return any(
        (
            config.mode == "fixed_mixed",
            config.router_diagnostic,
            bool(output.metadata.get("fixed_mixed_is_diagnostic")),
            isinstance(routing_summary, dict) and routing_summary.get("diagnostic") is True,
            config.model.startswith(("fake-", "fake_")),
            config.device == "cpu",
            "fake" in runtime_impl,
            "cpu" in runtime_impl,
        )
    )


def _mixed_weight_forward_applied(artifact: ResultArtifact) -> bool:
    return artifact.mixed_precision_forward_applied


def _artifact_scope(
    config: RunConfig,
    *,
    runtime_metadata: Mapping[str, Any],
    artifact_ref_mode: str,
) -> str:
    explicit = runtime_metadata.get("artifact_scope")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    if config.mode == "fp16":
        return "not_applicable_fp16"
    if artifact_ref_mode == "full_tensor_index":
        return "full_runtime_tensor_index"
    if artifact_ref_mode in REJECTED_ARTIFACT_REF_MODES:
        return "partial_or_legacy_runtime_index"
    return "missing_or_not_used"


def _gpu_selector_record(
    hardware: Mapping[str, Any],
    *,
    runtime_metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    for value in (
        runtime_metadata.get("gpu_selector_record"),
        runtime_metadata.get("gpu_run_status"),
        hardware.get("gpu_selector_record"),
        hardware.get("gpu_run_status"),
    ):
        if isinstance(value, dict):
            return dict(value)
    env_value = os.environ.get("QAQ_GPU_RUN_STATUS")
    if env_value:
        try:
            decoded = json.loads(env_value)
        except json.JSONDecodeError:
            return None
        if isinstance(decoded, dict):
            return decoded
    return None


def _contract_rejection_reasons(
    *,
    mode: str,
    completion_status: str,
    diagnostic: bool,
    dataset_is_fake: bool,
    model_is_fake: bool,
    artifact_scope: str,
    artifact_ref_mode: str,
    mixed_precision_forward_applied: bool,
    benchmark_name: str,
    benchmark_split: str,
    model: str,
    gpu_ids: Sequence[int],
    gpu_selector_record: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if completion_status != "completed":
        reasons.append("incomplete_result")
    if diagnostic:
        reasons.append("diagnostic_result")
    if dataset_is_fake:
        reasons.append("fake_dataset")
    if model_is_fake:
        reasons.append("fake_model")
    if _uses_smoke_fixture_or_synthetic_data(
        benchmark_name=benchmark_name,
        benchmark_split=benchmark_split,
        artifact_scope=artifact_scope,
        metadata=metadata,
    ):
        reasons.append("smoke_fixture_or_synthetic_data")
    if mode == "fixed_mixed":
        reasons.append("fixed_mixed_diagnostic_mode")
    if mode in MIXED_PRECISION_REQUIRED_MODES:
        if not mixed_precision_forward_applied:
            reasons.append("mixed_precision_forward_not_applied")
        if artifact_ref_mode in REJECTED_ARTIFACT_REF_MODES:
            reasons.append(f"unaccepted_artifact_ref_mode:{artifact_ref_mode}")
        elif artifact_ref_mode != "full_tensor_index":
            reasons.append(f"missing_full_tensor_artifact_index:{artifact_ref_mode}")
    if _is_large_model_experiment(model=model, gpu_ids=tuple(gpu_ids)) and not gpu_selector_record:
        reasons.append("missing_gpu_selector_record")
    return tuple(dict.fromkeys(reasons))


def _evidence_level_for_rejections(reasons: Sequence[str]) -> str:
    diagnostic_reasons = {
        "diagnostic_result",
        "fake_dataset",
        "fake_model",
        "smoke_fixture_or_synthetic_data",
        "fixed_mixed_diagnostic_mode",
    }
    if any(reason in diagnostic_reasons for reason in reasons):
        return "diagnostic_health_check"
    if reasons:
        return "real_path_implemented"
    return "accepted_experiment_result"


def _validate_acceptance_contract_fields(value: Mapping[str, Any]) -> None:
    metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
    computed_rejections = _contract_rejection_reasons(
        mode=str(value["mode"]),
        completion_status=str(value["completion_status"]),
        diagnostic=bool(value["diagnostic"]),
        dataset_is_fake=bool(value["dataset_is_fake"]),
        model_is_fake=bool(value["model_is_fake"]),
        artifact_scope=str(value["artifact_scope"]),
        artifact_ref_mode=str(value["artifact_ref_mode"]),
        mixed_precision_forward_applied=bool(value["mixed_precision_forward_applied"]),
        benchmark_name=str(value["benchmark_name"]),
        benchmark_split=str(value["benchmark_split"]),
        model=str(value["model"]),
        gpu_ids=tuple(int(item) for item in value["gpu_ids"]),
        gpu_selector_record=(
            dict(value["gpu_selector_record"])
            if isinstance(value["gpu_selector_record"], dict)
            else None
        ),
        metadata=metadata,
    )
    declared_rejections = tuple(str(reason) for reason in value["rejection_reasons"])
    missing_rejections = [
        reason for reason in computed_rejections if reason not in declared_rejections
    ]
    extra_rejections = [
        reason for reason in declared_rejections if reason not in computed_rejections
    ]
    if missing_rejections or extra_rejections:
        raise ResultValidationError(
            "invalid_acceptance_contract",
            f"rejection_reasons must match computed reasons; missing={missing_rejections}, extra={extra_rejections}",
            "rejection_reasons",
        )
    if bool(value["accepted_as_qaq_result"]) and computed_rejections:
        raise ResultValidationError(
            "invalid_acceptance_contract",
            "accepted_as_qaq_result must be false when rejection reasons exist",
            "accepted_as_qaq_result",
        )
    expected_level = _evidence_level_for_rejections(computed_rejections)
    if value["evidence_level"] != expected_level:
        raise ResultValidationError(
            "invalid_acceptance_contract",
            f"evidence_level must be {expected_level!r} for this artifact",
            "evidence_level",
        )
    if bool(value["accepted_as_qaq_result"]) != (not computed_rejections):
        raise ResultValidationError(
            "invalid_acceptance_contract",
            "accepted_as_qaq_result must match the computed acceptance contract",
            "accepted_as_qaq_result",
        )


def _is_fake_dataset(dataset: str) -> bool:
    lowered = dataset.lower()
    path = Path(dataset)
    return any(
        token in lowered
        for token in ("fake", "smoke", "fixture", "synthetic", "toy", "tiny")
    ) or "tests/fixtures" in str(path)


def _is_fake_model(model: str) -> bool:
    lowered = model.lower()
    path = Path(model)
    return any(
        token in lowered
        for token in ("fake", "smoke", "fixture", "synthetic", "toy", "tiny")
    ) or "tests/fixtures" in str(path)


def _uses_smoke_fixture_or_synthetic_data(
    *,
    benchmark_name: str,
    benchmark_split: str,
    artifact_scope: str,
    metadata: Mapping[str, Any],
) -> bool:
    haystack = " ".join(
        (
            benchmark_name,
            benchmark_split,
            artifact_scope,
            json.dumps(metadata, sort_keys=True, default=str),
        )
    ).lower()
    return any(
        token in haystack
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
    )


def _is_large_model_experiment(*, model: str, gpu_ids: tuple[int, ...]) -> bool:
    lowered = model.lower()
    if _is_fake_model(model):
        return False
    return bool(gpu_ids) or any(
        token in lowered
        for token in (
            "llama",
            "qwen",
            "mistral",
            "mixtral",
            "falcon",
            "8b",
            "7b",
            "4b",
        )
    )


def _optional_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ResultValidationError(
            "invalid_result_field",
            "optional summary fields must be objects when present",
        )
    return dict(value)


def _default_result_id(config: RunConfig) -> str:
    safe_model = config.model.replace("/", "_")
    safe_dataset = config.dataset.replace("/", "_")
    return f"{safe_model}:{safe_dataset}:{config.split}:{config.mode}:seed{config.seed}"


def _require_non_empty_string(value: Mapping[str, Any], field_name: str) -> None:
    if not isinstance(value[field_name], str) or not value[field_name]:
        raise ResultValidationError(
            "invalid_result_field",
            "field must be a non-empty string",
            field_name,
        )


def _require_int_sequence(value: Mapping[str, Any], field_name: str) -> None:
    sequence = value[field_name]
    if (
        not isinstance(sequence, list)
        or not sequence
        or any(isinstance(item, bool) or not isinstance(item, int) for item in sequence)
    ):
        raise ResultValidationError(
            "invalid_result_field",
            "field must be a non-empty integer list",
            field_name,
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _mode_sort_key(mode: str) -> tuple[int, str]:
    try:
        return (COMPARISON_REQUIRED_MODES.index(mode), mode)
    except ValueError:
        return (len(COMPARISON_REQUIRED_MODES), mode)
