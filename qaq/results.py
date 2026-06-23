"""Result artifact schema and QAQ comparison validation."""

from __future__ import annotations

import json
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
COMPARISON_REQUIRED_MODES = (
    "fp16",
    "static_8bit",
    "static_4bit",
    "qaq_on_demand_off",
    "qaq_on_demand_on",
)
COMPARISON_STATES = frozenset({"accepted", "diagnostic", "incomplete", "invalid"})
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
    constrained: bool = False
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
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
    metadata = {
        "runtime_metadata": dict(output.metadata),
        "raw_output_metadata": dict(output.raw_output.metadata),
        "runtime_status": output.status,
    }
    artifact = ResultArtifact(
        schema_version=RESULT_SCHEMA_VERSION,
        result_id=result_id or _default_result_id(config),
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
        diagnostic=_is_diagnostic_result(config, output),
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


def result_artifact_from_mapping(value: Mapping[str, Any]) -> ResultArtifact:
    """Load a result artifact from a JSON-compatible mapping."""

    validate_result_artifact(value)
    return ResultArtifact(
        schema_version=str(value["schema_version"]),
        result_id=str(value["result_id"]),
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

    if reasons:
        state = "incomplete" if reasons and all(reason.startswith("incomplete_result") for reason in reasons) else "invalid"
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
