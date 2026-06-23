"""Metric aggregation helpers for QAQ result artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

from qaq.model_adapter import ReferenceBatchOutput
from qaq.runtime.common import LatencyEvent, MemoryEvent


@dataclass(slots=True)
class MetricAggregationError(ValueError):
    """Raised when runtime outputs cannot produce a configured metric."""

    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class QualityMetric:
    """Primary quality metric computed from one runtime output bundle."""

    metric_name: str
    primary_value: float
    higher_is_better: bool
    num_examples: int
    score: float | None = None
    perplexity: float | None = None
    average_loss: float | None = None
    source: str = "runtime_raw_output"

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "primary_value": self.primary_value,
            "higher_is_better": self.higher_is_better,
            "num_examples": self.num_examples,
            "score": self.score,
            "perplexity": self.perplexity,
            "average_loss": self.average_loss,
            "source": self.source,
        }


def compute_quality_metric(
    raw_output: ReferenceBatchOutput,
    *,
    metric_name: str | None,
) -> QualityMetric:
    """Compute the configured quality metric from fake/tiny runtime outputs."""

    metric = (metric_name or "exact_match").lower()
    losses = tuple(loss for loss in raw_output.losses if loss is not None)
    num_examples = len(raw_output.predictions)
    if num_examples <= 0:
        raise MetricAggregationError(
            "empty_metric_input",
            "runtime output does not contain any predictions",
        )

    if metric in {"exact_match", "accuracy", "benchmark_score"}:
        if not losses:
            raise MetricAggregationError(
                "missing_targets",
                f"{metric} requires target-derived losses in the runtime output",
            )
        score = sum(1 for loss in losses if loss == 0.0) / len(losses)
        return QualityMetric(
            metric_name=metric,
            primary_value=score,
            higher_is_better=True,
            num_examples=len(losses),
            score=score,
        )

    if metric in {"perplexity", "ppl"}:
        average_loss = _average_loss(losses, metric=metric)
        perplexity = exp(average_loss)
        return QualityMetric(
            metric_name="perplexity",
            primary_value=perplexity,
            higher_is_better=False,
            num_examples=len(losses),
            perplexity=perplexity,
            average_loss=average_loss,
        )

    if metric in {"loss", "mean_loss"}:
        average_loss = _average_loss(losses, metric=metric)
        return QualityMetric(
            metric_name="loss",
            primary_value=average_loss,
            higher_is_better=False,
            num_examples=len(losses),
            average_loss=average_loss,
        )

    if losses:
        score = sum(1 for loss in losses if loss == 0.0) / len(losses)
        return QualityMetric(
            metric_name=metric,
            primary_value=score,
            higher_is_better=True,
            num_examples=len(losses),
            score=score,
            source="target_loss_zero_score",
        )

    raise MetricAggregationError(
        "unsupported_metric",
        f"unsupported metric {metric_name!r} has no registered aggregator and no target-derived losses",
    )


def summarize_latency(events: tuple[LatencyEvent, ...]) -> dict[str, Any]:
    """Aggregate latency events while preserving the raw measured records."""

    event_dicts = [event.as_dict() for event in events]
    end_to_end = [
        float(event["elapsed_seconds"])
        for event in event_dicts
        if event.get("name") == "end_to_end"
    ]
    if not end_to_end:
        raise MetricAggregationError(
            "missing_latency",
            "result artifacts require an end_to_end latency event",
        )
    latency_seconds = max(end_to_end)
    if latency_seconds < 0:
        raise MetricAggregationError(
            "invalid_latency",
            "latency values must be non-negative",
        )
    return {
        "end_to_end_seconds": latency_seconds,
        "event_count": len(event_dicts),
        "events": event_dicts,
    }


def summarize_memory(events: tuple[MemoryEvent, ...]) -> dict[str, Any]:
    """Aggregate peak GPU memory events while preserving measurement details."""

    event_dicts = [event.as_dict() for event in events]
    if not event_dicts:
        raise MetricAggregationError(
            "missing_memory",
            "result artifacts require at least one memory event",
        )
    peaks = [float(event["peak_gpu_memory_gb"]) for event in event_dicts]
    if any(value < 0 for value in peaks):
        raise MetricAggregationError(
            "invalid_memory",
            "peak GPU memory values must be non-negative",
        )
    return {
        "peak_gpu_memory_gb": max(peaks),
        "event_count": len(event_dicts),
        "events": event_dicts,
    }


def _average_loss(losses: tuple[float, ...], *, metric: str) -> float:
    if not losses:
        raise MetricAggregationError(
            "missing_losses",
            f"{metric} requires loss values in the runtime output",
        )
    return sum(losses) / len(losses)
