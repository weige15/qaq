"""Shared runtime result and measurement records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qaq.model_adapter import ReferenceBatchOutput
from qaq.precision_plan import PrecisionPlan


@dataclass(slots=True)
class RuntimeError(ValueError):
    """Raised when runtime execution cannot produce a valid run output."""

    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class LatencyEvent:
    """Measured wall-clock event for one runtime pass."""

    name: str
    elapsed_seconds: float
    measurement_source: str = "perf_counter"
    warmup_steps: int = 0
    cache_policy: str = "not_configured"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "elapsed_seconds": self.elapsed_seconds,
            "measurement_source": self.measurement_source,
            "warmup_steps": self.warmup_steps,
            "cache_policy": self.cache_policy,
        }


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    """Memory measurement event for CPU-only fake and future GPU runtimes."""

    name: str
    peak_gpu_memory_gb: float
    selected_gpu_ids: tuple[int, ...]
    measurement_source: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "peak_gpu_memory_gb": self.peak_gpu_memory_gb,
            "selected_gpu_ids": list(self.selected_gpu_ids),
            "measurement_source": self.measurement_source,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class RuntimeOutputBundle:
    """Result-ready output from one non-adaptive runtime pass."""

    mode: str
    status: str
    raw_output: ReferenceBatchOutput
    precision_plan: PrecisionPlan
    latency_events: tuple[LatencyEvent, ...]
    memory_events: tuple[MemoryEvent, ...]
    reconstruction_records: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]
    log_events: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "status": self.status,
            "raw_output": self.raw_output.as_dict(),
            "precision_plan": self.precision_plan.as_dict(),
            "latency_events": [event.as_dict() for event in self.latency_events],
            "memory_events": [event.as_dict() for event in self.memory_events],
            "reconstruction_records": list(self.reconstruction_records),
            "metadata": dict(self.metadata),
            "log_events": list(self.log_events),
        }
