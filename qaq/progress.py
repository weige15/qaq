"""Console progress state for long-running QAQ work."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, TextIO

from qaq.logging import LogEvent
from qaq.status import EventType, RunStatus


@dataclass(slots=True)
class ProgressState:
    """Latest known progress fields for console display and tests."""

    run_id: str
    status: str = RunStatus.STARTED.value
    mode: str | None = None
    benchmark: str | None = None
    step: int | None = None
    epoch: int | None = None
    processed_examples: int | None = None
    total_examples: int | None = None
    loss: float | None = None
    learning_rate: float | None = None
    elapsed_seconds: float | None = None
    latency_seconds: float | None = None
    peak_gpu_memory_gb: float | None = None
    routing_summary: dict[str, Any] | None = None
    loader_summary: dict[str, Any] | None = None
    last_checkpoint_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    failure: dict[str, str] | None = None

    def apply(self, event: LogEvent) -> None:
        if event.status is not None:
            self.status = event.status
        if event.mode is not None:
            self.mode = event.mode
        if event.benchmark is not None:
            self.benchmark = event.benchmark
        if event.step is not None:
            self.step = event.step
        if event.epoch is not None:
            self.epoch = event.epoch
        if event.processed_examples is not None:
            self.processed_examples = event.processed_examples
        if event.total_examples is not None:
            self.total_examples = event.total_examples
        if event.loss is not None:
            self.loss = event.loss
        if event.learning_rate is not None:
            self.learning_rate = event.learning_rate
        if event.elapsed_seconds is not None:
            self.elapsed_seconds = event.elapsed_seconds
        if event.latency_seconds is not None:
            self.latency_seconds = event.latency_seconds
        if event.peak_gpu_memory_gb is not None:
            self.peak_gpu_memory_gb = event.peak_gpu_memory_gb
        if event.routing_summary is not None:
            self.routing_summary = event.routing_summary
        if event.loader_summary is not None:
            self.loader_summary = event.loader_summary
        if event.checkpoint_path is not None:
            self.last_checkpoint_path = event.checkpoint_path
        if event.event_type == EventType.WARNING.value and event.message:
            self.warnings.append(event.message)
        if event.event_type == EventType.ERROR.value:
            self.status = RunStatus.FAILED.value
            self.failure = {
                "code": event.error_code or "error",
                "message": event.message or "",
            }

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "mode": self.mode,
            "benchmark": self.benchmark,
            "step": self.step,
            "epoch": self.epoch,
            "processed_examples": self.processed_examples,
            "total_examples": self.total_examples,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "elapsed_seconds": self.elapsed_seconds,
            "latency_seconds": self.latency_seconds,
            "peak_gpu_memory_gb": self.peak_gpu_memory_gb,
            "routing_summary": self.routing_summary,
            "loader_summary": self.loader_summary,
            "last_checkpoint_path": self.last_checkpoint_path,
            "warnings": list(self.warnings),
            "failure": self.failure,
        }


class ConsoleProgressMonitor:
    """Small console monitor that mirrors durable progress state."""

    def __init__(
        self,
        *,
        run_id: str,
        stream: TextIO | None = None,
        enabled: bool = True,
    ) -> None:
        self.state = ProgressState(run_id=run_id)
        self.stream = stream
        self.enabled = enabled

    def handle(self, event: LogEvent) -> ProgressState:
        self.state.apply(event)
        if self.enabled and self.stream is not None:
            self.stream.write(self.render() + "\n")
            self.stream.flush()
        return self.state

    def render(self) -> str:
        parts = [f"run={self.state.run_id}", f"status={self.state.status}"]
        if self.state.mode:
            parts.append(f"mode={self.state.mode}")
        if self.state.benchmark:
            parts.append(f"benchmark={self.state.benchmark}")
        if self.state.epoch is not None:
            parts.append(f"epoch={self.state.epoch}")
        if self.state.step is not None:
            parts.append(f"step={self.state.step}")
        if self.state.processed_examples is not None:
            if self.state.total_examples is not None:
                parts.append(
                    f"examples={self.state.processed_examples}/"
                    f"{self.state.total_examples}"
                )
            else:
                parts.append(f"examples={self.state.processed_examples}")
        if self.state.loss is not None:
            parts.append(f"loss={self.state.loss:.6g}")
        if self.state.learning_rate is not None:
            parts.append(f"lr={self.state.learning_rate:.6g}")
        if self.state.latency_seconds is not None:
            parts.append(f"latency_s={self.state.latency_seconds:.6g}")
        if self.state.peak_gpu_memory_gb is not None:
            parts.append(f"peak_gpu_gb={self.state.peak_gpu_memory_gb:.6g}")
        if self.state.last_checkpoint_path:
            parts.append(f"checkpoint={self.state.last_checkpoint_path}")
        if self.state.failure:
            parts.append(f"failure={self.state.failure['code']}")
        return " ".join(parts)


@dataclass(slots=True)
class TimingMeasurement:
    """A measured interval separated from progress display overhead."""

    name: str
    clock: Callable[[], float] = time.perf_counter
    start_seconds: float | None = None
    end_seconds: float | None = None
    elapsed_seconds: float | None = None

    def __enter__(self) -> "TimingMeasurement":
        self.start_seconds = self.clock()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.end_seconds = self.clock()
        self.elapsed_seconds = self.end_seconds - (self.start_seconds or 0.0)
