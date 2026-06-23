"""Structured JSONL logging for QAQ runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from qaq.manifest import RunManifest
from qaq.status import EventType, RunStatus


class LogWriteError(RuntimeError):
    """Raised when a durable log cannot be written."""


@dataclass(frozen=True, slots=True)
class LogEvent:
    """A structured event emitted by training, inference, evaluation, or loading."""

    event_type: str
    run_id: str
    module: str
    timestamp: str = field(default_factory=lambda: _now_iso())
    level: str = "info"
    message: str | None = None
    status: str | None = None
    step: int | None = None
    epoch: int | None = None
    mode: str | None = None
    benchmark: str | None = None
    processed_examples: int | None = None
    total_examples: int | None = None
    loss: float | None = None
    learning_rate: float | None = None
    elapsed_seconds: float | None = None
    latency_seconds: float | None = None
    peak_gpu_memory_gb: float | None = None
    checkpoint_path: str | None = None
    selected_gpu_ids: tuple[int, ...] | None = None
    routing_summary: dict[str, Any] | None = None
    loader_summary: dict[str, Any] | None = None
    error_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def progress(
        cls,
        *,
        run_id: str,
        module: str,
        message: str | None = None,
        **fields: Any,
    ) -> "LogEvent":
        return cls(
            event_type=EventType.PROGRESS.value,
            run_id=run_id,
            module=module,
            message=message,
            status=RunStatus.RUNNING.value,
            **fields,
        )

    @classmethod
    def warning(
        cls,
        *,
        run_id: str,
        module: str,
        message: str,
        **fields: Any,
    ) -> "LogEvent":
        return cls(
            event_type=EventType.WARNING.value,
            run_id=run_id,
            module=module,
            level="warning",
            message=message,
            **fields,
        )

    @classmethod
    def error(
        cls,
        *,
        run_id: str,
        module: str,
        code: str,
        message: str,
        **fields: Any,
    ) -> "LogEvent":
        return cls(
            event_type=EventType.ERROR.value,
            run_id=run_id,
            module=module,
            level="error",
            message=message,
            status=RunStatus.FAILED.value,
            error_code=code,
            **fields,
        )

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "event_type": self.event_type,
            "run_id": self.run_id,
            "module": self.module,
            "timestamp": self.timestamp,
            "level": self.level,
            "details": dict(self.details),
        }
        optional_fields = (
            "message",
            "status",
            "step",
            "epoch",
            "mode",
            "benchmark",
            "processed_examples",
            "total_examples",
            "loss",
            "learning_rate",
            "elapsed_seconds",
            "latency_seconds",
            "peak_gpu_memory_gb",
            "checkpoint_path",
            "selected_gpu_ids",
            "routing_summary",
            "loader_summary",
            "error_code",
        )
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is None:
                continue
            if field_name == "selected_gpu_ids":
                data[field_name] = list(value)
            else:
                data[field_name] = value
        return data


class JsonlLogWriter:
    """JSONL writer for durable run events."""

    def __init__(self, path: str | Path, *, append: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        try:
            self._handle: TextIO = self.path.open(mode, encoding="utf-8")
        except OSError as exc:
            raise LogWriteError(str(exc)) from exc
        self.events_written = 0

    def record(self, event: LogEvent) -> None:
        try:
            self._handle.write(json.dumps(event.as_dict(), sort_keys=True) + "\n")
            self._handle.flush()
        except OSError as exc:
            raise LogWriteError(str(exc)) from exc
        self.events_written += 1

    def flush(self) -> None:
        try:
            self._handle.flush()
        except OSError as exc:
            raise LogWriteError(str(exc)) from exc

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlLogWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def open_run_log(manifest: RunManifest, *, name: str = "events") -> JsonlLogWriter:
    """Create a JSONL writer and record its path in the manifest artifact map."""

    log_dir = manifest.config.logging.log_dir or manifest.config.output_dir / "logs"
    log_path = log_dir / f"{name}.jsonl"
    artifact_key = f"{name}_log"
    first_open_for_run = artifact_key not in manifest.artifact_paths
    manifest.artifact_paths[artifact_key] = str(log_path)
    manifest.write()
    append = not (manifest.config.overwrite and first_open_for_run)
    return JsonlLogWriter(log_path, append=append)


def record_failure(
    manifest: RunManifest,
    writer: JsonlLogWriter,
    *,
    module: str,
    code: str,
    message: str,
) -> None:
    """Durably record a failure event and mark the run manifest incomplete."""

    writer.record(
        LogEvent.error(
            run_id=manifest.run_id,
            module=module,
            code=code,
            message=message,
        )
    )
    writer.flush()
    manifest.mark_failed(code=code, message=message)


def record_completion(
    manifest: RunManifest,
    writer: JsonlLogWriter,
    *,
    module: str,
    message: str = "run completed",
) -> None:
    """Durably record completion and mark the run manifest completed."""

    writer.record(
        LogEvent(
            event_type=EventType.COMPLETION.value,
            run_id=manifest.run_id,
            module=module,
            status=RunStatus.COMPLETED.value,
            message=message,
        )
    )
    writer.flush()
    manifest.mark_completed()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
