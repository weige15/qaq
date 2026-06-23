"""Shared run status and event type names."""

from __future__ import annotations

from enum import StrEnum


class RunStatus(StrEnum):
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class EventType(StrEnum):
    RUN_START = "run_start"
    PROGRESS = "progress"
    METRIC = "metric"
    CHECKPOINT = "checkpoint"
    LOADER = "loader"
    ROUTING = "routing"
    WARNING = "warning"
    ERROR = "error"
    COMPLETION = "completion"
    INCOMPLETE = "incomplete"
