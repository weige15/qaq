"""Run manifest creation and lifecycle status updates."""

from __future__ import annotations

import json
import platform
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qaq.config import RunConfig
from qaq.errors import ManifestError


STATUS_STARTED = "started"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


@dataclass(frozen=True, slots=True)
class HardwareMetadata:
    """Hardware and environment fields captured before expensive work begins."""

    hostname: str
    platform: str
    python_version: str
    selected_gpu_ids: tuple[int, ...]
    detected_gpu_count: int | None = None

    @classmethod
    def collect(
        cls,
        *,
        selected_gpu_ids: tuple[int, ...],
        detected_gpu_count: int | None = None,
    ) -> "HardwareMetadata":
        return cls(
            hostname=socket.gethostname(),
            platform=platform.platform(),
            python_version=sys.version.split()[0],
            selected_gpu_ids=selected_gpu_ids,
            detected_gpu_count=detected_gpu_count,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "platform": self.platform,
            "python_version": self.python_version,
            "selected_gpu_ids": list(self.selected_gpu_ids),
            "detected_gpu_count": self.detected_gpu_count,
        }


@dataclass(slots=True)
class RunManifest:
    """Machine-readable run manifest stored under a run output directory."""

    run_id: str
    config: RunConfig
    hardware: HardwareMetadata
    manifest_path: Path
    artifact_paths: dict[str, str] = field(default_factory=dict)
    status: str = STATUS_STARTED
    started_at: str = field(default_factory=lambda: _now_iso())
    completed_at: str | None = None
    failure: dict[str, str] | None = None
    incomplete_marker: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config.as_dict(),
            "hardware": self.hardware.as_dict(),
            "artifact_paths": dict(self.artifact_paths),
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "failure": self.failure,
            "incomplete_marker": self.incomplete_marker,
        }

    def write(self) -> None:
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with self.manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(self.as_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
        except OSError as exc:
            raise ManifestError("manifest_write_failed", str(exc)) from exc

    def mark_completed(self, *, completed_at: str | None = None) -> None:
        self.status = STATUS_COMPLETED
        self.completed_at = completed_at or _now_iso()
        self.failure = None
        marker = Path(self.incomplete_marker) if self.incomplete_marker else None
        self.incomplete_marker = None
        self.write()
        if marker and marker.exists():
            try:
                marker.unlink()
            except OSError as exc:
                raise ManifestError("incomplete_marker_remove_failed", str(exc)) from exc

    def mark_failed(
        self,
        *,
        code: str,
        message: str,
        completed_at: str | None = None,
    ) -> None:
        self.status = STATUS_FAILED
        self.completed_at = completed_at or _now_iso()
        self.failure = {"code": code, "message": message}
        marker = self.manifest_path.parent / "INCOMPLETE"
        self.incomplete_marker = str(marker)
        try:
            marker.write_text(f"{code}: {message}\n", encoding="utf-8")
        except OSError as exc:
            raise ManifestError("incomplete_marker_write_failed", str(exc)) from exc
        self.write()


def create_run_manifest(
    config: RunConfig,
    *,
    run_id: str | None = None,
    hardware: HardwareMetadata | None = None,
    artifact_paths: dict[str, str] | None = None,
    started_at: str | None = None,
) -> RunManifest:
    """Create and persist a started manifest for a validated config."""

    output_dir = config.output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ManifestError("output_dir_create_failed", str(exc)) from exc

    manifest = RunManifest(
        run_id=run_id or _default_run_id(config),
        config=config,
        hardware=hardware
        or HardwareMetadata.collect(selected_gpu_ids=config.gpu_ids),
        manifest_path=output_dir / "manifest.json",
        artifact_paths=dict(artifact_paths or {}),
        started_at=started_at or _now_iso(),
    )
    manifest.write()
    return manifest


def _default_run_id(config: RunConfig) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{config.mode}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
