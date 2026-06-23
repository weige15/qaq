"""Artifact serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path

from qaq.bitplanes import BitPlaneArtifact, BitPlaneError


def save_bitplane_artifact(artifact: BitPlaneArtifact, path: str | Path) -> Path:
    """Save a bit-plane artifact as deterministic JSON."""

    artifact.validate()
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact.as_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return artifact_path


def load_bitplane_artifact(path: str | Path) -> BitPlaneArtifact:
    """Load and validate a bit-plane artifact JSON file."""

    artifact_path = Path(path)
    try:
        with artifact_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except OSError as exc:
        raise BitPlaneError("artifact_read_failed", str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise BitPlaneError("artifact_parse_failed", str(exc)) from exc
    if not isinstance(raw, dict):
        raise BitPlaneError("invalid_artifact", "artifact JSON must be an object")
    return BitPlaneArtifact.from_mapping(raw)
