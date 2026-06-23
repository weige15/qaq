"""Router checkpoint serialization and compatibility validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from dataclasses import dataclass

from qaq.blocks import BlockDescriptor
from qaq.router.types import (
    DEFAULT_DECISION_POLICY,
    ROUTER_CHECKPOINT_VERSION,
    RouterBlockParameters,
    RouterCheckpointMetadata,
    RouterPolicyError,
)


@dataclass(frozen=True, slots=True)
class RouterCheckpoint:
    """A lightweight router checkpoint with per-block linear scores."""

    metadata: RouterCheckpointMetadata
    parameters: dict[str, RouterBlockParameters]

    def as_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.as_dict(),
            "parameters": {
                block_id: params.as_dict()
                for block_id, params in sorted(self.parameters.items())
            },
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RouterCheckpoint":
        metadata_raw = value.get("metadata")
        if not isinstance(metadata_raw, dict):
            raise RouterPolicyError(
                "invalid_router_checkpoint",
                "metadata must be an object",
            )
        parameters_raw = value.get("parameters")
        if not isinstance(parameters_raw, dict):
            raise RouterPolicyError(
                "invalid_router_checkpoint",
                "parameters must be an object",
            )
        metadata = RouterCheckpointMetadata.from_mapping(metadata_raw)
        parameters = {
            block_id: RouterBlockParameters.from_mapping(params)
            for block_id, params in parameters_raw.items()
            if isinstance(block_id, str)
        }
        checkpoint = cls(metadata=metadata, parameters=parameters)
        checkpoint.validate()
        return checkpoint

    def validate(self) -> None:
        validate_router_metadata(self.metadata)
        expected_blocks = set(self.metadata.block_ids)
        parameter_blocks = set(self.parameters)
        if parameter_blocks != expected_blocks:
            missing = sorted(expected_blocks - parameter_blocks)
            extra = sorted(parameter_blocks - expected_blocks)
            raise RouterPolicyError(
                "router_parameter_mismatch",
                f"parameter block IDs do not match metadata; missing={missing}, extra={extra}",
            )
        for block_id, params in self.parameters.items():
            if len(params.weights) != len(self.metadata.candidate_bit_widths):
                raise RouterPolicyError(
                    "router_parameter_mismatch",
                    f"{block_id} must have one weight row per candidate bit-width",
                )
            if len(params.bias) != len(self.metadata.candidate_bit_widths):
                raise RouterPolicyError(
                    "router_parameter_mismatch",
                    f"{block_id} must have one bias per candidate bit-width",
                )
            for row in params.weights:
                if len(row) != self.metadata.hidden_size:
                    raise RouterPolicyError(
                        "router_parameter_mismatch",
                        f"{block_id} weight row width must match hidden_size",
                    )
                if any(not math.isfinite(value) for value in row):
                    raise RouterPolicyError(
                        "non_finite_router_parameter",
                        f"{block_id} contains non-finite weights",
                    )
            if any(not math.isfinite(value) for value in params.bias):
                raise RouterPolicyError(
                    "non_finite_router_parameter",
                    f"{block_id} contains non-finite bias values",
                )


def validate_router_metadata(metadata: RouterCheckpointMetadata) -> None:
    if metadata.version != ROUTER_CHECKPOINT_VERSION:
        raise RouterPolicyError(
            "unsupported_router_checkpoint",
            f"unsupported router checkpoint version {metadata.version}",
        )
    if metadata.hidden_size <= 0:
        raise RouterPolicyError(
            "invalid_router_metadata",
            "hidden_size must be positive",
        )
    if metadata.temperature <= 0 or not math.isfinite(metadata.temperature):
        raise RouterPolicyError(
            "invalid_temperature",
            "temperature must be finite and positive",
        )
    if metadata.decision_policy not in {
        DEFAULT_DECISION_POLICY,
        "argmax_highest_bit_width",
    }:
        raise RouterPolicyError(
            "unsupported_decision_policy",
            f"unsupported decision policy {metadata.decision_policy}",
        )
    if not metadata.candidate_bit_widths:
        raise RouterPolicyError(
            "invalid_router_metadata",
            "candidate_bit_widths are required",
        )
    if len(set(metadata.candidate_bit_widths)) != len(metadata.candidate_bit_widths):
        raise RouterPolicyError(
            "invalid_router_metadata",
            "candidate_bit_widths must be unique",
        )
    if any(bit_width <= 0 for bit_width in metadata.candidate_bit_widths):
        raise RouterPolicyError(
            "invalid_router_metadata",
            "candidate_bit_widths must be positive",
        )
    max_bit_width = metadata.max_bit_width or max(metadata.candidate_bit_widths)
    if any(bit_width > max_bit_width for bit_width in metadata.candidate_bit_widths):
        raise RouterPolicyError(
            "invalid_router_metadata",
            "candidate_bit_widths must be <= max_bit_width",
        )
    if len(set(metadata.block_ids)) != len(metadata.block_ids):
        raise RouterPolicyError(
            "invalid_router_metadata",
            "block_ids must be unique",
        )


def validate_checkpoint_compatibility(
    checkpoint: RouterCheckpoint,
    *,
    blocks: tuple[BlockDescriptor, ...],
    model_id: str,
    candidate_bit_widths: tuple[int, ...],
    feature_source: str,
) -> None:
    """Validate checkpoint metadata against the active model/runtime context."""

    checkpoint.validate()
    metadata = checkpoint.metadata
    active_block_ids = tuple(block.block_id for block in blocks)
    if metadata.model_id != model_id:
        raise RouterPolicyError(
            "router_model_mismatch",
            f"checkpoint model {metadata.model_id} does not match {model_id}",
        )
    if metadata.block_ids != active_block_ids:
        raise RouterPolicyError(
            "router_block_mismatch",
            "checkpoint block IDs do not match active block IDs",
        )
    if metadata.candidate_bit_widths != tuple(sorted(candidate_bit_widths)):
        raise RouterPolicyError(
            "router_candidate_mismatch",
            "checkpoint candidate bit-widths do not match active precision candidates",
        )
    if metadata.feature_source != feature_source:
        raise RouterPolicyError(
            "router_feature_source_mismatch",
            f"checkpoint feature source {metadata.feature_source} does not match {feature_source}",
        )
    for block in blocks:
        unsupported = [
            bit_width
            for bit_width in metadata.candidate_bit_widths
            if bit_width not in block.supported_bit_widths
        ]
        if unsupported:
            raise RouterPolicyError(
                "router_candidate_mismatch",
                f"{block.block_id} does not support candidate bit-widths {unsupported}",
            )


def save_router_checkpoint(checkpoint: RouterCheckpoint, path: str | Path) -> Path:
    checkpoint.validate()
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w", encoding="utf-8") as handle:
        json.dump(checkpoint.as_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return checkpoint_path


def load_router_checkpoint(path: str | Path) -> RouterCheckpoint:
    checkpoint_path = Path(path)
    try:
        raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RouterPolicyError("router_checkpoint_read_failed", str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise RouterPolicyError("router_checkpoint_parse_failed", str(exc)) from exc
    if not isinstance(raw, dict):
        raise RouterPolicyError(
            "invalid_router_checkpoint",
            "checkpoint must be a JSON object",
        )
    return RouterCheckpoint.from_mapping(raw)
