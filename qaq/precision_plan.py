"""Precision-plan construction and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from qaq.blocks import BlockDescriptor, block_map
from qaq.config import QAQ_MODES, VALID_MODES


@dataclass(slots=True)
class PrecisionPlanError(ValueError):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class PrecisionPlan:
    """Per-query or per-run precision decisions keyed by block ID."""

    mode: str
    precision_candidates: tuple[int, ...]
    decisions: dict[str, int]
    decision_source: str
    query_id: str | None = None
    router_checkpoint: str | None = None
    temperature: float | None = None
    tie_break_policy: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "precision_candidates": list(self.precision_candidates),
            "decisions": dict(self.decisions),
            "decision_source": self.decision_source,
            "query_id": self.query_id,
            "router_checkpoint": self.router_checkpoint,
            "temperature": self.temperature,
            "tie_break_policy": self.tie_break_policy,
        }


def build_precision_plan(
    blocks: tuple[BlockDescriptor, ...],
    *,
    mode: str,
    precision_candidates: tuple[int, ...],
    max_bit_width: int,
    fixed_precision_by_block: Mapping[str, int] | None = None,
    router_decisions: Mapping[str, int] | None = None,
    query_id: str | None = None,
    router_checkpoint: str | None = None,
    temperature: float | None = None,
    tie_break_policy: str | None = None,
    require_artifacts: bool = False,
) -> PrecisionPlan:
    """Construct a mode-specific precision plan and validate all decisions."""

    if mode not in VALID_MODES:
        raise PrecisionPlanError("invalid_mode", f"unsupported mode {mode}")
    if max_bit_width <= 0:
        raise PrecisionPlanError(
            "invalid_max_bit_width",
            "max_bit_width must be positive",
        )

    candidate_set = set(precision_candidates)
    if not candidate_set:
        raise PrecisionPlanError(
            "invalid_precision_candidates",
            "at least one precision candidate is required",
        )
    if any(bit_width <= 0 or bit_width > max_bit_width for bit_width in candidate_set):
        raise PrecisionPlanError(
            "invalid_precision_candidates",
            "precision candidates must be positive and <= max_bit_width",
        )

    descriptors = block_map(blocks)
    if mode == "fp16":
        return PrecisionPlan(
            mode=mode,
            precision_candidates=tuple(sorted(candidate_set)),
            decisions={},
            decision_source="full_precision",
            query_id=query_id,
        )
    if mode == "static_8bit":
        decisions = {block_id: 8 for block_id in descriptors}
        source = "static"
    elif mode == "static_4bit":
        decisions = {block_id: 4 for block_id in descriptors}
        source = "static"
    elif mode == "fixed_mixed":
        decisions = _require_mapping(
            fixed_precision_by_block,
            code="missing_fixed_profile",
            message="fixed_mixed mode requires fixed_precision_by_block",
        )
        source = "fixed_profile"
    elif mode in QAQ_MODES:
        decisions = _require_mapping(
            router_decisions,
            code="missing_router_decisions",
            message="QAQ modes require router_decisions",
        )
        source = "router"
    else:
        raise PrecisionPlanError("invalid_mode", f"unsupported mode {mode}")

    _validate_decisions(
        descriptors,
        decisions,
        candidate_set=candidate_set,
        max_bit_width=max_bit_width,
        require_artifacts=require_artifacts,
    )
    return PrecisionPlan(
        mode=mode,
        precision_candidates=tuple(sorted(candidate_set)),
        decisions=dict(decisions),
        decision_source=source,
        query_id=query_id,
        router_checkpoint=router_checkpoint,
        temperature=temperature,
        tie_break_policy=tie_break_policy,
    )


def _require_mapping(
    value: Mapping[str, int] | None,
    *,
    code: str,
    message: str,
) -> dict[str, int]:
    if value is None:
        raise PrecisionPlanError(code, message)
    if not isinstance(value, Mapping):
        raise PrecisionPlanError(code, message)
    return dict(value)


def _validate_decisions(
    descriptors: dict[str, BlockDescriptor],
    decisions: Mapping[str, int],
    *,
    candidate_set: set[int],
    max_bit_width: int,
    require_artifacts: bool,
) -> None:
    expected_ids = set(descriptors)
    decision_ids = set(decisions)
    missing = sorted(expected_ids - decision_ids)
    if missing:
        raise PrecisionPlanError(
            "missing_precision_decision",
            f"missing decisions for blocks: {missing}",
        )
    extra = sorted(decision_ids - expected_ids)
    if extra:
        raise PrecisionPlanError(
            "unknown_block_id",
            f"decisions reference unknown blocks: {extra}",
        )

    for block_id, bit_width in decisions.items():
        if isinstance(bit_width, bool) or not isinstance(bit_width, int):
            raise PrecisionPlanError(
                "invalid_precision_decision",
                f"{block_id} precision must be an integer",
            )
        if bit_width <= 0 or bit_width > max_bit_width:
            raise PrecisionPlanError(
                "invalid_precision_decision",
                f"{block_id} precision must be positive and <= max_bit_width",
            )
        if bit_width not in candidate_set:
            raise PrecisionPlanError(
                "invalid_precision_decision",
                f"{block_id} precision is not in precision_candidates",
            )
        descriptor = descriptors[block_id]
        if bit_width not in descriptor.supported_bit_widths:
            raise PrecisionPlanError(
                "unsupported_block_precision",
                f"{block_id} does not support {bit_width}-bit precision",
            )
        if (
            require_artifacts
            and str(bit_width) not in descriptor.artifact_refs
            and not _has_full_tensor_artifacts(descriptor)
        ):
            raise PrecisionPlanError(
                "missing_artifact",
                f"{block_id} is missing artifact for {bit_width}-bit precision",
            )


def _has_full_tensor_artifacts(descriptor: BlockDescriptor) -> bool:
    return all(tensor_name in descriptor.artifact_refs for tensor_name in descriptor.tensor_names)
