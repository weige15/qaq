"""Router policy data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qaq.precision_plan import PrecisionPlan


DEFAULT_DECISION_POLICY = "argmax_lowest_bit_width"
ROUTER_CHECKPOINT_VERSION = "qaq.router.v1"


@dataclass(slots=True)
class RouterPolicyError(ValueError):
    """Raised when router policy inputs or outputs are invalid."""

    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class RouterCheckpointMetadata:
    """Compatibility metadata for a reusable router checkpoint."""

    checkpoint_id: str
    model_id: str
    block_ids: tuple[str, ...]
    candidate_bit_widths: tuple[int, ...]
    feature_source: str
    hidden_size: int
    temperature: float = 1.0
    decision_policy: str = DEFAULT_DECISION_POLICY
    max_bit_width: int | None = None
    version: str = ROUTER_CHECKPOINT_VERSION
    diagnostic: bool = False
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "model_id": self.model_id,
            "block_ids": list(self.block_ids),
            "candidate_bit_widths": list(self.candidate_bit_widths),
            "feature_source": self.feature_source,
            "hidden_size": self.hidden_size,
            "temperature": self.temperature,
            "decision_policy": self.decision_policy,
            "max_bit_width": self.max_bit_width,
            "version": self.version,
            "diagnostic": self.diagnostic,
            "training_metadata": dict(self.training_metadata),
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RouterCheckpointMetadata":
        candidate_bit_widths = tuple(_require_int_list(value, "candidate_bit_widths"))
        max_bit_width = value.get("max_bit_width")
        return cls(
            checkpoint_id=_require_string(value, "checkpoint_id"),
            model_id=_require_string(value, "model_id"),
            block_ids=tuple(_require_string_list(value, "block_ids")),
            candidate_bit_widths=tuple(sorted(candidate_bit_widths)),
            feature_source=_require_string(value, "feature_source"),
            hidden_size=_require_int(value, "hidden_size"),
            temperature=_require_float(value, "temperature", default=1.0),
            decision_policy=value.get("decision_policy", DEFAULT_DECISION_POLICY),
            max_bit_width=(
                _require_int_value(max_bit_width, "max_bit_width")
                if max_bit_width is not None
                else None
            ),
            version=value.get("version", ROUTER_CHECKPOINT_VERSION),
            diagnostic=_require_bool(value, "diagnostic", default=False),
            training_metadata=_optional_dict(value, "training_metadata"),
        )


@dataclass(frozen=True, slots=True)
class RouterBlockParameters:
    """Single linear scoring layer for one controlled block."""

    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": [list(row) for row in self.weights],
            "bias": list(self.bias),
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RouterBlockParameters":
        weights_raw = value.get("weights")
        if not isinstance(weights_raw, list) or not weights_raw:
            raise RouterPolicyError(
                "invalid_router_parameters",
                "weights must be a non-empty 2D list",
            )
        weights = tuple(tuple(_coerce_float(item) for item in row) for row in weights_raw)
        if any(not row for row in weights):
            raise RouterPolicyError(
                "invalid_router_parameters",
                "weight rows must be non-empty",
            )
        row_width = len(weights[0])
        if any(len(row) != row_width for row in weights):
            raise RouterPolicyError(
                "invalid_router_parameters",
                "weight rows must all have the same width",
            )

        bias_raw = value.get("bias")
        if not isinstance(bias_raw, list) or not bias_raw:
            raise RouterPolicyError(
                "invalid_router_parameters",
                "bias must be a non-empty list",
            )
        bias = tuple(_coerce_float(item) for item in bias_raw)
        return cls(weights=weights, bias=bias)


@dataclass(frozen=True, slots=True)
class RouterTrace:
    """Trace for one query/block precision decision."""

    query_id: str
    block_id: str
    raw_scores: dict[int, float]
    probabilities: dict[int, float]
    selected_bit_width: int
    temperature: float
    checkpoint_id: str
    feature_source: str
    decision_policy: str
    tie_break_applied: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "block_id": self.block_id,
            "raw_scores": {str(key): value for key, value in self.raw_scores.items()},
            "probabilities": {str(key): value for key, value in self.probabilities.items()},
            "selected_bit_width": self.selected_bit_width,
            "temperature": self.temperature,
            "checkpoint_id": self.checkpoint_id,
            "feature_source": self.feature_source,
            "decision_policy": self.decision_policy,
            "tie_break_applied": self.tie_break_applied,
        }


@dataclass(frozen=True, slots=True)
class RouterSummary:
    """Aggregate routing evidence for later acceptance guards."""

    total_decisions: int
    precision_counts: dict[int, int]
    per_block_precision_counts: dict[str, dict[int, int]]
    per_query_decisions: dict[str, dict[str, int]]
    constant_global_precision: bool
    constant_precision_flagged: bool
    diagnostic: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_decisions": self.total_decisions,
            "precision_counts": {
                str(key): value for key, value in self.precision_counts.items()
            },
            "per_block_precision_counts": {
                block_id: {str(key): value for key, value in counts.items()}
                for block_id, counts in self.per_block_precision_counts.items()
            },
            "per_query_decisions": {
                query_id: dict(decisions)
                for query_id, decisions in self.per_query_decisions.items()
            },
            "constant_global_precision": self.constant_global_precision,
            "constant_precision_flagged": self.constant_precision_flagged,
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True, slots=True)
class RoutingResult:
    """Router output for a batch of queries."""

    plans: tuple[PrecisionPlan, ...]
    traces: tuple[RouterTrace, ...]
    summary: RouterSummary

    def as_dict(self) -> dict[str, Any]:
        return {
            "plans": [plan.as_dict() for plan in self.plans],
            "traces": [trace.as_dict() for trace in self.traces],
            "summary": self.summary.as_dict(),
        }


def _require_string(value: dict[str, Any], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw:
        raise RouterPolicyError("invalid_router_metadata", f"{key} must be a string")
    return raw


def _require_string_list(value: dict[str, Any], key: str) -> list[str]:
    raw = value.get(key)
    if not isinstance(raw, list) or not raw:
        raise RouterPolicyError(
            "invalid_router_metadata",
            f"{key} must be a non-empty list",
        )
    if any(not isinstance(item, str) or not item for item in raw):
        raise RouterPolicyError(
            "invalid_router_metadata",
            f"{key} must contain non-empty strings",
        )
    return raw


def _require_int_list(value: dict[str, Any], key: str) -> list[int]:
    raw = value.get(key)
    if not isinstance(raw, list) or not raw:
        raise RouterPolicyError(
            "invalid_router_metadata",
            f"{key} must be a non-empty list",
        )
    return [_require_int_value(item, key) for item in raw]


def _require_int(value: dict[str, Any], key: str) -> int:
    return _require_int_value(value.get(key), key)


def _require_int_value(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RouterPolicyError("invalid_router_metadata", f"{key} must be an integer")
    return value


def _require_float(value: dict[str, Any], key: str, *, default: float) -> float:
    raw = value.get(key, default)
    return _coerce_float(raw)


def _require_bool(value: dict[str, Any], key: str, *, default: bool) -> bool:
    raw = value.get(key, default)
    if not isinstance(raw, bool):
        raise RouterPolicyError("invalid_router_metadata", f"{key} must be a boolean")
    return raw


def _optional_dict(value: dict[str, Any], key: str) -> dict[str, Any]:
    raw = value.get(key, {})
    if not isinstance(raw, dict):
        raise RouterPolicyError("invalid_router_metadata", f"{key} must be an object")
    return dict(raw)


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RouterPolicyError("invalid_router_parameters", "expected a numeric value")
    return float(value)
