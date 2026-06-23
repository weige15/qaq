"""Router-training loss helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


ROUTER_COST_CROSS_ENTROPY = "router_cost_cross_entropy"
SUPPORTED_DISTILLATION_LOSSES = frozenset({ROUTER_COST_CROSS_ENTROPY})
PROBABILITY_EPSILON = 1e-12


@dataclass(slots=True)
class RouterLossError(ValueError):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class LossRecord:
    """Single router-training loss measurement."""

    step: int
    loss: float
    distillation_loss: float
    efficiency_penalty: float
    learning_rate: float
    objective: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "distillation_loss": self.distillation_loss,
            "efficiency_penalty": self.efficiency_penalty,
            "learning_rate": self.learning_rate,
            "objective": self.objective,
        }


@dataclass(frozen=True, slots=True)
class RouterObjectiveSample:
    """One sample/block contribution to the router training objective."""

    target_probabilities: Mapping[int, float]
    router_probabilities: Mapping[int, float]
    candidate_distillation_costs: Mapping[int, float]
    candidate_efficiency_penalties: Mapping[int, float]


def softmax_from_costs(
    costs: Mapping[int, float],
    *,
    temperature: float,
) -> dict[int, float]:
    """Convert lower-is-better candidate costs into target probabilities."""

    if temperature <= 0 or not math.isfinite(temperature):
        raise RouterLossError(
            "invalid_target_temperature",
            "target_temperature must be finite and positive",
        )
    _validate_probability_keys(costs, field="costs")
    scaled = {bit_width: -cost / temperature for bit_width, cost in costs.items()}
    max_value = max(scaled.values())
    exp_values = {
        bit_width: math.exp(value - max_value) for bit_width, value in scaled.items()
    }
    total = sum(exp_values.values())
    if total <= 0 or not math.isfinite(total):
        raise RouterLossError(
            "invalid_target_distribution",
            "target probability denominator is invalid",
        )
    return {bit_width: value / total for bit_width, value in exp_values.items()}


def compute_router_objective_loss(
    *,
    samples: tuple[RouterObjectiveSample, ...],
    objective: str,
    step: int,
    learning_rate: float,
) -> LossRecord:
    """Compute cross-entropy from cost-derived targets to router probabilities."""

    _validate_objective_inputs(
        objective=objective,
        step=step,
        learning_rate=learning_rate,
    )
    if not samples:
        raise RouterLossError(
            "empty_objective_batch",
            "at least one router objective sample is required",
        )

    total_cross_entropy = 0.0
    total_distillation = 0.0
    total_efficiency = 0.0
    for sample in samples:
        target = _validate_probabilities(sample.target_probabilities, field="target")
        router = _validate_probabilities(sample.router_probabilities, field="router")
        if set(target) != set(router):
            raise RouterLossError(
                "objective_candidate_mismatch",
                "target and router probabilities must use the same candidates",
            )
        distillation_costs = _validate_probability_keys(
            sample.candidate_distillation_costs,
            field="candidate_distillation_costs",
        )
        efficiency_penalties = _validate_probability_keys(
            sample.candidate_efficiency_penalties,
            field="candidate_efficiency_penalties",
        )
        if set(target) != set(distillation_costs) or set(target) != set(efficiency_penalties):
            raise RouterLossError(
                "objective_candidate_mismatch",
                "candidate costs and probabilities must use the same candidates",
            )

        total_cross_entropy += -sum(
            target[bit_width] * math.log(max(router[bit_width], PROBABILITY_EPSILON))
            for bit_width in target
        )
        total_distillation += sum(
            target[bit_width] * distillation_costs[bit_width] for bit_width in target
        )
        total_efficiency += sum(
            target[bit_width] * efficiency_penalties[bit_width] for bit_width in target
        )

    sample_count = len(samples)
    return LossRecord(
        step=step,
        loss=total_cross_entropy / sample_count,
        distillation_loss=total_distillation / sample_count,
        efficiency_penalty=total_efficiency / sample_count,
        learning_rate=learning_rate,
        objective=objective,
    )


def compute_distillation_loss(
    *,
    teacher_logits: tuple[tuple[float, ...], ...],
    student_logits: tuple[tuple[float, ...], ...],
    objective: str,
    step: int,
    learning_rate: float,
    efficiency_penalty: float = 0.0,
) -> LossRecord:
    """Compute a direct teacher/student MSE loss.

    This remains available for callers that already have routed student logits,
    but accepted router training uses ``router_cost_cross_entropy``.
    """

    _validate_objective_inputs(
        objective=objective,
        step=step,
        learning_rate=learning_rate,
    )
    if efficiency_penalty < 0 or not math.isfinite(efficiency_penalty):
        raise RouterLossError(
            "invalid_efficiency_penalty",
            "efficiency_penalty must be finite and non-negative",
        )

    distillation = mean_squared_error(teacher_logits, student_logits)
    loss = distillation + efficiency_penalty
    return LossRecord(
        step=step,
        loss=loss,
        distillation_loss=distillation,
        efficiency_penalty=efficiency_penalty,
        learning_rate=learning_rate,
        objective=objective,
    )


def _validate_objective_inputs(
    *,
    objective: str,
    step: int,
    learning_rate: float,
) -> None:
    if objective not in SUPPORTED_DISTILLATION_LOSSES:
        raise RouterLossError(
            "unsupported_distillation_loss",
            f"unsupported router distillation loss {objective!r}",
        )
    if step <= 0:
        raise RouterLossError("invalid_step", "step must be positive")
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise RouterLossError(
            "invalid_learning_rate",
            "learning_rate must be finite and positive",
        )


def mean_squared_error(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> float:
    left_rows = _validate_logits(left, field="left")
    right_rows = _validate_logits(right, field="right")
    if len(left_rows) != len(right_rows):
        raise RouterLossError("logit_shape_mismatch", "row counts must match")

    total = 0.0
    count = 0
    for left_row, right_row in zip(left_rows, right_rows, strict=True):
        if len(left_row) != len(right_row):
            raise RouterLossError(
                "logit_shape_mismatch",
                "all logit row widths must match",
            )
        for left_value, right_value in zip(left_row, right_row, strict=True):
            delta = left_value - right_value
            total += delta * delta
            count += 1
    if count == 0:
        raise RouterLossError("empty_logits", "at least one logit value is required")
    return total / count


def _validate_logits(
    logits: tuple[tuple[float, ...], ...],
    *,
    field: str,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(logits, tuple) or not logits:
        raise RouterLossError("empty_logits", f"{field} must contain rows")
    for row in logits:
        if not isinstance(row, tuple) or not row:
            raise RouterLossError("empty_logits", f"{field} rows must be non-empty")
        if any(not isinstance(value, int | float) or not math.isfinite(value) for value in row):
            raise RouterLossError(
                "non_finite_logits",
                f"{field} must contain finite numeric values",
            )
    return logits


def _validate_probability_keys(
    values: Mapping[int, float],
    *,
    field: str,
) -> dict[int, float]:
    if not values:
        raise RouterLossError("empty_objective_values", f"{field} must not be empty")
    result: dict[int, float] = {}
    for key, value in values.items():
        if isinstance(key, bool) or not isinstance(key, int) or key <= 0:
            raise RouterLossError(
                "invalid_candidate",
                f"{field} candidate bit-widths must be positive integers",
            )
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise RouterLossError(
                "invalid_objective_value",
                f"{field} values must be numeric",
            )
        numeric = float(value)
        if numeric < 0 or not math.isfinite(numeric):
            raise RouterLossError(
                "invalid_objective_value",
                f"{field} values must be finite and non-negative",
            )
        result[key] = numeric
    return result


def _validate_probabilities(
    values: Mapping[int, float],
    *,
    field: str,
) -> dict[int, float]:
    probabilities = _validate_probability_keys(values, field=field)
    total = sum(probabilities.values())
    if abs(total - 1.0) > 1e-9:
        raise RouterLossError(
            "invalid_probability_distribution",
            f"{field} probabilities must sum to 1",
        )
    return probabilities
