"""Lightweight router inference over hidden features."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from qaq.blocks import BlockDescriptor
from qaq.model_adapter import HiddenStateBundle
from qaq.precision_plan import build_precision_plan
from qaq.router.checkpoint import RouterCheckpoint, validate_checkpoint_compatibility
from qaq.router.types import (
    DEFAULT_DECISION_POLICY,
    RouterPolicyError,
    RouterSummary,
    RouterTrace,
    RoutingResult,
)


PROBABILITY_TOLERANCE = 1e-9


def route_hidden_states(
    checkpoint: RouterCheckpoint,
    *,
    hidden_states: HiddenStateBundle,
    blocks: tuple[BlockDescriptor, ...],
    model_id: str,
    mode: str,
    query_ids: tuple[str, ...] | None = None,
    diagnostic: bool | None = None,
) -> RoutingResult:
    """Route each query/block hidden feature to a deterministic precision plan."""

    validate_checkpoint_compatibility(
        checkpoint,
        blocks=blocks,
        model_id=model_id,
        candidate_bit_widths=checkpoint.metadata.candidate_bit_widths,
        feature_source=hidden_states.feature_source,
    )
    block_ids = tuple(block.block_id for block in blocks)
    _validate_hidden_state_blocks(hidden_states, block_ids)
    query_count = _query_count(hidden_states, block_ids)
    resolved_query_ids = query_ids or tuple(f"query-{index}" for index in range(query_count))
    if len(resolved_query_ids) != query_count:
        raise RouterPolicyError(
            "query_count_mismatch",
            "query_ids length must match hidden-state query count",
        )

    traces: list[RouterTrace] = []
    plans = []
    for query_index, query_id in enumerate(resolved_query_ids):
        decisions: dict[str, int] = {}
        for block_id in block_ids:
            feature = hidden_states.by_block[block_id][query_index]
            raw_scores = score_block(checkpoint, block_id=block_id, feature=feature)
            probabilities = normalize_scores(
                raw_scores,
                temperature=checkpoint.metadata.temperature,
            )
            selected, tie_break_applied = select_bit_width(
                probabilities,
                decision_policy=checkpoint.metadata.decision_policy,
            )
            decisions[block_id] = selected
            traces.append(
                RouterTrace(
                    query_id=query_id,
                    block_id=block_id,
                    raw_scores=raw_scores,
                    probabilities=probabilities,
                    selected_bit_width=selected,
                    temperature=checkpoint.metadata.temperature,
                    checkpoint_id=checkpoint.metadata.checkpoint_id,
                    feature_source=checkpoint.metadata.feature_source,
                    decision_policy=checkpoint.metadata.decision_policy,
                    tie_break_applied=tie_break_applied,
                )
            )
        plans.append(
            build_precision_plan(
                blocks,
                mode=mode,
                precision_candidates=checkpoint.metadata.candidate_bit_widths,
                max_bit_width=checkpoint.metadata.max_bit_width
                or max(checkpoint.metadata.candidate_bit_widths),
                router_decisions=decisions,
                query_id=query_id,
                router_checkpoint=checkpoint.metadata.checkpoint_id,
                temperature=checkpoint.metadata.temperature,
                tie_break_policy=checkpoint.metadata.decision_policy,
            )
        )

    summary = summarize_traces(
        tuple(traces),
        diagnostic=checkpoint.metadata.diagnostic if diagnostic is None else diagnostic,
    )
    return RoutingResult(plans=tuple(plans), traces=tuple(traces), summary=summary)


def score_block(
    checkpoint: RouterCheckpoint,
    *,
    block_id: str,
    feature: tuple[float, ...],
) -> dict[int, float]:
    """Compute linear router scores for one block feature vector."""

    checkpoint.validate()
    if block_id not in checkpoint.parameters:
        raise RouterPolicyError(
            "missing_router_block",
            f"checkpoint has no parameters for {block_id}",
        )
    if len(feature) != checkpoint.metadata.hidden_size:
        raise RouterPolicyError(
            "feature_size_mismatch",
            f"feature width {len(feature)} does not match hidden_size {checkpoint.metadata.hidden_size}",
        )
    if any(not math.isfinite(value) for value in feature):
        raise RouterPolicyError(
            "non_finite_feature",
            "router features must be finite values",
        )

    params = checkpoint.parameters[block_id]
    scores: dict[int, float] = {}
    for bit_width, weights, bias in zip(
        checkpoint.metadata.candidate_bit_widths,
        params.weights,
        params.bias,
        strict=True,
    ):
        score = sum(weight * value for weight, value in zip(weights, feature, strict=True)) + bias
        if not math.isfinite(score):
            raise RouterPolicyError(
                "non_finite_score",
                f"router score for {block_id}/{bit_width} is not finite",
            )
        scores[bit_width] = score
    return scores


def normalize_scores(
    raw_scores: dict[int, float],
    *,
    temperature: float,
) -> dict[int, float]:
    """Normalize raw scores with the paper-style temperature sharpness factor."""

    if temperature <= 0 or not math.isfinite(temperature):
        raise RouterPolicyError(
            "invalid_temperature",
            "temperature must be finite and positive",
        )
    if not raw_scores:
        raise RouterPolicyError("missing_scores", "raw_scores are required")
    if any(not math.isfinite(value) for value in raw_scores.values()):
        raise RouterPolicyError("non_finite_score", "raw scores must be finite")

    scaled = {bit_width: score * temperature for bit_width, score in raw_scores.items()}
    max_score = max(scaled.values())
    exp_values = {
        bit_width: math.exp(score - max_score) for bit_width, score in scaled.items()
    }
    total = sum(exp_values.values())
    if total <= 0 or not math.isfinite(total):
        raise RouterPolicyError(
            "invalid_probability_distribution",
            "softmax denominator is invalid",
        )
    probabilities = {
        bit_width: value / total for bit_width, value in exp_values.items()
    }
    validate_probabilities(probabilities)
    return probabilities


def validate_probabilities(probabilities: dict[int, float]) -> None:
    if not probabilities:
        raise RouterPolicyError("missing_probabilities", "probabilities are required")
    if any(value < 0 or not math.isfinite(value) for value in probabilities.values()):
        raise RouterPolicyError(
            "invalid_probability_distribution",
            "probabilities must be finite and non-negative",
        )
    total = sum(probabilities.values())
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise RouterPolicyError(
            "invalid_probability_distribution",
            f"probabilities must sum to 1, got {total}",
        )


def select_bit_width(
    probabilities: dict[int, float],
    *,
    decision_policy: str = DEFAULT_DECISION_POLICY,
) -> tuple[int, bool]:
    """Convert probabilities to a deterministic bit-width decision."""

    validate_probabilities(probabilities)
    max_probability = max(probabilities.values())
    tied = tuple(
        sorted(
            bit_width
            for bit_width, probability in probabilities.items()
            if abs(probability - max_probability) <= PROBABILITY_TOLERANCE
        )
    )
    if decision_policy == DEFAULT_DECISION_POLICY:
        return tied[0], len(tied) > 1
    if decision_policy == "argmax_highest_bit_width":
        return tied[-1], len(tied) > 1
    raise RouterPolicyError(
        "unsupported_decision_policy",
        f"unsupported decision policy {decision_policy}",
    )


def summarize_traces(
    traces: tuple[RouterTrace, ...],
    *,
    diagnostic: bool = False,
) -> RouterSummary:
    if not traces:
        raise RouterPolicyError("missing_router_traces", "at least one trace is required")

    precision_counts = Counter(trace.selected_bit_width for trace in traces)
    per_block: dict[str, Counter[int]] = defaultdict(Counter)
    per_query: dict[str, dict[str, int]] = defaultdict(dict)
    for trace in traces:
        per_block[trace.block_id][trace.selected_bit_width] += 1
        per_query[trace.query_id][trace.block_id] = trace.selected_bit_width

    constant_global_precision = len(precision_counts) == 1
    return RouterSummary(
        total_decisions=len(traces),
        precision_counts=dict(sorted(precision_counts.items())),
        per_block_precision_counts={
            block_id: dict(sorted(counts.items()))
            for block_id, counts in sorted(per_block.items())
        },
        per_query_decisions={query_id: decisions for query_id, decisions in sorted(per_query.items())},
        constant_global_precision=constant_global_precision,
        constant_precision_flagged=constant_global_precision and not diagnostic,
        diagnostic=diagnostic,
    )


def _validate_hidden_state_blocks(
    hidden_states: HiddenStateBundle,
    block_ids: tuple[str, ...],
) -> None:
    hidden_block_ids = set(hidden_states.by_block)
    expected = set(block_ids)
    missing = sorted(expected - hidden_block_ids)
    extra = sorted(hidden_block_ids - expected)
    if missing:
        raise RouterPolicyError(
            "missing_hidden_state_block",
            f"hidden states are missing blocks: {missing}",
        )
    if extra:
        raise RouterPolicyError(
            "unknown_hidden_state_block",
            f"hidden states include unknown blocks: {extra}",
        )


def _query_count(hidden_states: HiddenStateBundle, block_ids: tuple[str, ...]) -> int:
    counts = {block_id: len(hidden_states.by_block[block_id]) for block_id in block_ids}
    if any(count <= 0 for count in counts.values()):
        raise RouterPolicyError(
            "missing_hidden_state_features",
            "each block must have at least one query feature",
        )
    if len(set(counts.values())) != 1:
        raise RouterPolicyError(
            "query_count_mismatch",
            "all blocks must expose the same query count",
        )
    return next(iter(counts.values()))
