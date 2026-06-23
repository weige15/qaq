import json
from pathlib import Path

import pytest

from qaq.blocks import discover_mha_ffn_blocks
from qaq.config import RunConfig
from qaq.model_adapter import HiddenStateBundle, load_model_adapter
from qaq.router.checkpoint import RouterCheckpoint
from qaq.router.policy import normalize_scores, route_hidden_states, select_bit_width
from qaq.router.types import (
    DEFAULT_DECISION_POLICY,
    RouterBlockParameters,
    RouterCheckpointMetadata,
    RouterPolicyError,
)


GOLDEN = Path("tests/golden/router_decision.json")


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig.from_mapping(
        {
            "model": "fake-qaq-smoke-model",
            "tokenizer": "fake-qaq-smoke-tokenizer",
            "dataset": "fake_smoke",
            "split": "validation",
            "mode": "qaq_on_demand_off",
            "precision_candidates": [4, 8],
            "max_bit_width": 8,
            "block_granularity": "mha_ffn",
            "device": "cpu",
            "gpu_ids": [],
            "seed": 0,
            "output_dir": str(tmp_path / "run"),
            "overwrite": False,
            "logging": {"console": False},
            "router_diagnostic": True,
        },
        validate_output=False,
    )


def _checkpoint(
    block_ids: tuple[str, ...],
    *,
    model_id: str = "fake-qaq-smoke-model",
    feature_source: str = "block_output_pooled",
    hidden_size: int = 4,
    temperature: float = 1.0,
    favor_high: bool = True,
    diagnostic: bool = False,
) -> RouterCheckpoint:
    if favor_high:
        weights = ((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        bias = (0.0, 0.25)
    else:
        weights = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
        bias = (0.25, 0.0)
    return RouterCheckpoint(
        metadata=RouterCheckpointMetadata(
            checkpoint_id="router-fixture",
            model_id=model_id,
            block_ids=block_ids,
            candidate_bit_widths=(4, 8),
            feature_source=feature_source,
            hidden_size=hidden_size,
            temperature=temperature,
            decision_policy=DEFAULT_DECISION_POLICY,
            max_bit_width=8,
            diagnostic=diagnostic,
        ),
        parameters={
            block_id: RouterBlockParameters(weights=weights, bias=bias)
            for block_id in block_ids
        },
    )


def test_probability_normalization_is_valid_and_temperature_sharpens() -> None:
    low_temperature = normalize_scores({4: 0.0, 8: 1.0}, temperature=0.5)
    high_temperature = normalize_scores({4: 0.0, 8: 1.0}, temperature=2.0)

    assert pytest.approx(sum(low_temperature.values())) == 1.0
    assert pytest.approx(sum(high_temperature.values())) == 1.0
    assert high_temperature[8] > low_temperature[8]
    assert all(value >= 0 for value in high_temperature.values())


def test_deterministic_tie_break_matches_golden_fixture() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    probabilities = {int(key): value for key, value in golden["probabilities"].items()}

    selected, tie_break = select_bit_width(
        probabilities,
        decision_policy=golden["decision_policy"],
    )

    assert selected == golden["selected_bit_width"]
    assert tie_break is True


def test_router_policy_outputs_plan_and_trace_for_every_block(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter = load_model_adapter(config)
    examples = adapter.load_examples(config)
    batch = adapter.build_batch(config, examples)
    blocks = discover_mha_ffn_blocks(adapter.architecture_metadata)
    output = adapter.reference_forward(batch, block_ids=tuple(block.block_id for block in blocks))
    checkpoint = _checkpoint(tuple(block.block_id for block in blocks))

    routing = route_hidden_states(
        checkpoint,
        hidden_states=output.hidden_states,
        blocks=blocks,
        model_id=config.model,
        mode="qaq_on_demand_off",
        query_ids=batch.example_ids,
    )

    assert len(routing.plans) == len(examples)
    assert len(routing.traces) == len(examples) * len(blocks)
    assert routing.plans[0].decision_source == "router"
    assert set(routing.plans[0].decisions) == {block.block_id for block in blocks}
    assert all(trace.checkpoint_id == "router-fixture" for trace in routing.traces)
    assert all(trace.probabilities[8] > trace.probabilities[4] for trace in routing.traces)


def test_checkpoint_mismatches_and_missing_hidden_blocks_fail(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter = load_model_adapter(config)
    examples = adapter.load_examples(config)
    batch = adapter.build_batch(config, examples)
    blocks = discover_mha_ffn_blocks(adapter.architecture_metadata)
    output = adapter.reference_forward(batch, block_ids=tuple(block.block_id for block in blocks))

    wrong_model = _checkpoint(tuple(block.block_id for block in blocks), model_id="wrong")
    with pytest.raises(RouterPolicyError) as model_exc:
        route_hidden_states(
            wrong_model,
            hidden_states=output.hidden_states,
            blocks=blocks,
            model_id=config.model,
            mode="qaq_on_demand_off",
        )
    assert model_exc.value.code == "router_model_mismatch"

    missing_block_states = HiddenStateBundle(
        feature_source=output.hidden_states.feature_source,
        by_block={blocks[0].block_id: output.hidden_states.by_block[blocks[0].block_id]},
    )
    checkpoint = _checkpoint(tuple(block.block_id for block in blocks))
    with pytest.raises(RouterPolicyError) as hidden_exc:
        route_hidden_states(
            checkpoint,
            hidden_states=missing_block_states,
            blocks=blocks,
            model_id=config.model,
            mode="qaq_on_demand_off",
        )
    assert hidden_exc.value.code == "missing_hidden_state_block"


def test_constant_global_precision_is_flagged_unless_diagnostic(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter = load_model_adapter(config)
    examples = adapter.load_examples(config)
    batch = adapter.build_batch(config, examples)
    blocks = discover_mha_ffn_blocks(adapter.architecture_metadata)
    output = adapter.reference_forward(batch, block_ids=tuple(block.block_id for block in blocks))

    checkpoint = _checkpoint(tuple(block.block_id for block in blocks), favor_high=False)
    routing = route_hidden_states(
        checkpoint,
        hidden_states=output.hidden_states,
        blocks=blocks,
        model_id=config.model,
        mode="qaq_on_demand_off",
    )
    diagnostic_checkpoint = _checkpoint(
        tuple(block.block_id for block in blocks),
        favor_high=False,
        diagnostic=True,
    )
    diagnostic = route_hidden_states(
        diagnostic_checkpoint,
        hidden_states=output.hidden_states,
        blocks=blocks,
        model_id=config.model,
        mode="qaq_on_demand_off",
    )

    assert routing.summary.constant_global_precision is True
    assert routing.summary.constant_precision_flagged is True
    assert diagnostic.summary.constant_global_precision is True
    assert diagnostic.summary.constant_precision_flagged is False


def test_invalid_probabilities_fail() -> None:
    with pytest.raises(RouterPolicyError) as exc:
        select_bit_width({4: 0.25, 8: 0.25})

    assert exc.value.code == "invalid_probability_distribution"
