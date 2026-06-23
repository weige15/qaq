from dataclasses import replace

import pytest

from qaq.blocks import BlockRegistryError, discover_mha_ffn_blocks
from qaq.precision_plan import PrecisionPlanError, build_precision_plan
from tests.fixtures.fake_transformer import UnsupportedTransformer, make_fake_transformer


def test_fake_transformer_discovery_produces_stable_mha_ffn_ids() -> None:
    model = make_fake_transformer(num_layers=2)

    first = discover_mha_ffn_blocks(model, supported_bit_widths=(8, 4))
    second = discover_mha_ffn_blocks(model, supported_bit_widths=(8, 4))

    assert [block.block_id for block in first] == [
        "layer_000.mha",
        "layer_000.ffn",
        "layer_001.mha",
        "layer_001.ffn",
    ]
    assert first == second
    assert first[0].module_path == "layers.0.mha"
    assert first[0].tensor_names == (
        "layers.0.mha.q_proj.weight",
        "layers.0.mha.o_proj.weight",
    )
    assert first[0].supported_bit_widths == (4, 8)


def test_unsupported_transformer_layout_fails() -> None:
    with pytest.raises(BlockRegistryError) as exc:
        discover_mha_ffn_blocks(UnsupportedTransformer())

    assert exc.value.code == "unsupported_layout"


def test_invalid_supported_bit_widths_fail() -> None:
    with pytest.raises(BlockRegistryError) as exc:
        discover_mha_ffn_blocks(make_fake_transformer(), supported_bit_widths=(4, 4))

    assert exc.value.code == "invalid_supported_bit_widths"


def test_fp16_precision_plan_has_no_quantized_decisions() -> None:
    blocks = discover_mha_ffn_blocks(make_fake_transformer(num_layers=1))

    plan = build_precision_plan(
        blocks,
        mode="fp16",
        precision_candidates=(4, 8),
        max_bit_width=8,
    )

    assert plan.decision_source == "full_precision"
    assert plan.decisions == {}


def test_static_precision_plans_cover_every_block() -> None:
    blocks = discover_mha_ffn_blocks(make_fake_transformer(num_layers=1))

    plan_8 = build_precision_plan(
        blocks,
        mode="static_8bit",
        precision_candidates=(4, 8),
        max_bit_width=8,
    )
    plan_4 = build_precision_plan(
        blocks,
        mode="static_4bit",
        precision_candidates=(4, 8),
        max_bit_width=8,
    )

    assert plan_8.decisions == {"layer_000.mha": 8, "layer_000.ffn": 8}
    assert plan_4.decisions == {"layer_000.mha": 4, "layer_000.ffn": 4}
    assert plan_8.decision_source == "static"


def test_fixed_mixed_plan_requires_complete_supported_profile() -> None:
    blocks = discover_mha_ffn_blocks(make_fake_transformer(num_layers=1))

    with pytest.raises(PrecisionPlanError) as exc:
        build_precision_plan(
            blocks,
            mode="fixed_mixed",
            precision_candidates=(4, 8),
            max_bit_width=8,
            fixed_precision_by_block={"layer_000.mha": 4},
        )

    assert exc.value.code == "missing_precision_decision"

    plan = build_precision_plan(
        blocks,
        mode="fixed_mixed",
        precision_candidates=(4, 8),
        max_bit_width=8,
        fixed_precision_by_block={"layer_000.mha": 4, "layer_000.ffn": 8},
    )

    assert plan.decision_source == "fixed_profile"
    assert plan.decisions == {"layer_000.mha": 4, "layer_000.ffn": 8}


def test_precision_plan_rejects_unknown_blocks_and_invalid_bit_widths() -> None:
    blocks = discover_mha_ffn_blocks(make_fake_transformer(num_layers=1))

    with pytest.raises(PrecisionPlanError) as exc:
        build_precision_plan(
            blocks,
            mode="fixed_mixed",
            precision_candidates=(4, 8),
            max_bit_width=8,
            fixed_precision_by_block={
                "layer_000.mha": 4,
                "layer_000.ffn": 8,
                "layer_999.mha": 4,
            },
        )
    assert exc.value.code == "unknown_block_id"

    with pytest.raises(PrecisionPlanError) as exc:
        build_precision_plan(
            blocks,
            mode="fixed_mixed",
            precision_candidates=(4, 8),
            max_bit_width=8,
            fixed_precision_by_block={"layer_000.mha": 6, "layer_000.ffn": 8},
        )
    assert exc.value.code == "invalid_precision_decision"


def test_qaq_plan_requires_router_decisions_for_every_block() -> None:
    blocks = discover_mha_ffn_blocks(make_fake_transformer(num_layers=1))

    with pytest.raises(PrecisionPlanError) as exc:
        build_precision_plan(
            blocks,
            mode="qaq_on_demand_off",
            precision_candidates=(4, 8),
            max_bit_width=8,
            router_decisions={"layer_000.mha": 4},
        )

    assert exc.value.code == "missing_precision_decision"

    plan = build_precision_plan(
        blocks,
        mode="qaq_on_demand_on",
        precision_candidates=(4, 8),
        max_bit_width=8,
        router_decisions={"layer_000.mha": 4, "layer_000.ffn": 8},
        query_id="query-1",
        router_checkpoint="router.json",
        temperature=1.0,
        tie_break_policy="lowest_bit_width",
    )

    assert plan.decision_source == "router"
    assert plan.query_id == "query-1"
    assert plan.router_checkpoint == "router.json"
    assert plan.decisions == {"layer_000.mha": 4, "layer_000.ffn": 8}


def test_artifact_requirement_rejects_missing_artifacts() -> None:
    blocks = discover_mha_ffn_blocks(make_fake_transformer(num_layers=1))
    blocks_with_artifacts = tuple(
        replace(block, artifact_refs={"4": f"{block.block_id}.4.bin"})
        for block in blocks
    )

    with pytest.raises(PrecisionPlanError) as exc:
        build_precision_plan(
            blocks_with_artifacts,
            mode="static_8bit",
            precision_candidates=(4, 8),
            max_bit_width=8,
            require_artifacts=True,
        )

    assert exc.value.code == "missing_artifact"

    plan = build_precision_plan(
        blocks_with_artifacts,
        mode="static_4bit",
        precision_candidates=(4, 8),
        max_bit_width=8,
        require_artifacts=True,
    )

    assert plan.decisions == {"layer_000.mha": 4, "layer_000.ffn": 4}
