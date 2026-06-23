import json
from pathlib import Path

from qaq.artifacts import save_bitplane_artifact
from qaq.bitplanes import create_bitplane_artifact_from_quantized_values
from qaq.blocks import discover_mha_ffn_blocks
from qaq.config import RunConfig
from qaq.evaluate import main as evaluate_main
from qaq.model_adapter import load_model_adapter
from qaq.report import build_report
from qaq.router.checkpoint import RouterCheckpoint, save_router_checkpoint
from qaq.router.types import (
    DEFAULT_DECISION_POLICY,
    RouterBlockParameters,
    RouterCheckpointMetadata,
)
from qaq.results import (
    COMPARISON_REQUIRED_MODES,
    build_result_artifact,
    save_result_artifact,
)
from qaq.runtime.adaptive import (
    run_adaptive_runtime,
    validate_adaptive_acceptance_metadata,
)
from qaq.runtime.static import run_static_runtime


def _base_config(tmp_path: Path, mode: str, **overrides: object) -> RunConfig:
    data = {
        "model": "fake-qaq-smoke-model",
        "tokenizer": "fake-qaq-smoke-tokenizer",
        "dataset": "fake_smoke",
        "split": "validation",
        "mode": mode,
        "precision_candidates": [4, 8],
        "max_bit_width": 8,
        "block_granularity": "mha_ffn",
        "device": "cpu",
        "gpu_ids": [],
        "seed": 0,
        "output_dir": str(tmp_path / mode),
        "overwrite": False,
        "logging": {"console": False},
        "prompt_format": "fake_smoke_v1",
        "metric": "exact_match",
    }
    data.update(overrides)
    return RunConfig.from_mapping(data, validate_output=False)


def _refs(tmp_path: Path, config: RunConfig) -> dict[str, dict[str, str]]:
    adapter = load_model_adapter(config)
    refs: dict[str, dict[str, str]] = {}
    for index, block in enumerate(discover_mha_ffn_blocks(adapter.architecture_metadata)):
        artifact = create_bitplane_artifact_from_quantized_values(
            [[index, 255 - index]],
            model_id=config.model,
            block_id=block.block_id,
            tensor_name=block.tensor_names[0],
            max_bit_width=8,
        )
        path = save_bitplane_artifact(
            artifact,
            tmp_path / "artifacts" / f"{block.block_id}.json",
        )
        refs[block.block_id] = {"4": str(path), "8": str(path)}
    return refs


def _router_checkpoint(tmp_path: Path, config: RunConfig) -> Path:
    adapter = load_model_adapter(config)
    blocks = discover_mha_ffn_blocks(
        adapter.architecture_metadata,
        supported_bit_widths=config.precision_candidates,
    )
    block_ids = tuple(block.block_id for block in blocks)
    checkpoint = RouterCheckpoint(
        metadata=RouterCheckpointMetadata(
            checkpoint_id="adaptive-smoke-router",
            model_id=config.model,
            block_ids=block_ids,
            candidate_bit_widths=config.precision_candidates,
            feature_source=adapter.feature_source,
            hidden_size=adapter.architecture_metadata.hidden_size,
            temperature=1.0,
            decision_policy=DEFAULT_DECISION_POLICY,
            max_bit_width=config.max_bit_width,
            diagnostic=False,
        ),
        parameters={
            block_id: RouterBlockParameters(
                weights=((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
                bias=(0.0, -0.85),
            )
            for block_id in block_ids
        },
    )
    return save_router_checkpoint(checkpoint, tmp_path / "router.json")


def test_fake_prompt_completes_static_runtime_smoke_modes(tmp_path: Path) -> None:
    fp16 = _base_config(tmp_path, "fp16")
    refs = _refs(tmp_path, fp16)
    modes = {
        "fp16": fp16,
        "static_8bit": _base_config(tmp_path, "static_8bit"),
        "static_4bit": _base_config(tmp_path, "static_4bit"),
        "fixed_mixed": _base_config(
            tmp_path,
            "fixed_mixed",
            fixed_precision_by_block={
                block_id: 4 if index % 2 == 0 else 8
                for index, block_id in enumerate(refs)
            },
        ),
    }

    outputs = {
        mode: run_static_runtime(config, artifact_refs=refs if mode != "fp16" else None)
        for mode, config in modes.items()
    }

    assert set(outputs) == {"fp16", "static_8bit", "static_4bit", "fixed_mixed"}
    assert all(output.status == "completed" for output in outputs.values())
    assert outputs["fp16"].precision_plan.decision_source == "full_precision"
    assert outputs["fixed_mixed"].metadata["fixed_mixed_is_diagnostic"] is True
    assert len(outputs["static_8bit"].reconstruction_records) == len(refs)
    assert len(outputs["static_4bit"].reconstruction_records) == len(refs)
    assert all(output.latency_events[0].elapsed_seconds >= 0 for output in outputs.values())
    assert all(output.memory_events[0].peak_gpu_memory_gb == 0.0 for output in outputs.values())


def test_fake_prompt_completes_qaq_on_demand_off_and_on_with_shared_routing(
    tmp_path: Path,
) -> None:
    base = _base_config(tmp_path, "qaq_on_demand_off", router_diagnostic=True)
    refs = _refs(tmp_path, base)
    checkpoint = _router_checkpoint(tmp_path, base)
    off_config = _base_config(
        tmp_path,
        "qaq_on_demand_off",
        router_checkpoint=str(checkpoint),
    )
    on_config = _base_config(
        tmp_path,
        "qaq_on_demand_on",
        router_checkpoint=str(checkpoint),
    )

    off_output = run_adaptive_runtime(off_config, artifact_refs=refs)
    on_output = run_adaptive_runtime(on_config, artifact_refs=refs)

    assert off_output.status == "completed"
    assert on_output.status == "completed"
    assert off_output.raw_output.predictions == on_output.raw_output.predictions
    assert off_output.metadata["routing_summary"] == on_output.metadata["routing_summary"]
    assert off_output.metadata["loader_summary"] is None
    assert on_output.metadata["loader_summary"]["loads"] > 0
    assert on_output.metadata["loader_summary"]["total_bytes_transferred"] > 0
    assert [
        plan["decisions"] for plan in off_output.metadata["precision_plans"]
    ] == [plan["decisions"] for plan in on_output.metadata["precision_plans"]]
    assert len(off_output.metadata["adaptive_traces"]) == 2
    assert len(on_output.metadata["adaptive_traces"]) == 2
    assert all(
        len(trace["routing_trace_refs"]) == len(refs)
        for trace in on_output.metadata["adaptive_traces"]
    )
    assert all(
        len(trace["loader_request_refs"]) == len(refs)
        for trace in on_output.metadata["adaptive_traces"]
    )
    validate_adaptive_acceptance_metadata(off_output)
    validate_adaptive_acceptance_metadata(on_output)


def test_fake_prompt_static_and_adaptive_results_group_through_reporter(
    tmp_path: Path,
) -> None:
    base = _base_config(tmp_path, "fp16")
    refs = _refs(tmp_path, base)
    checkpoint = _router_checkpoint(tmp_path, base)
    result_paths = []

    for mode in ("fp16", "static_8bit", "static_4bit"):
        config = _base_config(tmp_path, mode)
        output = run_static_runtime(
            config,
            artifact_refs=refs if mode != "fp16" else None,
        )
        result_paths.append(
            save_result_artifact(
                build_result_artifact(config, output),
                tmp_path / "results" / f"{mode}.json",
            )
        )

    for mode in ("qaq_on_demand_off", "qaq_on_demand_on"):
        config = _base_config(tmp_path, mode, router_checkpoint=str(checkpoint))
        output = run_adaptive_runtime(config, artifact_refs=refs)
        validate_adaptive_acceptance_metadata(output)
        result_paths.append(
            save_result_artifact(
                build_result_artifact(config, output),
                tmp_path / "results" / f"{mode}.json",
            )
        )

    report = build_report(result_paths)
    comparison = report["comparisons"][0]
    rows_by_mode = {row["mode"]: row for row in report["rows"]}

    assert report["schema_version"] == "qaq.report.v1"
    assert len(report["comparisons"]) == 1
    assert comparison["validation"]["state"] == "diagnostic"
    assert comparison["validation"]["missing_modes"] == []
    assert comparison["validation"]["present_modes"] == list(COMPARISON_REQUIRED_MODES)
    assert set(rows_by_mode) == set(COMPARISON_REQUIRED_MODES)
    assert rows_by_mode["qaq_on_demand_off"]["routing_constant"] is False
    assert rows_by_mode["qaq_on_demand_on"]["routing_constant"] is False
    assert rows_by_mode["qaq_on_demand_on"]["loader_loads"] > 0


def test_evaluate_cli_runs_static_smoke_config(tmp_path: Path, capsys) -> None:
    config = _base_config(tmp_path, "static_8bit")
    refs = _refs(tmp_path, config)
    config_path = tmp_path / "config.json"
    artifact_index = tmp_path / "artifact-index.json"
    config_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    artifact_index.write_text(json.dumps(refs), encoding="utf-8")

    exit_code = evaluate_main(
        [
            "--config",
            str(config_path),
            "--artifact-index",
            str(artifact_index),
            "--skip-output-dir-check",
            "--print-json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["mode"] == "static_8bit"
    assert payload["status"] == "completed"
    assert payload["precision_plan"]["decision_source"] == "static"


def test_evaluate_cli_runs_qaq_on_demand_on_smoke_config(
    tmp_path: Path,
    capsys,
) -> None:
    base = _base_config(tmp_path, "qaq_on_demand_on", router_diagnostic=True)
    refs = _refs(tmp_path, base)
    checkpoint = _router_checkpoint(tmp_path, base)
    config = _base_config(
        tmp_path,
        "qaq_on_demand_on",
        router_checkpoint=str(checkpoint),
    )
    config_path = tmp_path / "qaq-config.json"
    artifact_index = tmp_path / "qaq-artifact-index.json"
    config_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    artifact_index.write_text(json.dumps(refs), encoding="utf-8")

    exit_code = evaluate_main(
        [
            "--config",
            str(config_path),
            "--artifact-index",
            str(artifact_index),
            "--skip-output-dir-check",
            "--print-json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["mode"] == "qaq_on_demand_on"
    assert payload["status"] == "completed"
    assert payload["precision_plan"]["decision_source"] == "router"
    assert payload["metadata"]["loader_summary"]["loads"] > 0
    assert len(payload["metadata"]["adaptive_traces"]) == 2
