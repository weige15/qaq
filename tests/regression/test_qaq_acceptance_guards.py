from dataclasses import replace
from pathlib import Path

import pytest

from qaq.artifacts import save_bitplane_artifact
from qaq.bitplanes import create_bitplane_artifact_from_quantized_values
from qaq.blocks import discover_mha_ffn_blocks
from qaq.config import RunConfig
from qaq.model_adapter import load_model_adapter
from qaq.router.checkpoint import RouterCheckpoint, save_router_checkpoint
from qaq.router.types import (
    DEFAULT_DECISION_POLICY,
    RouterBlockParameters,
    RouterCheckpointMetadata,
)
from qaq.results import build_result_artifact, validate_comparison
from qaq.runtime.adaptive import (
    run_adaptive_runtime,
    validate_adaptive_acceptance_metadata,
)
from qaq.runtime.common import RuntimeError as QaqRuntimeError
from qaq.runtime.static import run_static_runtime


def _config(tmp_path: Path, mode: str, **overrides: object) -> RunConfig:
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
    blocks = discover_mha_ffn_blocks(
        adapter.architecture_metadata,
        supported_bit_widths=config.precision_candidates,
    )
    for index, block in enumerate(blocks):
        artifact = create_bitplane_artifact_from_quantized_values(
            [[index, 255 - index]],
            model_id=config.model,
            block_id=block.block_id,
            tensor_name=block.tensor_names[0],
            max_bit_width=config.max_bit_width,
            compatibility={"block_granularity": config.block_granularity},
        )
        path = save_bitplane_artifact(
            artifact,
            tmp_path / "artifacts" / f"{block.block_id}.json",
        )
        refs[block.block_id] = {"4": str(path), "8": str(path)}
    return refs


def _checkpoint(
    tmp_path: Path,
    config: RunConfig,
    *,
    constant: bool = False,
    diagnostic: bool = False,
) -> Path:
    adapter = load_model_adapter(config)
    blocks = discover_mha_ffn_blocks(
        adapter.architecture_metadata,
        supported_bit_widths=config.precision_candidates,
    )
    block_ids = tuple(block.block_id for block in blocks)
    params = (
        RouterBlockParameters(
            weights=((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
            bias=(0.0, 1.0),
        )
        if constant
        else RouterBlockParameters(
            weights=((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            bias=(0.0, -0.85),
        )
    )
    checkpoint = RouterCheckpoint(
        metadata=RouterCheckpointMetadata(
            checkpoint_id="adaptive-regression-router",
            model_id=config.model,
            block_ids=block_ids,
            candidate_bit_widths=config.precision_candidates,
            feature_source=adapter.feature_source,
            hidden_size=adapter.architecture_metadata.hidden_size,
            temperature=1.0,
            decision_policy=DEFAULT_DECISION_POLICY,
            max_bit_width=config.max_bit_width,
            diagnostic=diagnostic,
        ),
        parameters={block_id: params for block_id in block_ids},
    )
    return save_router_checkpoint(
        checkpoint,
        tmp_path / ("constant-router.json" if constant else "variable-router.json"),
    )


def test_non_diagnostic_constant_precision_cannot_support_qaq_claim(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path, "qaq_on_demand_off", router_diagnostic=True)
    refs = _refs(tmp_path, base)
    checkpoint = _checkpoint(tmp_path, base, constant=True)
    config = _config(
        tmp_path,
        "qaq_on_demand_off",
        router_checkpoint=str(checkpoint),
    )
    output = run_adaptive_runtime(config, artifact_refs=refs)

    with pytest.raises(QaqRuntimeError) as exc:
        validate_adaptive_acceptance_metadata(output)

    assert exc.value.code == "constant_precision_not_adaptive"
    assert output.metadata["routing_summary"]["constant_precision_flagged"] is True


def test_diagnostic_constant_precision_is_flagged_but_not_runtime_rejected(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path, "qaq_on_demand_off", router_diagnostic=True)
    refs = _refs(tmp_path, base)
    checkpoint = _checkpoint(tmp_path, base, constant=True, diagnostic=True)
    config = _config(
        tmp_path,
        "qaq_on_demand_off",
        router_checkpoint=str(checkpoint),
        router_diagnostic=True,
    )
    output = run_adaptive_runtime(config, artifact_refs=refs)

    validate_adaptive_acceptance_metadata(output)

    assert output.metadata["routing_summary"]["constant_global_precision"] is True
    assert output.metadata["routing_summary"]["constant_precision_flagged"] is False
    assert output.metadata["routing_summary"]["diagnostic"] is True


def test_on_demand_acceptance_guard_requires_loader_summary(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path, "qaq_on_demand_on", router_diagnostic=True)
    refs = _refs(tmp_path, base)
    checkpoint = _checkpoint(tmp_path, base)
    config = _config(
        tmp_path,
        "qaq_on_demand_on",
        router_checkpoint=str(checkpoint),
    )
    output = run_adaptive_runtime(config, artifact_refs=refs)
    metadata = dict(output.metadata)
    metadata["loader_summary"] = None

    with pytest.raises(QaqRuntimeError) as exc:
        validate_adaptive_acceptance_metadata(replace(output, metadata=metadata))

    assert exc.value.code == "missing_loader_summary"


def test_cuda_on_demand_runtime_never_silently_uses_cpu_loader(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path, "qaq_on_demand_on", router_diagnostic=True)
    refs = _refs(tmp_path, base)
    checkpoint = _checkpoint(tmp_path, base)
    config = _config(
        tmp_path,
        "qaq_on_demand_on",
        device="cuda",
        gpu_ids=[0],
        router_checkpoint=str(checkpoint),
    )

    if not _cuda_available():
        with pytest.raises(QaqRuntimeError) as exc:
            run_adaptive_runtime(config, artifact_refs=refs)
        assert exc.value.code == "cuda_unavailable"
        return

    output = run_adaptive_runtime(config, artifact_refs=refs)
    loader_summary = output.metadata["loader_summary"]

    assert output.status == "completed"
    assert output.metadata["runtime_impl"] == "qaq.runtime.adaptive.cuda_loader"
    assert loader_summary["loads"] > 0
    assert loader_summary["target_devices"] == ["cuda:0"]
    assert output.memory_events[0].measurement_source == "torch_cuda_max_memory_allocated"
    assert output.memory_events[0].peak_gpu_memory_gb > 0


def test_adaptive_runtime_rejects_checkpoint_precision_mismatch(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path, "qaq_on_demand_off", router_diagnostic=True)
    refs = _refs(tmp_path, base)
    adapter = load_model_adapter(base)
    blocks = discover_mha_ffn_blocks(
        adapter.architecture_metadata,
        supported_bit_widths=base.precision_candidates,
    )
    block_ids = tuple(block.block_id for block in blocks)
    checkpoint = RouterCheckpoint(
        metadata=RouterCheckpointMetadata(
            checkpoint_id="candidate-mismatch",
            model_id=base.model,
            block_ids=block_ids,
            candidate_bit_widths=(4,),
            feature_source=adapter.feature_source,
            hidden_size=adapter.architecture_metadata.hidden_size,
            temperature=1.0,
            decision_policy=DEFAULT_DECISION_POLICY,
            max_bit_width=base.max_bit_width,
        ),
        parameters={
            block_id: RouterBlockParameters(
                weights=((0.0, 0.0, 0.0, 0.0),),
                bias=(0.0,),
            )
            for block_id in block_ids
        },
    )
    path = save_router_checkpoint(checkpoint, tmp_path / "candidate-mismatch.json")
    config = _config(
        tmp_path,
        "qaq_on_demand_off",
        router_checkpoint=str(path),
    )

    with pytest.raises(QaqRuntimeError) as exc:
        run_adaptive_runtime(config, artifact_refs=refs)

    assert exc.value.code == "router_candidate_mismatch"


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)


def test_result_reporter_rejects_qaq_without_static_baselines(tmp_path: Path) -> None:
    base = _config(tmp_path, "qaq_on_demand_off", router_diagnostic=True)
    refs = _refs(tmp_path, base)
    checkpoint = _checkpoint(tmp_path, base)
    off_config = _config(
        tmp_path,
        "qaq_on_demand_off",
        router_checkpoint=str(checkpoint),
    )
    on_config = _config(
        tmp_path,
        "qaq_on_demand_on",
        router_checkpoint=str(checkpoint),
    )
    artifacts = (
        build_result_artifact(
            off_config,
            run_adaptive_runtime(off_config, artifact_refs=refs),
        ),
        build_result_artifact(
            on_config,
            run_adaptive_runtime(on_config, artifact_refs=refs),
        ),
    )

    validation = validate_comparison(artifacts)

    assert validation.state == "invalid"
    assert "missing_required_modes:fp16,static_8bit,static_4bit" in validation.reasons


def test_result_reporter_rejects_on_demand_without_loader_summary(
    tmp_path: Path,
) -> None:
    artifacts = list(_reporter_matrix(tmp_path))
    on_index = next(
        index
        for index, artifact in enumerate(artifacts)
        if artifact.mode == "qaq_on_demand_on"
    )
    artifacts[on_index] = replace(artifacts[on_index], loader_summary=None)

    validation = validate_comparison(tuple(artifacts))

    assert validation.state == "invalid"
    assert "missing_loader_summary:qaq_on_demand_on" in validation.reasons


def test_result_reporter_marks_fake_cpu_matrix_diagnostic(tmp_path: Path) -> None:
    validation = validate_comparison(_reporter_matrix(tmp_path))

    assert validation.state == "diagnostic"
    assert "diagnostic_or_constrained_results" in validation.reasons


def _reporter_matrix(tmp_path: Path):
    base = _config(tmp_path, "fp16", router_diagnostic=True)
    refs = _refs(tmp_path, base)
    checkpoint = _checkpoint(tmp_path, base)
    outputs = []
    for mode in ("fp16", "static_8bit", "static_4bit"):
        config = _config(tmp_path, mode)
        runtime_output = run_static_runtime(
            config,
            artifact_refs=refs if mode != "fp16" else None,
        )
        outputs.append(build_result_artifact(config, runtime_output))
    for mode in ("qaq_on_demand_off", "qaq_on_demand_on"):
        config = _config(tmp_path, mode, router_checkpoint=str(checkpoint))
        runtime_output = run_adaptive_runtime(config, artifact_refs=refs)
        outputs.append(build_result_artifact(config, runtime_output))
    return tuple(outputs)
