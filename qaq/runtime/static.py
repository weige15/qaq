"""Static and fixed mixed-precision runtime paths for fake QAQ smoke runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from qaq.artifacts import load_bitplane_artifact
from qaq.bitplanes import reconstruct_weight
from qaq.blocks import BlockDescriptor, block_map, discover_mha_ffn_blocks
from qaq.config import RunConfig
from qaq.logging import LogEvent
from qaq.model_adapter import load_model_adapter
from qaq.precision_plan import PrecisionPlan, build_precision_plan
from qaq.runtime.common import LatencyEvent, MemoryEvent, RuntimeError, RuntimeOutputBundle
from qaq.tensor_bitplanes import (
    is_tensor_bitplane_artifact_path,
    load_tensor_bitplane_artifact,
    reconstruct_tensor_weight,
)


STATIC_RUNTIME_MODES = frozenset({"fp16", "static_8bit", "static_4bit", "fixed_mixed"})
REQUIRED_BASELINE_MODES = frozenset({"fp16", "static_8bit", "static_4bit"})


def run_static_runtime(
    config: RunConfig,
    *,
    artifact_refs: Mapping[str, Mapping[str | int, str | Path]] | None = None,
    run_id: str = "static-runtime-smoke",
    example_limit: int | None = None,
) -> RuntimeOutputBundle:
    """Run FP16/static/fixed fake inference and return result-ready metadata."""

    if config.mode not in STATIC_RUNTIME_MODES:
        raise RuntimeError(
            "unsupported_runtime_mode",
            f"static runtime does not support mode {config.mode}",
        )

    start = perf_counter()
    adapter = load_model_adapter(config)
    examples = adapter.load_examples(config, limit=example_limit)
    batch = adapter.build_batch(config, examples)
    blocks = discover_mha_ffn_blocks(
        adapter.architecture_metadata,
        supported_bit_widths=config.precision_candidates,
    )
    blocks = _attach_artifact_refs(blocks, artifact_refs or {})
    require_artifacts = config.mode != "fp16"
    plan = build_precision_plan(
        blocks,
        mode=config.mode,
        precision_candidates=config.precision_candidates,
        max_bit_width=config.max_bit_width,
        fixed_precision_by_block=config.fixed_precision_by_block,
        require_artifacts=require_artifacts,
    )
    reconstruction_records = (
        _materialize_plan_artifacts(config, blocks, plan) if require_artifacts else ()
    )
    raw_output = adapter.reference_forward(
        batch,
        block_ids=tuple(block.block_id for block in blocks),
    )
    elapsed = perf_counter() - start
    log_event = LogEvent.progress(
        run_id=run_id,
        module="static_runtime",
        message=f"{config.mode} runtime completed",
        mode=config.mode,
        benchmark=config.dataset,
        processed_examples=len(examples),
        total_examples=len(examples),
        elapsed_seconds=elapsed,
        latency_seconds=elapsed,
        peak_gpu_memory_gb=0.0,
        selected_gpu_ids=config.gpu_ids,
    )
    return RuntimeOutputBundle(
        mode=config.mode,
        status="completed",
        raw_output=raw_output,
        precision_plan=plan,
        latency_events=(
            LatencyEvent(
                name="end_to_end",
                elapsed_seconds=elapsed,
                warmup_steps=0,
                cache_policy="none_cpu_fake",
            ),
        ),
        memory_events=(
            MemoryEvent(
                name="peak_gpu_memory",
                peak_gpu_memory_gb=0.0,
                selected_gpu_ids=config.gpu_ids,
                measurement_source="cpu_fake_no_cuda",
                details={
                    "device": config.device,
                    "reconstructed_blocks": len(reconstruction_records),
                },
            ),
        ),
        reconstruction_records=reconstruction_records,
        metadata={
            "model": config.model,
            "tokenizer": config.tokenizer,
            "dataset": config.dataset,
            "split": config.split,
            "prompt_format": config.prompt_format or "plain",
            "metric": config.metric,
            "precision_candidates": list(config.precision_candidates),
            "block_granularity": config.block_granularity,
            "seed": config.seed,
            "selected_gpu_ids": list(config.gpu_ids),
            "fixed_mixed_is_diagnostic": config.mode == "fixed_mixed",
            "runtime_impl": "qaq.runtime.static.fake_cpu",
        },
        log_events=(log_event.as_dict(),),
    )


def load_artifact_index(path: str | Path) -> dict[str, dict[str, str]]:
    """Load a block -> bit-width -> artifact path mapping from JSON."""

    index_path = Path(path)
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError("artifact_index_read_failed", str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("artifact_index_parse_failed", str(exc)) from exc
    if not isinstance(raw, dict):
        raise RuntimeError("invalid_artifact_index", "artifact index must be an object")

    result: dict[str, dict[str, str]] = {}
    for block_id, refs in raw.items():
        if not isinstance(block_id, str) or not block_id:
            raise RuntimeError(
                "invalid_artifact_index",
                "artifact index block IDs must be non-empty strings",
            )
        if not isinstance(refs, dict) or not refs:
            raise RuntimeError(
                "invalid_artifact_index",
                f"artifact refs for {block_id} must be a non-empty object",
            )
        result[block_id] = {}
        for bit_width, artifact_path in refs.items():
            if not str(bit_width).isdigit():
                raise RuntimeError(
                    "invalid_artifact_index",
                    f"artifact bit-width {bit_width!r} for {block_id} is invalid",
                )
            if not isinstance(artifact_path, str) or not artifact_path:
                raise RuntimeError(
                    "invalid_artifact_index",
                    f"artifact path for {block_id}/{bit_width} must be non-empty",
                )
            result[block_id][str(bit_width)] = artifact_path
    return result


def validate_required_static_baselines(completed_modes: set[str] | frozenset[str]) -> None:
    """Reject QAQ comparison acceptance when required static baselines are absent."""

    missing = sorted(REQUIRED_BASELINE_MODES - set(completed_modes))
    if missing:
        raise RuntimeError(
            "missing_static_baseline",
            f"QAQ comparison acceptance requires baselines: {missing}",
        )


def _attach_artifact_refs(
    blocks: tuple[BlockDescriptor, ...],
    artifact_refs: Mapping[str, Mapping[str | int, str | Path]],
) -> tuple[BlockDescriptor, ...]:
    if not artifact_refs:
        return blocks

    known_blocks = block_map(blocks)
    unknown = sorted(set(artifact_refs) - set(known_blocks))
    if unknown:
        raise RuntimeError(
            "unknown_artifact_block",
            f"artifact refs include unknown blocks: {unknown}",
        )

    updated: list[BlockDescriptor] = []
    for block in blocks:
        refs = artifact_refs.get(block.block_id)
        if refs is None:
            updated.append(block)
            continue
        normalized = {str(bit_width): str(path) for bit_width, path in refs.items()}
        updated.append(replace(block, artifact_refs=normalized))
    return tuple(updated)


def _materialize_plan_artifacts(
    config: RunConfig,
    blocks: tuple[BlockDescriptor, ...],
    plan: PrecisionPlan,
) -> tuple[dict[str, Any], ...]:
    descriptors = block_map(blocks)
    records: list[dict[str, Any]] = []
    for block_id, bit_width in plan.decisions.items():
        block = descriptors[block_id]
        artifact_path = block.artifact_refs[str(bit_width)]
        if is_tensor_bitplane_artifact_path(artifact_path):
            artifact = load_tensor_bitplane_artifact(artifact_path)
            if artifact.metadata.tensor_name not in block.tensor_names:
                raise RuntimeError(
                    "artifact_tensor_mismatch",
                    f"{block_id} artifact tensor {artifact.metadata.tensor_name} is not owned by the block",
                )
            reconstructed = reconstruct_tensor_weight(
                artifact,
                bit_width=bit_width,
                model_id=config.model,
                block_id=block_id,
            )
            records.append(
                {
                    "block_id": block_id,
                    "bit_width": bit_width,
                    "artifact_path": str(artifact_path),
                    "tensor_name": artifact.metadata.tensor_name,
                    "selected_planes": list(reconstructed.selected_planes),
                    "shape": list(artifact.metadata.original_shape),
                    "quantization_scheme": artifact.metadata.quantization.scheme,
                    "checksum": artifact.metadata.checksum,
                    "storage_layout": "packed_uint8_bitplanes",
                }
            )
            continue
        artifact = load_bitplane_artifact(artifact_path)
        if artifact.metadata.tensor_name not in block.tensor_names:
            raise RuntimeError(
                "artifact_tensor_mismatch",
                f"{block_id} artifact tensor {artifact.metadata.tensor_name} is not owned by the block",
            )
        reconstructed = reconstruct_weight(
            artifact,
            bit_width=bit_width,
            model_id=config.model,
            block_id=block_id,
        )
        records.append(
            {
                "block_id": block_id,
                "bit_width": bit_width,
                "artifact_path": str(artifact_path),
                "tensor_name": artifact.metadata.tensor_name,
                "selected_planes": list(reconstructed.selected_planes),
                "shape": list(artifact.metadata.original_shape),
                "quantization_scheme": artifact.metadata.quantization.scheme,
                "checksum": artifact.metadata.checksum,
            }
        )
    return tuple(records)
