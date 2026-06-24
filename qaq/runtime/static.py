"""Static and fixed mixed-precision runtime paths for fake QAQ smoke runs."""

from __future__ import annotations

import builtins
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
from qaq.runtime.weight_overrides import (
    adapter_supports_weight_overrides,
    artifact_paths_for_block,
    artifact_ref_mode,
    build_weight_overrides,
    combine_reference_outputs,
    runtime_can_apply_weight_overrides,
)
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
    """Run FP16/static/fixed inference with streamed evaluation batches."""

    if config.mode not in STATIC_RUNTIME_MODES:
        raise RuntimeError(
            "unsupported_runtime_mode",
            f"static runtime does not support mode {config.mode}",
        )

    start = perf_counter()
    _reset_cuda_peak_memory_if_available(config)
    adapter = load_model_adapter(config)
    all_examples = adapter.load_examples(config)
    examples, total_examples, subset_run = _select_examples(
        all_examples,
        config=config,
        example_limit=example_limit,
    )
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
    mixed_weight_forward_applied = False
    mixed_weight_forward_reason = "fp16_reference" if not require_artifacts else "not_attempted"
    weight_override_records: tuple[dict[str, Any], ...] = ()
    weight_overrides: Mapping[str, Any] | None = None
    block_ids = tuple(block.block_id for block in blocks)
    precision_label = "fp16_reference"
    if require_artifacts:
        can_apply, reason = runtime_can_apply_weight_overrides(adapter, blocks)
        mixed_weight_forward_reason = reason
        if can_apply:
            weight_overrides, weight_override_records = build_weight_overrides(
                config,
                blocks=blocks,
                plan=plan,
            )
            mixed_weight_forward_applied = True
            precision_label = f"{config.mode}_bitplane_weight_overrides"
        else:
            precision_label = f"{config.mode}_reference_without_weight_overrides"

    raw_outputs = []
    for chunk in _chunk_examples(examples, config.eval_batch_size):
        batch = adapter.build_batch(config, chunk)
        raw_outputs.append(
            adapter.reference_forward(
                batch,
                block_ids=block_ids,
                weight_overrides=weight_overrides,
                precision_label=precision_label,
                collect_hidden_states=config.collect_hidden_states,
                store_full_logits=config.store_full_logits,
            )
        )

    elapsed = perf_counter() - start
    peak_gpu_memory_gb = _peak_gpu_memory_gb(config)
    raw_output = combine_reference_outputs(
        tuple(raw_outputs),
        precision_label=precision_label,
        metadata_updates={
            "eval_batch_size": config.eval_batch_size,
            "processed_examples": len(examples),
            "total_examples": total_examples,
            "max_examples": config.max_examples,
            "subset_run": subset_run,
            "micro_batch_count": len(raw_outputs),
            "collect_hidden_states": config.collect_hidden_states,
            "full_logits_stored": config.store_full_logits,
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
        },
    )
    model_device_map = raw_output.metadata.get("model_device_map")
    log_event = LogEvent.progress(
        run_id=run_id,
        module="static_runtime",
        message=f"{config.mode} runtime completed",
        mode=config.mode,
        benchmark=config.dataset,
        processed_examples=len(examples),
        total_examples=total_examples,
        elapsed_seconds=elapsed,
        latency_seconds=elapsed,
        peak_gpu_memory_gb=peak_gpu_memory_gb,
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
                cache_policy="streamed_micro_batches",
            ),
        ),
        memory_events=(
            MemoryEvent(
                name="peak_gpu_memory",
                peak_gpu_memory_gb=peak_gpu_memory_gb,
                selected_gpu_ids=config.gpu_ids,
                measurement_source=_memory_measurement_source(config),
                details={
                    "device": config.device,
                    "reconstructed_blocks": len(reconstruction_records),
                    "processed_examples": len(examples),
                    "eval_batch_size": config.eval_batch_size,
                    "micro_batch_count": len(raw_outputs),
                    "model_device_map": model_device_map,
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
            "artifact_ref_mode": artifact_ref_mode(blocks) if require_artifacts else "none",
            "mixed_precision_forward_applied": mixed_weight_forward_applied,
            "mixed_precision_forward_reason": mixed_weight_forward_reason,
            "weight_override_tensor_count": len(weight_override_records),
            "weight_override_records": list(weight_override_records),
            "runtime_impl": _static_runtime_impl(
                config,
                adapter=adapter,
                mixed_weight_forward_applied=mixed_weight_forward_applied,
            ),
            "eval_batch_size": config.eval_batch_size,
            "processed_examples": len(examples),
            "total_examples": total_examples,
            "max_examples": config.max_examples,
            "subset_run": subset_run,
            "micro_batch_count": len(raw_outputs),
            "collect_hidden_states": config.collect_hidden_states,
            "store_full_logits": config.store_full_logits,
            "hf_device_map": config.hf_device_map or "single",
            "hf_max_memory_per_gpu": config.hf_max_memory_per_gpu,
            "model_device_map": model_device_map,
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
        },
        log_events=(log_event.as_dict(),),
    )

def load_artifact_index(path: str | Path) -> dict[str, dict[str, str]]:
    """Load a block artifact index from JSON.

    Supported shapes are legacy ``block -> bit-width -> artifact path`` and
    tensor-native ``block -> tensor_name -> artifact path``.
    """

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
        for ref_key, artifact_path in refs.items():
            normalized_key = str(ref_key)
            if not normalized_key or (not normalized_key.isdigit() and "." not in normalized_key):
                raise RuntimeError(
                    "invalid_artifact_index",
                    f"artifact ref key {ref_key!r} for {block_id} is invalid",
                )
            if not isinstance(artifact_path, str) or not artifact_path:
                raise RuntimeError(
                    "invalid_artifact_index",
                    f"artifact path for {block_id}/{ref_key} must be non-empty",
                )
            result[block_id][normalized_key] = artifact_path
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
        for ref in artifact_paths_for_block(block, bit_width):
            artifact_path = ref.artifact_path
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
                    tensor_name=ref.tensor_name,
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
                tensor_name=ref.tensor_name,
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


def _select_examples(
    examples: tuple[Any, ...],
    *,
    config: RunConfig,
    example_limit: int | None,
) -> tuple[tuple[Any, ...], int, bool]:
    limits = tuple(
        limit for limit in (config.max_examples, example_limit) if limit is not None
    )
    limit = min(limits) if limits else None
    selected = examples[:limit] if limit is not None else examples
    return selected, len(examples), bool(limits or len(selected) < len(examples))


def _chunk_examples(
    examples: tuple[Any, ...],
    batch_size: int,
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        examples[start : start + batch_size]
        for start in range(0, len(examples), batch_size)
    )


def _reset_cuda_peak_memory_if_available(config: RunConfig) -> None:
    if config.device != "cuda":
        return
    torch = _try_import_torch()
    if torch is None or not torch.cuda.is_available():
        return
    for index in _cuda_memory_indices(torch, config):
        _safe_reset_peak_memory_stats(torch, index)


def _peak_gpu_memory_gb(config: RunConfig) -> float:
    if config.device != "cuda":
        return 0.0
    torch = _try_import_torch()
    if torch is None or not torch.cuda.is_available():
        return 0.0
    indices = _cuda_memory_indices(torch, config)
    if not indices:
        return 0.0
    peak_bytes = max(_safe_max_memory_allocated(torch, index) for index in indices)
    return peak_bytes / float(1024**3)


def _cuda_memory_indices(torch: Any, config: RunConfig) -> tuple[int, ...]:
    device_count = int(torch.cuda.device_count())
    if device_count <= 0:
        return ()
    if config.hf_device_map == "auto":
        return tuple(range(device_count))
    selected = tuple(gpu_id for gpu_id in config.gpu_ids if 0 <= gpu_id < device_count)
    return selected or (0,)


def _memory_measurement_source(config: RunConfig) -> str:
    if config.device == "cuda":
        return "torch_cuda_max_memory_allocated"
    return "cpu_fake_no_cuda"


def _try_import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _safe_reset_peak_memory_stats(torch: Any, index: int) -> None:
    try:
        torch.cuda.reset_peak_memory_stats(index)
    except builtins.RuntimeError as exc:
        if "invalid device argument" not in str(exc).lower():
            raise


def _safe_max_memory_allocated(torch: Any, index: int) -> int:
    try:
        return int(torch.cuda.max_memory_allocated(index))
    except builtins.RuntimeError as exc:
        if "invalid device argument" not in str(exc).lower():
            raise
        return 0


def _static_runtime_impl(
    config: RunConfig,
    *,
    adapter: Any,
    mixed_weight_forward_applied: bool,
) -> str:
    if mixed_weight_forward_applied:
        return "qaq.runtime.static.hf_bitplane_weight_overrides"
    if config.mode == "fp16" and adapter_supports_weight_overrides(adapter):
        return "qaq.runtime.static.hf_reference"
    return "qaq.runtime.static.fake_cpu"
