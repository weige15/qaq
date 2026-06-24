"""Adaptive QAQ runtime driven by a trained router checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from qaq.artifacts import load_bitplane_artifact
from qaq.bitplanes import BitPlaneError, reconstruct_weight, selected_msb_planes
from qaq.blocks import BlockDescriptor, block_map, discover_mha_ffn_blocks
from qaq.config import QAQ_MODES, RunConfig
from qaq.loader import LoaderError, LoaderRequest, OnDemandLoader
from qaq.logging import LogEvent
from qaq.model_adapter import HiddenStateBundle, ReferenceBatchOutput, load_model_adapter
from qaq.precision_plan import PrecisionPlan
from qaq.router.checkpoint import load_router_checkpoint
from qaq.router.policy import route_hidden_states, summarize_traces
from qaq.router.types import RouterPolicyError
from qaq.runtime.common import LatencyEvent, MemoryEvent, RuntimeError, RuntimeOutputBundle
from qaq.runtime.weight_overrides import (
    artifact_paths_for_block,
    artifact_ref_mode,
    build_weight_overrides,
    combine_reference_outputs,
    runtime_can_apply_weight_overrides,
    slice_batch,
)
from qaq.tensor_bitplanes import (
    TensorBitPlaneError,
    is_tensor_bitplane_artifact_path,
    load_tensor_bitplane_artifact,
    reconstruct_tensor_weight,
)


def run_adaptive_runtime(
    config: RunConfig,
    *,
    artifact_refs: Mapping[str, Mapping[str | int, str | Path]] | None = None,
    run_id: str = "adaptive-runtime",
    example_limit: int | None = None,
) -> RuntimeOutputBundle:
    """Run checkpoint-loaded adaptive routing over local benchmark examples."""

    if config.mode not in QAQ_MODES:
        raise RuntimeError(
            "unsupported_runtime_mode",
            f"adaptive runtime does not support mode {config.mode}",
        )
    if config.router_checkpoint is None:
        raise RuntimeError(
            "missing_router_checkpoint",
            "adaptive runtime requires router_checkpoint",
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
    checkpoint = load_router_checkpoint(config.router_checkpoint)
    _validate_checkpoint_runtime_metadata(config, checkpoint)
    block_ids = tuple(block.block_id for block in blocks)

    compact_routing_outputs: list[ReferenceBatchOutput] = []
    query_batches = []
    all_plans = []
    all_traces = []
    for chunk in _chunk_examples(examples, config.eval_batch_size):
        batch = adapter.build_batch(config, chunk)
        routing_output = adapter.reference_forward(
            batch,
            block_ids=block_ids,
            collect_hidden_states=True,
            store_full_logits=config.store_full_logits,
        )
        try:
            routing = route_hidden_states(
                checkpoint,
                hidden_states=routing_output.hidden_states,
                blocks=blocks,
                model_id=config.model,
                mode=config.mode,
                query_ids=batch.example_ids,
                diagnostic=config.router_diagnostic,
            )
        except RouterPolicyError as exc:
            raise RuntimeError(exc.code, exc.message) from exc
        all_plans.extend(routing.plans)
        all_traces.extend(routing.traces)
        compact_routing_outputs.append(_drop_hidden_states(routing_output))
        for query_index in range(len(routing.plans)):
            query_batches.append(slice_batch(batch, query_index))

    plans = tuple(all_plans)
    traces = tuple(all_traces)
    routing_summary = summarize_traces(
        traces,
        diagnostic=config.router_diagnostic,
    )
    _validate_selected_artifacts(config, blocks=blocks, plans=plans)
    reconstruction_records, loader_summary, loader_events = _materialize_adaptive_plans(
        config,
        blocks=blocks,
        plans=plans,
    )
    mixed_weight_forward_applied = False
    mixed_weight_forward_reason = "not_attempted"
    weight_override_records: tuple[dict[str, Any], ...] = ()
    raw_output = combine_reference_outputs(
        tuple(compact_routing_outputs),
        precision_label="qaq_routing_reference",
        metadata_updates={
            "eval_batch_size": config.eval_batch_size,
            "processed_examples": len(examples),
            "total_examples": total_examples,
            "max_examples": config.max_examples,
            "subset_run": subset_run,
            "micro_batch_count": len(compact_routing_outputs),
            "collect_hidden_states": False,
            "full_logits_stored": config.store_full_logits,
        },
    )
    can_apply, reason = runtime_can_apply_weight_overrides(adapter, blocks)
    mixed_weight_forward_reason = reason
    if can_apply:
        raw_output, weight_override_records = _run_adaptive_weight_overridden_forward(
            config,
            adapter=adapter,
            query_batches=tuple(query_batches),
            blocks=blocks,
            block_ids=block_ids,
            plans=plans,
        )
        mixed_weight_forward_applied = True
    elapsed = perf_counter() - start
    peak_gpu_memory_gb = _peak_gpu_memory_gb(config)
    memory_source = _memory_measurement_source(config)
    first_plan = plans[0]
    adaptive_traces = _build_adaptive_traces(
        plans=plans,
        routing_traces=traces,
        materialization_records=reconstruction_records,
        elapsed_seconds=elapsed,
        peak_gpu_memory_gb=peak_gpu_memory_gb,
        memory_measurement_source=memory_source,
    )
    raw_output.metadata.update(
        {
            "eval_batch_size": config.eval_batch_size,
            "processed_examples": len(examples),
            "total_examples": total_examples,
            "max_examples": config.max_examples,
            "subset_run": subset_run,
            "micro_batch_count": len(compact_routing_outputs),
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
        }
    )
    log_event = LogEvent.progress(
        run_id=run_id,
        module="adaptive_runtime",
        message=f"{config.mode} runtime completed",
        mode=config.mode,
        benchmark=config.dataset,
        processed_examples=len(examples),
        total_examples=total_examples,
        elapsed_seconds=elapsed,
        latency_seconds=elapsed,
        peak_gpu_memory_gb=peak_gpu_memory_gb,
        selected_gpu_ids=config.gpu_ids,
        details={
            "router_checkpoint": str(config.router_checkpoint),
            "routing_summary": routing_summary.as_dict(),
            "loader_summary": loader_summary,
            "adaptive_trace_count": len(adaptive_traces),
        },
    )
    metadata: dict[str, Any] = {
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
        "router_checkpoint": str(config.router_checkpoint),
        "router_checkpoint_id": checkpoint.metadata.checkpoint_id,
        "router_checkpoint_loaded": True,
        "routing_summary": routing_summary.as_dict(),
        "routing_traces": [trace.as_dict() for trace in traces],
        "precision_plans": [plan.as_dict() for plan in plans],
        "adaptive_traces": adaptive_traces,
        "loader_summary": loader_summary,
        "loader_events": loader_events,
        "artifact_ref_mode": artifact_ref_mode(blocks),
        "mixed_precision_forward_applied": mixed_weight_forward_applied,
        "mixed_precision_forward_reason": mixed_weight_forward_reason,
        "weight_override_tensor_count": len(weight_override_records),
        "weight_override_records": list(weight_override_records),
        "runtime_impl": _runtime_impl(
            config,
            mixed_weight_forward_applied=mixed_weight_forward_applied,
        ),
        "eval_batch_size": config.eval_batch_size,
        "processed_examples": len(examples),
        "total_examples": total_examples,
        "max_examples": config.max_examples,
        "subset_run": subset_run,
        "micro_batch_count": len(compact_routing_outputs),
        "collect_hidden_states": True,
        "store_full_logits": config.store_full_logits,
        "hf_device_map": config.hf_device_map or "single",
        "hf_max_memory_per_gpu": config.hf_max_memory_per_gpu,
        "model_device_map": raw_output.metadata.get("model_device_map"),
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
    }
    return RuntimeOutputBundle(
        mode=config.mode,
        status="completed",
        raw_output=raw_output,
        precision_plan=first_plan,
        latency_events=(
            LatencyEvent(
                name="end_to_end",
                elapsed_seconds=elapsed,
                warmup_steps=0,
                cache_policy=(
                    "cpu_on_demand_loader" if config.mode == "qaq_on_demand_on" else "gpu_resident_simulated"
                ),
            ),
        ),
        memory_events=(
            MemoryEvent(
                name="peak_gpu_memory",
                peak_gpu_memory_gb=peak_gpu_memory_gb,
                selected_gpu_ids=config.gpu_ids,
                measurement_source=memory_source,
                details={
                    "device": config.device,
                    "routed_queries": len(plans),
                    "reconstructed_blocks": len(reconstruction_records),
                    "processed_examples": len(examples),
                    "eval_batch_size": config.eval_batch_size,
                    "micro_batch_count": len(compact_routing_outputs),
                    "model_device_map": raw_output.metadata.get("model_device_map"),
                },
            ),
        ),
        reconstruction_records=reconstruction_records,
        metadata=metadata,
        log_events=(log_event.as_dict(),),
    )

def validate_adaptive_acceptance_metadata(output: RuntimeOutputBundle) -> None:
    """Validate adaptive runtime evidence before any QAQ acceptance claim."""

    if output.mode not in QAQ_MODES:
        raise RuntimeError(
            "unsupported_acceptance_mode",
            f"adaptive acceptance metadata does not apply to mode {output.mode}",
        )

    routing_summary = output.metadata.get("routing_summary")
    if not isinstance(routing_summary, dict):
        raise RuntimeError(
            "missing_routing_summary",
            "QAQ adaptive results require a routing summary",
        )
    if (
        routing_summary.get("constant_precision_flagged") is True
        or (
            routing_summary.get("constant_global_precision") is True
            and routing_summary.get("diagnostic") is not True
        )
    ):
        raise RuntimeError(
            "constant_precision_not_adaptive",
            "constant global precision cannot support a non-diagnostic QAQ claim",
        )

    adaptive_traces = output.metadata.get("adaptive_traces")
    if not isinstance(adaptive_traces, list | tuple) or not adaptive_traces:
        raise RuntimeError(
            "missing_adaptive_trace",
            "QAQ adaptive results require per-query adaptive traces",
        )

    if output.mode == "qaq_on_demand_on":
        loader_summary = output.metadata.get("loader_summary")
        if not isinstance(loader_summary, dict):
            raise RuntimeError(
                "missing_loader_summary",
                "qaq_on_demand_on results require a loader summary",
            )
        if int(loader_summary.get("loads", 0)) <= 0 and int(loader_summary.get("cache_hits", 0)) <= 0:
            raise RuntimeError(
                "missing_loader_activity",
                "qaq_on_demand_on results require loader activity",
            )


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


def _drop_hidden_states(output: ReferenceBatchOutput) -> ReferenceBatchOutput:
    return ReferenceBatchOutput(
        logits=output.logits,
        losses=output.losses,
        predictions=output.predictions,
        hidden_states=HiddenStateBundle(
            feature_source=output.hidden_states.feature_source,
            by_block={},
        ),
        metadata={
            **output.metadata,
            "collect_hidden_states": False,
            "hidden_state_block_count": 0,
        },
    )


def _attach_artifact_refs(
    blocks: tuple[BlockDescriptor, ...],
    artifact_refs: Mapping[str, Mapping[str | int, str | Path]],
) -> tuple[BlockDescriptor, ...]:
    if not artifact_refs:
        raise RuntimeError(
            "missing_artifact_index",
            "adaptive runtime requires artifact refs for selected precision materialization",
        )

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
            raise RuntimeError(
                "missing_artifact",
                f"{block.block_id} is missing artifact refs",
            )
        normalized = {str(bit_width): str(path) for bit_width, path in refs.items()}
        updated.append(replace(block, artifact_refs=normalized))
    return tuple(updated)


def _validate_checkpoint_runtime_metadata(config: RunConfig, checkpoint: Any) -> None:
    metadata = checkpoint.metadata
    if metadata.candidate_bit_widths != config.precision_candidates:
        raise RuntimeError(
            "router_candidate_mismatch",
            "router checkpoint candidate bit-widths do not match active config",
        )
    checkpoint_max_bit_width = metadata.max_bit_width or max(metadata.candidate_bit_widths)
    if checkpoint_max_bit_width != config.max_bit_width:
        raise RuntimeError(
            "router_max_bit_width_mismatch",
            "router checkpoint max_bit_width does not match active config",
        )


def _validate_selected_artifacts(
    config: RunConfig,
    *,
    blocks: tuple[BlockDescriptor, ...],
    plans: tuple[PrecisionPlan, ...],
) -> None:
    descriptors = block_map(blocks)
    for plan in plans:
        for block_id, bit_width in plan.decisions.items():
            block = descriptors[block_id]
            for ref in artifact_paths_for_block(block, bit_width):
                artifact_path = ref.artifact_path
                try:
                    if is_tensor_bitplane_artifact_path(artifact_path):
                        artifact = load_tensor_bitplane_artifact(artifact_path)
                        reconstructed = reconstruct_tensor_weight(
                            artifact,
                            bit_width=bit_width,
                            model_id=config.model,
                            block_id=block_id,
                            tensor_name=ref.tensor_name,
                        )
                        selected_planes = reconstructed.selected_planes
                    else:
                        artifact = load_bitplane_artifact(artifact_path)
                        selected_planes = selected_msb_planes(
                            bit_width,
                            max_bit_width=artifact.metadata.max_bit_width,
                        )
                except BitPlaneError as exc:
                    raise RuntimeError(exc.code, exc.message) from exc
                except TensorBitPlaneError as exc:
                    raise RuntimeError(exc.code, exc.message) from exc

                metadata = artifact.metadata
                if metadata.model_id != config.model:
                    raise RuntimeError(
                        "artifact_model_mismatch",
                        f"{block_id} artifact model {metadata.model_id} does not match {config.model}",
                    )
                if metadata.block_id != block_id:
                    raise RuntimeError(
                        "artifact_block_mismatch",
                        f"{block_id} artifact block metadata is {metadata.block_id}",
                    )
                if metadata.tensor_name not in block.tensor_names:
                    raise RuntimeError(
                        "artifact_tensor_mismatch",
                        f"{block_id} artifact tensor {metadata.tensor_name} is not owned by the block",
                    )
                if ref.tensor_name is not None and metadata.tensor_name != ref.tensor_name:
                    raise RuntimeError(
                        "artifact_tensor_mismatch",
                        f"{block_id} artifact index key {ref.tensor_name} points to {metadata.tensor_name}",
                    )
                if metadata.max_bit_width != config.max_bit_width:
                    raise RuntimeError(
                        "artifact_max_bit_width_mismatch",
                        f"{block_id} artifact max_bit_width {metadata.max_bit_width} does not match {config.max_bit_width}",
                    )
                artifact_granularity = (metadata.compatibility or {}).get("block_granularity")
                if artifact_granularity and artifact_granularity != config.block_granularity:
                    raise RuntimeError(
                        "artifact_granularity_mismatch",
                        f"{block_id} artifact granularity {artifact_granularity} does not match {config.block_granularity}",
                    )
                missing_planes = [
                    plane_index
                    for plane_index in selected_planes
                    if plane_index not in metadata.available_planes
                ]
                if missing_planes:
                    raise RuntimeError(
                        "missing_plane",
                        f"{block_id} artifact is missing selected planes {missing_planes}",
                    )


def _materialize_adaptive_plans(
    config: RunConfig,
    *,
    blocks: tuple[BlockDescriptor, ...],
    plans: tuple[PrecisionPlan, ...],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any] | None, tuple[dict[str, Any], ...]]:
    descriptors = block_map(blocks)
    records: list[dict[str, Any]] = []
    if config.mode == "qaq_on_demand_on":
        loader = OnDemandLoader(known_block_ids=tuple(descriptors))
        for plan in plans:
            for block_id, bit_width in plan.decisions.items():
                block = descriptors[block_id]
                for ref in artifact_paths_for_block(block, bit_width):
                    request = LoaderRequest(
                        request_id=f"{plan.query_id}:{block_id}:{bit_width}:{ref.tensor_name or 'legacy'}",
                        query_id=plan.query_id,
                        block_id=block_id,
                        bit_width=bit_width,
                        artifact_path=ref.artifact_path,
                        target_device=_loader_target_device(config),
                        model_id=config.model,
                        tensor_name=ref.tensor_name,
                    )
                    try:
                        materialized = loader.load(request)
                    except LoaderError as exc:
                        raise RuntimeError(exc.code, exc.message) from exc
                    records.append(
                        {
                            "query_id": plan.query_id,
                            "block_id": block_id,
                            "bit_width": bit_width,
                            "artifact_path": str(ref.artifact_path),
                            "tensor_name": ref.tensor_name,
                            "selected_planes": list(materialized.selected_planes),
                            "loader_request_id": request.request_id,
                            "bytes_loaded": materialized.bytes_loaded,
                        }
                    )
        return (
            tuple(records),
            loader.summary().as_dict(),
            tuple(event.as_dict() for event in loader.events),
        )

    for plan in plans:
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
                            "query_id": plan.query_id,
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
                        "query_id": plan.query_id,
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
    return tuple(records), None, ()


def _loader_target_device(config: RunConfig) -> str:
    if config.device == "cpu":
        return "cpu"
    if config.device == "cuda":
        if not config.gpu_ids:
            raise RuntimeError(
                "invalid_device",
                "cuda on-demand loading requires at least one selected gpu_id",
            )
        return f"cuda:{config.gpu_ids[0]}"
    raise RuntimeError(
        "invalid_device",
        "adaptive on-demand loading supports only cpu or cuda devices",
    )


def _run_adaptive_weight_overridden_forward(
    config: RunConfig,
    *,
    adapter: Any,
    query_batches: tuple[Any, ...],
    blocks: tuple[BlockDescriptor, ...],
    block_ids: tuple[str, ...],
    plans: tuple[PrecisionPlan, ...],
) -> tuple[Any, tuple[dict[str, Any], ...]]:
    outputs = []
    all_override_records: list[dict[str, Any]] = []
    per_query_counts: dict[str, int] = {}
    for query_batch, plan in zip(query_batches, plans, strict=True):
        weight_overrides, override_records = build_weight_overrides(
            config,
            blocks=blocks,
            plan=plan,
        )
        query_id = plan.query_id or query_batch.example_ids[0]
        for record in override_records:
            all_override_records.append({"query_id": query_id, **record})
        per_query_counts[query_id] = len(weight_overrides)
        outputs.append(
            adapter.reference_forward(
                query_batch,
                block_ids=block_ids,
                weight_overrides=weight_overrides,
                precision_label="qaq_selected_bitplane_weight_overrides",
                collect_hidden_states=False,
                store_full_logits=config.store_full_logits,
            )
        )
    combined = combine_reference_outputs(
        tuple(outputs),
        precision_label="qaq_selected_bitplane_weight_overrides",
        metadata_updates={
            "mixed_precision_forward_applied": True,
            "execution_granularity": "per_query",
            "per_query_weight_override_counts": per_query_counts,
            "eval_batch_size": config.eval_batch_size,
            "max_examples": config.max_examples,
        },
    )
    return combined, tuple(all_override_records)

def _build_adaptive_traces(
    *,
    plans: tuple[PrecisionPlan, ...],
    routing_traces: tuple[Any, ...],
    materialization_records: tuple[dict[str, Any], ...],
    elapsed_seconds: float,
    peak_gpu_memory_gb: float,
    memory_measurement_source: str,
) -> list[dict[str, Any]]:
    trace_refs_by_query: dict[str, list[str]] = {}
    for trace in routing_traces:
        trace_refs_by_query.setdefault(trace.query_id, []).append(
            f"{trace.query_id}:{trace.block_id}"
        )

    materialized_by_query: dict[str, list[dict[str, Any]]] = {}
    loader_refs_by_query: dict[str, list[str]] = {}
    for record in materialization_records:
        query_id = str(record.get("query_id"))
        materialized_by_query.setdefault(query_id, []).append(
            {
                "block_id": record["block_id"],
                "bit_width": record["bit_width"],
                "selected_planes": list(record["selected_planes"]),
            }
        )
        loader_request_id = record.get("loader_request_id")
        if loader_request_id:
            loader_refs_by_query.setdefault(query_id, []).append(str(loader_request_id))

    per_query_latency = elapsed_seconds / len(plans) if plans else 0.0
    traces: list[dict[str, Any]] = []
    for plan in plans:
        query_id = plan.query_id or "unknown-query"
        traces.append(
            {
                "query_id": query_id,
                "runtime_status": "completed",
                "precision_plan": plan.as_dict(),
                "routing_trace_refs": trace_refs_by_query.get(query_id, []),
                "loader_request_refs": loader_refs_by_query.get(query_id, []),
                "materialized_blocks": materialized_by_query.get(query_id, []),
                "latency_seconds": per_query_latency,
                "latency_measurement_source": "batch_perf_counter_average",
                "memory": {
                    "peak_gpu_memory_gb": peak_gpu_memory_gb,
                    "measurement_source": memory_measurement_source,
                },
            }
        )
    return traces


def _runtime_impl(config: RunConfig, *, mixed_weight_forward_applied: bool) -> str:
    if mixed_weight_forward_applied:
        if config.mode == "qaq_on_demand_on":
            return "qaq.runtime.adaptive.hf_per_query_bitplane_weight_overrides_with_loader"
        return "qaq.runtime.adaptive.hf_per_query_bitplane_weight_overrides"
    if config.mode == "qaq_on_demand_on" and config.device == "cuda":
        return "qaq.runtime.adaptive.cuda_loader"
    if config.device == "cuda":
        return "qaq.runtime.adaptive.cuda_reference"
    return "qaq.runtime.adaptive.cpu"


def _reset_cuda_peak_memory_if_available(config: RunConfig) -> None:
    if config.device != "cuda":
        return
    torch = _try_import_torch()
    if torch is None or not torch.cuda.is_available():
        return
    for gpu_id in config.gpu_ids:
        torch.cuda.reset_peak_memory_stats(gpu_id)


def _peak_gpu_memory_gb(config: RunConfig) -> float:
    if config.device != "cuda":
        return 0.0
    torch = _try_import_torch()
    if torch is None or not torch.cuda.is_available() or not config.gpu_ids:
        return 0.0
    peak_bytes = max(torch.cuda.max_memory_allocated(gpu_id) for gpu_id in config.gpu_ids)
    return peak_bytes / float(1024**3)


def _memory_measurement_source(config: RunConfig) -> str:
    if config.device == "cuda":
        return "torch_cuda_max_memory_allocated"
    return "cpu_simulated_no_cuda"


def _try_import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch
