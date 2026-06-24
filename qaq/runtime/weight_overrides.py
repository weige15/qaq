"""Helpers for applying selected bit-plane artifact weights during runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qaq.artifacts import load_bitplane_artifact
from qaq.benchmark_adapter import BenchmarkBatchMetadata, TokenizedBenchmarkBatch
from qaq.bitplanes import reconstruct_weight
from qaq.blocks import BlockDescriptor, block_map
from qaq.config import RunConfig
from qaq.model_adapter import HiddenStateBundle, ReferenceBatchOutput
from qaq.precision_plan import PrecisionPlan
from qaq.runtime.common import RuntimeError
from qaq.tensor_bitplanes import (
    is_tensor_bitplane_artifact_path,
    load_tensor_bitplane_artifact,
    reconstruct_tensor_weight,
)


@dataclass(frozen=True, slots=True)
class PlanArtifactRef:
    block_id: str
    bit_width: int
    tensor_name: str | None
    artifact_path: Path


def adapter_supports_weight_overrides(adapter: Any) -> bool:
    return bool(getattr(adapter, "supports_weight_overrides", False))


def block_has_full_tensor_artifacts(block: BlockDescriptor) -> bool:
    return all(tensor_name in block.artifact_refs for tensor_name in block.tensor_names)


def artifact_ref_mode(blocks: tuple[BlockDescriptor, ...]) -> str:
    if not blocks:
        return "none"
    full = [block_has_full_tensor_artifacts(block) for block in blocks]
    if all(full):
        return "full_tensor_index"
    if any(
        tensor_name in block.artifact_refs
        for block in blocks
        for tensor_name in block.tensor_names
    ):
        return "partial_tensor_index"
    if any(any(key.isdigit() for key in block.artifact_refs) for block in blocks):
        return "legacy_bit_width_index"
    return "missing"


def artifact_paths_for_block(block: BlockDescriptor, bit_width: int) -> tuple[PlanArtifactRef, ...]:
    if block_has_full_tensor_artifacts(block):
        return tuple(
            PlanArtifactRef(
                block_id=block.block_id,
                bit_width=bit_width,
                tensor_name=tensor_name,
                artifact_path=Path(block.artifact_refs[tensor_name]),
            )
            for tensor_name in block.tensor_names
        )
    tensor_refs = [name for name in block.tensor_names if name in block.artifact_refs]
    if tensor_refs:
        missing = [name for name in block.tensor_names if name not in block.artifact_refs]
        raise RuntimeError(
            "missing_tensor_artifact",
            f"{block.block_id} tensor artifact index is incomplete; missing tensors: {missing}",
        )
    artifact_path = block.artifact_refs.get(str(bit_width))
    if artifact_path is None:
        raise RuntimeError(
            "missing_artifact",
            f"{block.block_id} is missing artifact for {bit_width}-bit precision",
        )
    return (
        PlanArtifactRef(
            block_id=block.block_id,
            bit_width=bit_width,
            tensor_name=None,
            artifact_path=Path(artifact_path),
        ),
    )


def runtime_can_apply_weight_overrides(
    adapter: Any,
    blocks: tuple[BlockDescriptor, ...],
) -> tuple[bool, str]:
    if not adapter_supports_weight_overrides(adapter):
        return False, "adapter_does_not_support_weight_overrides"
    mode = artifact_ref_mode(blocks)
    if mode != "full_tensor_index":
        return False, f"artifact_ref_mode:{mode}"
    return True, "full_tensor_index_available"


def build_weight_overrides(
    config: RunConfig,
    *,
    blocks: tuple[BlockDescriptor, ...],
    plan: PrecisionPlan,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    torch = _import_torch()
    descriptors = block_map(blocks)
    overrides: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for block_id, bit_width in plan.decisions.items():
        block = descriptors[block_id]
        for ref in artifact_paths_for_block(block, bit_width):
            if ref.tensor_name is None:
                raise RuntimeError(
                    "missing_tensor_artifact",
                    f"{block_id} needs per-tensor artifacts before real weight override execution",
                )
            tensor = _load_reconstructed_weight_tensor(
                config,
                ref=ref,
                torch=torch,
            )
            overrides[ref.tensor_name] = tensor
            records.append(
                {
                    "block_id": block_id,
                    "bit_width": bit_width,
                    "tensor_name": ref.tensor_name,
                    "artifact_path": str(ref.artifact_path),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "device": str(tensor.device),
                }
            )
    return overrides, tuple(records)


def slice_batch(batch: TokenizedBenchmarkBatch, index: int) -> TokenizedBenchmarkBatch:
    example_id = batch.example_ids[index]
    truncated = tuple(
        item for item in batch.metadata.truncated_examples if item == example_id
    )
    metadata = BenchmarkBatchMetadata(
        dataset=batch.metadata.dataset,
        split=batch.metadata.split,
        prompt_format=batch.metadata.prompt_format,
        tokenizer=batch.metadata.tokenizer,
        batch_size=1,
        max_length=batch.metadata.max_length,
        context_length_policy=batch.metadata.context_length_policy,
        truncated_examples=truncated,
        example_ids=(example_id,),
    )
    return TokenizedBenchmarkBatch(
        input_ids=(batch.input_ids[index],),
        attention_mask=(batch.attention_mask[index],),
        targets=(batch.targets[index],),
        examples=(batch.examples[index],),
        metadata=metadata,
    )


def combine_reference_outputs(
    outputs: tuple[ReferenceBatchOutput, ...],
    *,
    batch: TokenizedBenchmarkBatch,
    precision_label: str,
    metadata_updates: dict[str, Any],
) -> ReferenceBatchOutput:
    if not outputs:
        raise RuntimeError("empty_runtime_output", "at least one per-query output is required")
    block_ids = tuple(outputs[0].hidden_states.by_block)
    by_block = {
        block_id: tuple(output.hidden_states.by_block[block_id][0] for output in outputs)
        for block_id in block_ids
    }
    metadata = dict(outputs[0].metadata)
    metadata.update(
        {
            "batch_size": batch.metadata.batch_size,
            "precision": precision_label,
            "example_ids": list(batch.example_ids),
        }
    )
    metadata.update(metadata_updates)
    return ReferenceBatchOutput(
        logits=tuple(row for output in outputs for row in output.logits),
        losses=tuple(loss for output in outputs for loss in output.losses),
        predictions=tuple(prediction for output in outputs for prediction in output.predictions),
        hidden_states=HiddenStateBundle(
            feature_source=outputs[0].hidden_states.feature_source,
            by_block=by_block,
        ),
        metadata=metadata,
    )


def _load_reconstructed_weight_tensor(
    config: RunConfig,
    *,
    ref: PlanArtifactRef,
    torch: Any,
) -> Any:
    if is_tensor_bitplane_artifact_path(ref.artifact_path):
        artifact = load_tensor_bitplane_artifact(ref.artifact_path)
        reconstructed = reconstruct_tensor_weight(
            artifact,
            bit_width=ref.bit_width,
            model_id=config.model,
            block_id=ref.block_id,
            tensor_name=ref.tensor_name,
        )
        params = reconstructed.metadata.quantization
        return (
            reconstructed.quantized_values.to(dtype=torch.float32) - float(params.zero_point)
        ) * float(params.scale)

    artifact = load_bitplane_artifact(ref.artifact_path)
    reconstructed = reconstruct_weight(
        artifact,
        bit_width=ref.bit_width,
        model_id=config.model,
        block_id=ref.block_id,
        tensor_name=ref.tensor_name,
    )
    return torch.tensor(reconstructed.values, dtype=torch.float32)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch_unavailable",
            "real bit-plane weight override execution requires torch",
        ) from exc
    return torch
