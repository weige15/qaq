"""Generate QAQ bit-plane artifacts from local Hugging Face LLaMA weights."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from qaq.artifacts import save_bitplane_artifact
from qaq.bitplanes import create_bitplane_artifact
from qaq.blocks import BlockDescriptor, discover_mha_ffn_blocks
from qaq.config import RunConfig
from qaq.model_adapter import load_model_adapter
from qaq.tensor_bitplanes import create_tensor_bitplane_artifact, save_tensor_bitplane_artifact


DEFAULT_MAX_ELEMENTS = 4096
ARTIFACT_FORMAT_JSON = "json"
ARTIFACT_FORMAT_SAFETENSORS = "safetensors"
ARTIFACT_FORMATS = frozenset({ARTIFACT_FORMAT_JSON, ARTIFACT_FORMAT_SAFETENSORS})


@dataclass(slots=True)
class LlamaBitPlaneGenerationError(ValueError):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class GeneratedArtifactRecord:
    block_id: str
    tensor_name: str
    artifact_path: Path
    source_tensor_shape: tuple[int, ...]
    source_tensor_dtype: str
    artifact_element_count: int
    source_tensor_numel: int
    truncated: bool
    storage_layout: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "tensor_name": self.tensor_name,
            "artifact_path": str(self.artifact_path),
            "source_tensor_shape": list(self.source_tensor_shape),
            "source_tensor_dtype": self.source_tensor_dtype,
            "artifact_element_count": self.artifact_element_count,
            "source_tensor_numel": self.source_tensor_numel,
            "truncated": self.truncated,
            "storage_layout": self.storage_layout,
        }


@dataclass(frozen=True, slots=True)
class LlamaBitPlaneGenerationResult:
    output_dir: Path
    manifest_path: Path
    tensor_index_path: Path
    runtime_index_path: Path
    records: tuple[GeneratedArtifactRecord, ...]
    model_snapshot: Path
    artifact_format: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "tensor_index_path": str(self.tensor_index_path),
            "runtime_index_path": str(self.runtime_index_path),
            "artifact_count": len(self.records),
            "model_snapshot": str(self.model_snapshot),
            "artifact_format": self.artifact_format,
            "records": [record.as_dict() for record in self.records],
        }


def generate_llama_bitplane_artifacts(
    *,
    model: str,
    tokenizer: str | None,
    output_dir: Path,
    precision_candidates: tuple[int, ...] = (4, 8),
    max_bit_width: int = 8,
    block_limit: int | None = None,
    tensor_limit_per_block: int | None = None,
    max_elements_per_tensor: int | None = DEFAULT_MAX_ELEMENTS,
    allow_full_tensor_json: bool = False,
    artifact_format: str = ARTIFACT_FORMAT_JSON,
    overwrite: bool = False,
) -> LlamaBitPlaneGenerationResult:
    """Create QAQ bit-plane artifacts from locally cached LLaMA safetensors.

    ``max_elements_per_tensor`` limits artifact size for acceptance probes. Use
    ``allow_full_tensor_json=True`` with ``max_elements_per_tensor=None`` only
    when intentionally generating full JSON artifacts.
    """

    if artifact_format not in ARTIFACT_FORMATS:
        raise LlamaBitPlaneGenerationError(
            "unsupported_artifact_format",
            f"artifact_format must be one of {sorted(ARTIFACT_FORMATS)}",
        )
    if output_dir.exists():
        if not overwrite:
            raise LlamaBitPlaneGenerationError(
                "unsafe_output_reuse",
                "output directory already exists; pass overwrite=True to replace it",
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    snapshot = _resolve_model_snapshot(model)
    adapter_config = RunConfig.from_mapping(
        {
            "model": model,
            "tokenizer": tokenizer or model,
            "dataset": "fake_smoke",
            "split": "validation",
            "mode": "fp16",
            "precision_candidates": list(precision_candidates),
            "max_bit_width": max_bit_width,
            "block_granularity": "mha_ffn",
            "device": "cpu",
            "gpu_ids": [],
            "seed": 0,
            "output_dir": str(output_dir / "_adapter_probe"),
            "overwrite": True,
            "logging": {"console": False},
            "prompt_format": "plain",
            "metric": "artifact_generation",
        },
        validate_output=False,
    )
    adapter = load_model_adapter(adapter_config)
    blocks = discover_mha_ffn_blocks(
        adapter.architecture_metadata,
        supported_bit_widths=precision_candidates,
    )
    selected_blocks = blocks[:block_limit] if block_limit is not None else blocks
    if not selected_blocks:
        raise LlamaBitPlaneGenerationError(
            "no_blocks_selected",
            "at least one LLaMA block must be selected",
        )

    weight_map = _load_safetensor_weight_map(snapshot)
    records: list[GeneratedArtifactRecord] = []
    for block in selected_blocks:
        tensor_names = (
            block.tensor_names[:tensor_limit_per_block]
            if tensor_limit_per_block is not None
            else block.tensor_names
        )
        for tensor_name in tensor_names:
            records.append(
                _generate_tensor_artifact(
                    model=model,
                    snapshot=snapshot,
                    block=block,
                    tensor_name=tensor_name,
                    weight_map=weight_map,
                    output_dir=output_dir,
                    max_bit_width=max_bit_width,
                    max_elements_per_tensor=max_elements_per_tensor,
                    allow_full_tensor_json=allow_full_tensor_json,
                    artifact_format=artifact_format,
                )
            )
    if not records:
        raise LlamaBitPlaneGenerationError(
            "no_artifacts_generated",
            "no selected block tensors produced artifacts",
        )

    tensor_index_path = _write_tensor_index(
        output_dir,
        records=tuple(records),
    )
    runtime_index_path = _write_runtime_index(
        output_dir,
        blocks=tuple(selected_blocks),
        records=tuple(records),
    )
    manifest_path = _write_generation_manifest(
        output_dir,
        model=model,
        snapshot=snapshot,
        precision_candidates=precision_candidates,
        max_bit_width=max_bit_width,
        blocks=tuple(selected_blocks),
        discovered_block_count=len(blocks),
        block_count=len(selected_blocks),
        tensor_limit_per_block=tensor_limit_per_block,
        max_elements_per_tensor=max_elements_per_tensor,
        allow_full_tensor_json=allow_full_tensor_json,
        artifact_format=artifact_format,
        records=tuple(records),
        tensor_index_path=tensor_index_path,
        runtime_index_path=runtime_index_path,
    )
    return LlamaBitPlaneGenerationResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        tensor_index_path=tensor_index_path,
        runtime_index_path=runtime_index_path,
        records=tuple(records),
        model_snapshot=snapshot,
        artifact_format=artifact_format,
    )


def _generate_tensor_artifact(
    *,
    model: str,
    snapshot: Path,
    block: BlockDescriptor,
    tensor_name: str,
    weight_map: Mapping[str, Path],
    output_dir: Path,
    max_bit_width: int,
    max_elements_per_tensor: int | None,
    allow_full_tensor_json: bool,
    artifact_format: str,
) -> GeneratedArtifactRecord:
    if artifact_format == ARTIFACT_FORMAT_SAFETENSORS:
        return _generate_tensor_native_artifact(
            model=model,
            snapshot=snapshot,
            block=block,
            tensor_name=tensor_name,
            weight_map=weight_map,
            output_dir=output_dir,
            max_bit_width=max_bit_width,
            max_elements_per_tensor=max_elements_per_tensor,
        )

    (
        artifact_values,
        artifact_shape,
        source_shape,
        source_dtype,
        source_numel,
        truncated,
    ) = _load_safetensor_tensor_values(
        snapshot,
        tensor_name=tensor_name,
        weight_map=weight_map,
        max_elements_per_tensor=max_elements_per_tensor,
        allow_full_tensor_json=allow_full_tensor_json,
    )
    artifact = create_bitplane_artifact(
        artifact_values,
        model_id=model,
        block_id=block.block_id,
        tensor_name=tensor_name,
        max_bit_width=max_bit_width,
        original_dtype=source_dtype,
        checkpoint_ref=str(snapshot),
        compatibility={
            "block_granularity": "mha_ffn",
            "framework": "transformers_llama",
            "llama_compatible": True,
            "source_tensor_name": tensor_name,
            "source_tensor_shape": list(source_shape),
            "source_tensor_dtype": source_dtype,
            "source_tensor_numel": source_numel,
            "artifact_shape": list(artifact_shape),
            "artifact_element_count": _element_count(artifact_shape),
            "source_tensor_offset": 0,
            "truncated_source_tensor": truncated,
            "artifact_scope": (
                "sampled_weight_values"
                if truncated
                else "full_weight_tensor_json"
            ),
            "accepted_as_full_quantized_inference_artifact": not truncated,
        },
    )
    artifact_path = output_dir / block.block_id / f"{_safe_filename(tensor_name)}.json"
    save_bitplane_artifact(artifact, artifact_path)
    return GeneratedArtifactRecord(
        block_id=block.block_id,
        tensor_name=tensor_name,
        artifact_path=artifact_path,
        source_tensor_shape=source_shape,
        source_tensor_dtype=source_dtype,
        artifact_element_count=_element_count(artifact_shape),
        source_tensor_numel=source_numel,
        truncated=truncated,
        storage_layout="json_bitplanes",
    )


def _generate_tensor_native_artifact(
    *,
    model: str,
    snapshot: Path,
    block: BlockDescriptor,
    tensor_name: str,
    weight_map: Mapping[str, Path],
    output_dir: Path,
    max_bit_width: int,
    max_elements_per_tensor: int | None,
) -> GeneratedArtifactRecord:
    (
        tensor,
        artifact_shape,
        source_shape,
        source_dtype,
        source_numel,
        truncated,
    ) = _load_safetensor_tensor_for_artifact(
        snapshot,
        tensor_name=tensor_name,
        weight_map=weight_map,
        max_elements_per_tensor=max_elements_per_tensor,
    )
    artifact = create_tensor_bitplane_artifact(
        tensor,
        model_id=model,
        block_id=block.block_id,
        tensor_name=tensor_name,
        max_bit_width=max_bit_width,
        original_dtype=source_dtype,
        checkpoint_ref=str(snapshot),
        compatibility={
            "block_granularity": "mha_ffn",
            "framework": "transformers_llama",
            "llama_compatible": True,
            "source_tensor_name": tensor_name,
            "source_tensor_shape": list(source_shape),
            "source_tensor_dtype": source_dtype,
            "source_tensor_numel": source_numel,
            "artifact_shape": list(artifact_shape),
            "artifact_element_count": _element_count(artifact_shape),
            "source_tensor_offset": 0,
            "truncated_source_tensor": truncated,
            "artifact_scope": (
                "sampled_weight_values"
                if truncated
                else "full_weight_tensor_safetensors"
            ),
            "accepted_as_full_quantized_inference_artifact": not truncated,
        },
    )
    artifact_path = output_dir / block.block_id / f"{_safe_filename(tensor_name)}.qaq.safetensors"
    save_tensor_bitplane_artifact(artifact, artifact_path)
    return GeneratedArtifactRecord(
        block_id=block.block_id,
        tensor_name=tensor_name,
        artifact_path=artifact_path,
        source_tensor_shape=source_shape,
        source_tensor_dtype=source_dtype,
        artifact_element_count=_element_count(artifact_shape),
        source_tensor_numel=source_numel,
        truncated=truncated,
        storage_layout="packed_uint8_bitplanes",
    )


def _load_safetensor_tensor_values(
    snapshot: Path,
    *,
    tensor_name: str,
    weight_map: Mapping[str, Path],
    max_elements_per_tensor: int | None,
    allow_full_tensor_json: bool,
) -> tuple[Any, tuple[int, ...], tuple[int, ...], str, int, bool]:
    shard_path = weight_map.get(tensor_name)
    if shard_path is None:
        raise LlamaBitPlaneGenerationError(
            "missing_weight_tensor",
            f"LLaMA safetensors do not contain {tensor_name!r}",
        )
    if not shard_path.is_absolute():
        shard_path = snapshot / shard_path
    if not shard_path.is_file():
        raise LlamaBitPlaneGenerationError(
            "missing_weight_shard",
            f"safetensor shard is missing: {shard_path}",
        )
    safe_open = _import_safe_open()
    try:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(tensor_name)
            source_shape = tuple(int(value) for value in tensor_slice.get_shape())
            source_dtype = str(tensor_slice.get_dtype())
            source_numel = _element_count(source_shape)
            if max_elements_per_tensor is not None:
                tensor = _slice_head_values(
                    tensor_slice,
                    source_shape,
                    max_elements_per_tensor,
                )
                values = [
                    float(value)
                    for value in tensor.reshape(-1).float().cpu().tolist()[
                        :max_elements_per_tensor
                    ]
                ]
                return (
                    values,
                    (len(values),),
                    source_shape,
                    source_dtype,
                    source_numel,
                    source_numel > len(values),
                )
            if not allow_full_tensor_json:
                raise LlamaBitPlaneGenerationError(
                    "full_tensor_json_requires_explicit_allow",
                    "full-tensor JSON artifacts are large; pass allow_full_tensor_json=True",
                )
            tensor = handle.get_tensor(tensor_name)
    except LlamaBitPlaneGenerationError:
        raise
    except Exception as exc:
        raise LlamaBitPlaneGenerationError(
            "safetensor_tensor_read_failed",
            f"failed to load {tensor_name!r} from {shard_path}: {exc}",
        ) from exc

    values = tensor.detach().float().cpu().tolist()
    return values, source_shape, source_shape, source_dtype, source_numel, False


def _load_safetensor_tensor_for_artifact(
    snapshot: Path,
    *,
    tensor_name: str,
    weight_map: Mapping[str, Path],
    max_elements_per_tensor: int | None,
) -> tuple[Any, tuple[int, ...], tuple[int, ...], str, int, bool]:
    shard_path = weight_map.get(tensor_name)
    if shard_path is None:
        raise LlamaBitPlaneGenerationError(
            "missing_weight_tensor",
            f"LLaMA safetensors do not contain {tensor_name!r}",
        )
    if not shard_path.is_absolute():
        shard_path = snapshot / shard_path
    if not shard_path.is_file():
        raise LlamaBitPlaneGenerationError(
            "missing_weight_shard",
            f"safetensor shard is missing: {shard_path}",
        )
    safe_open = _import_safe_open()
    try:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(tensor_name)
            source_shape = tuple(int(value) for value in tensor_slice.get_shape())
            source_dtype = str(tensor_slice.get_dtype())
            source_numel = _element_count(source_shape)
            if max_elements_per_tensor is not None:
                tensor = _slice_head_values(
                    tensor_slice,
                    source_shape,
                    max_elements_per_tensor,
                ).reshape(-1)[:max_elements_per_tensor].contiguous()
                return (
                    tensor,
                    tuple(int(value) for value in tensor.shape),
                    source_shape,
                    source_dtype,
                    source_numel,
                    source_numel > int(tensor.numel()),
                )
            tensor = handle.get_tensor(tensor_name).contiguous()
            return tensor, source_shape, source_shape, source_dtype, source_numel, False
    except LlamaBitPlaneGenerationError:
        raise
    except Exception as exc:
        raise LlamaBitPlaneGenerationError(
            "safetensor_tensor_read_failed",
            f"failed to load {tensor_name!r} from {shard_path}: {exc}",
        ) from exc


def _slice_head_values(tensor_slice: Any, shape: tuple[int, ...], max_elements: int) -> Any:
    if max_elements <= 0:
        raise LlamaBitPlaneGenerationError(
            "invalid_max_elements",
            "max_elements_per_tensor must be positive when provided",
        )
    if not shape:
        return tensor_slice[()]
    if len(shape) == 1:
        return tensor_slice[: min(max_elements, shape[0])]
    if len(shape) == 2:
        row_width = max(shape[1], 1)
        rows = min(shape[0], max(1, math.ceil(max_elements / row_width)))
        return tensor_slice[:rows, :]
    raise LlamaBitPlaneGenerationError(
        "unsupported_tensor_rank",
        f"truncated artifact generation supports rank <= 2 tensors, got rank {len(shape)}",
    )


def _load_safetensor_weight_map(snapshot: Path) -> dict[str, Path]:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LlamaBitPlaneGenerationError(
                "safetensor_index_read_failed",
                str(exc),
            ) from exc
        weight_map = raw.get("weight_map")
        if not isinstance(weight_map, dict):
            raise LlamaBitPlaneGenerationError(
                "invalid_safetensor_index",
                "model.safetensors.index.json must contain a weight_map object",
            )
        return {
            tensor_name: snapshot / shard
            for tensor_name, shard in weight_map.items()
            if isinstance(tensor_name, str) and isinstance(shard, str)
        }

    safe_open = _import_safe_open()
    result: dict[str, Path] = {}
    for shard in sorted(snapshot.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for tensor_name in handle.keys():
                result[str(tensor_name)] = shard
    if not result:
        raise LlamaBitPlaneGenerationError(
            "missing_safetensors",
            f"{snapshot} does not contain safetensor weights",
        )
    return result

def _resolve_model_snapshot(model: str) -> Path:
    path = Path(model)
    if path.is_dir():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise LlamaBitPlaneGenerationError(
            "huggingface_hub_unavailable",
            "local Hugging Face model resolution requires huggingface_hub",
        ) from exc
    try:
        return Path(snapshot_download(model, local_files_only=True))
    except Exception as exc:
        raise LlamaBitPlaneGenerationError(
            "model_snapshot_unavailable",
            f"could not resolve local model snapshot for {model!r}: {exc}",
        ) from exc


def _import_safe_open() -> Any:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise LlamaBitPlaneGenerationError(
            "safetensors_unavailable",
            "LLaMA bit-plane generation requires the optional safetensors package",
        ) from exc
    return safe_open


def _write_tensor_index(
    output_dir: Path,
    *,
    records: tuple[GeneratedArtifactRecord, ...],
) -> Path:
    path = output_dir / "tensor_artifact_index.json"
    path.write_text(
        json.dumps(_tensor_index_payload(records), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_runtime_index(
    output_dir: Path,
    *,
    blocks: tuple[BlockDescriptor, ...],
    records: tuple[GeneratedArtifactRecord, ...],
) -> Path:
    coverage = _runtime_index_coverage(blocks, records)
    path = output_dir / "runtime_artifact_index.json"
    path.write_text(
        json.dumps(_tensor_index_payload(records), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if coverage["runtime_index_artifact_ref_mode"] == "missing":
        raise LlamaBitPlaneGenerationError(
            "missing_runtime_artifacts",
            "runtime artifact index did not contain any generated tensor artifacts",
        )
    return path


def _tensor_index_payload(
    records: tuple[GeneratedArtifactRecord, ...],
) -> dict[str, dict[str, str]]:
    payload: dict[str, dict[str, str]] = {}
    for record in records:
        payload.setdefault(record.block_id, {})[record.tensor_name] = str(
            record.artifact_path.resolve()
        )
    return payload


def _runtime_index_coverage(
    blocks: tuple[BlockDescriptor, ...],
    records: tuple[GeneratedArtifactRecord, ...],
) -> dict[str, Any]:
    by_block_tensor = {(record.block_id, record.tensor_name): record for record in records}
    incomplete_blocks: list[dict[str, Any]] = []
    full_block_count = 0
    generated_tensor_refs = 0
    for block in blocks:
        missing = [
            tensor_name
            for tensor_name in block.tensor_names
            if (block.block_id, tensor_name) not in by_block_tensor
        ]
        truncated = [
            tensor_name
            for tensor_name in block.tensor_names
            if (record := by_block_tensor.get((block.block_id, tensor_name))) is not None
            and record.truncated
        ]
        generated = [
            tensor_name
            for tensor_name in block.tensor_names
            if (block.block_id, tensor_name) in by_block_tensor
        ]
        generated_tensor_refs += len(generated)
        if not missing and not truncated:
            full_block_count += 1
            continue
        incomplete_blocks.append(
            {
                "block_id": block.block_id,
                "missing_tensor_names": missing,
                "truncated_tensor_names": truncated,
                "generated_tensor_names": generated,
            }
        )
    full_coverage = bool(blocks) and full_block_count == len(blocks)
    if full_coverage:
        mode = "full_tensor_index"
    elif generated_tensor_refs > 0:
        mode = "partial_tensor_index"
    else:
        mode = "missing"
    return {
        "runtime_index_format": "block_tensor_artifact_paths",
        "runtime_index_artifact_ref_mode": mode,
        "full_tensor_runtime_coverage": full_coverage,
        "full_runtime_block_count": full_block_count,
        "selected_block_count": len(blocks),
        "runtime_index_incomplete_blocks": incomplete_blocks,
    }


def _write_generation_manifest(
    output_dir: Path,
    *,
    model: str,
    snapshot: Path,
    precision_candidates: tuple[int, ...],
    max_bit_width: int,
    blocks: tuple[BlockDescriptor, ...],
    discovered_block_count: int,
    block_count: int,
    tensor_limit_per_block: int | None,
    max_elements_per_tensor: int | None,
    allow_full_tensor_json: bool,
    artifact_format: str,
    records: tuple[GeneratedArtifactRecord, ...],
    tensor_index_path: Path,
    runtime_index_path: Path,
) -> Path:
    path = output_dir / "generation_manifest.json"
    runtime_coverage = _runtime_index_coverage(blocks, records)
    all_discovered_blocks_covered = (
        bool(runtime_coverage["full_tensor_runtime_coverage"])
        and block_count == discovered_block_count
    )
    payload = {
        "model": model,
        "model_snapshot": str(snapshot),
        "precision_candidates": list(precision_candidates),
        "max_bit_width": max_bit_width,
        "discovered_block_count": discovered_block_count,
        "block_count": block_count,
        "tensor_limit_per_block": tensor_limit_per_block,
        "max_elements_per_tensor": max_elements_per_tensor,
        "allow_full_tensor_json": allow_full_tensor_json,
        "artifact_format": artifact_format,
        "artifact_count": len(records),
        "truncated_artifact_count": sum(record.truncated for record in records),
        "tensor_index_path": str(tensor_index_path),
        "runtime_index_path": str(runtime_index_path),
        "runtime_index_uses_first_tensor_per_block": False,
        "all_discovered_blocks_covered": all_discovered_blocks_covered,
        "accepted_as_full_quantized_inference_artifact": all_discovered_blocks_covered,
        **runtime_coverage,
        "records": [record.as_dict() for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _safe_filename(value: str) -> str:
    return value.replace("/", "__").replace(".", "_")


def _element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def _positive_int_or_none(value: str) -> int | None:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return None if parsed == 0 else parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate QAQ bit-plane artifacts from local LLaMA safetensors.",
    )
    parser.add_argument("--model", required=True, help="Local path or HF model id.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer id/path; defaults to model.")
    parser.add_argument("--output-dir", required=True, help="Artifact output directory.")
    parser.add_argument(
        "--precision-candidates",
        nargs="+",
        type=int,
        default=[4, 8],
        help="Candidate bit-widths to write into runtime indexes.",
    )
    parser.add_argument("--max-bit-width", type=int, default=8)
    parser.add_argument("--block-limit", type=_positive_int_or_none, default=None)
    parser.add_argument("--tensor-limit-per-block", type=_positive_int_or_none, default=None)
    parser.add_argument(
        "--max-elements-per-tensor",
        type=_positive_int_or_none,
        default=DEFAULT_MAX_ELEMENTS,
        help=(
            "Limit elements per tensor artifact; pass 0 for full tensor artifacts. "
            "Full JSON output also requires --allow-full-tensor-json."
        ),
    )
    parser.add_argument(
        "--artifact-format",
        choices=sorted(ARTIFACT_FORMATS),
        default=ARTIFACT_FORMAT_JSON,
        help="Artifact storage format to generate.",
    )
    parser.add_argument("--allow-full-tensor-json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = generate_llama_bitplane_artifacts(
            model=args.model,
            tokenizer=args.tokenizer,
            output_dir=Path(args.output_dir),
            precision_candidates=tuple(sorted(set(args.precision_candidates))),
            max_bit_width=args.max_bit_width,
            block_limit=args.block_limit,
            tensor_limit_per_block=args.tensor_limit_per_block,
            max_elements_per_tensor=args.max_elements_per_tensor,
            allow_full_tensor_json=args.allow_full_tensor_json,
            artifact_format=args.artifact_format,
            overwrite=args.overwrite,
        )
    except LlamaBitPlaneGenerationError as exc:
        print(
            json.dumps({"status": "failed", "code": exc.code, "message": exc.message}),
            file=sys.stderr,
        )
        return 1
    if args.print_json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
