"""Prepare QAQ bit-plane artifacts from local Hugging Face safetensors."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from qaq.artifacts import save_bitplane_artifact
from qaq.bitplanes import create_bitplane_artifact
from qaq.blocks import BlockDescriptor, discover_mha_ffn_blocks
from qaq.config import ConfigValidationError, RunConfig
from qaq.model_adapter import ModelAdapterError, load_model_adapter


MANIFEST_VERSION = "qaq.bitplane_prepare.v1"
ARTIFACT_SCOPE = "sampled_weight_values"
SAMPLE_POLICY = "head_row_major"


@dataclass(slots=True)
class BitPlanePreparationError(ValueError):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class PreparedArtifactRecord:
    block_id: str
    tensor_name: str
    artifact_path: Path
    source_shard: str
    source_tensor_shape: tuple[int, ...]
    source_dtype: str
    sample_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "tensor_name": self.tensor_name,
            "artifact_path": str(self.artifact_path),
            "source_shard": self.source_shard,
            "source_tensor_shape": list(self.source_tensor_shape),
            "source_dtype": self.source_dtype,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class BitPlanePreparationResult:
    model_id: str
    resolved_model_ref: Path
    output_dir: Path
    artifact_index_path: Path
    manifest_path: Path
    precision_candidates: tuple[int, ...]
    max_bit_width: int
    sample_values: int
    block_count: int
    artifact_records: tuple[PreparedArtifactRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_VERSION,
            "model_id": self.model_id,
            "resolved_model_ref": str(self.resolved_model_ref),
            "output_dir": str(self.output_dir),
            "artifact_index_path": str(self.artifact_index_path),
            "manifest_path": str(self.manifest_path),
            "artifact_scope": ARTIFACT_SCOPE,
            "sample_policy": SAMPLE_POLICY,
            "precision_candidates": list(self.precision_candidates),
            "max_bit_width": self.max_bit_width,
            "sample_values": self.sample_values,
            "block_count": self.block_count,
            "artifact_count": len(self.artifact_records),
            "artifacts": [record.as_dict() for record in self.artifact_records],
            "accepted_as_full_quantized_inference_artifact": False,
        }


def prepare_bitplane_artifacts(
    *,
    model: str,
    tokenizer: str | None,
    output_dir: str | Path,
    precision_candidates: tuple[int, ...] = (4, 8),
    max_bit_width: int = 8,
    sample_values: int = 16,
    block_limit: int | None = None,
    overwrite: bool = False,
) -> BitPlanePreparationResult:
    """Write sampled real-weight bit-plane artifacts for each Llama MHA/FFN block."""

    precision_candidates = _validate_precision_candidates(
        precision_candidates,
        max_bit_width=max_bit_width,
    )
    if (
        isinstance(sample_values, bool)
        or not isinstance(sample_values, int)
        or sample_values <= 0
    ):
        raise BitPlanePreparationError(
            "invalid_sample_values",
            "sample_values must be a positive integer",
        )
    if block_limit is not None and (
        isinstance(block_limit, bool)
        or not isinstance(block_limit, int)
        or block_limit <= 0
    ):
        raise BitPlanePreparationError(
            "invalid_block_limit",
            "block_limit must be a positive integer when provided",
        )

    output_path = Path(output_dir)
    _prepare_output_dir(output_path, overwrite=overwrite)

    adapter = _load_metadata_adapter(
        model=model,
        tokenizer=tokenizer or model,
        output_dir=output_path,
        precision_candidates=precision_candidates,
        max_bit_width=max_bit_width,
    )
    resolved_model_ref = Path(getattr(adapter, "model_ref", model))
    if not resolved_model_ref.is_dir():
        raise BitPlanePreparationError(
            "unsupported_model_storage",
            "bit-plane preparation currently requires a local Hugging Face snapshot directory",
        )

    blocks = discover_mha_ffn_blocks(
        adapter.architecture_metadata,
        supported_bit_widths=precision_candidates,
    )
    selected_blocks = blocks[:block_limit] if block_limit is not None else blocks
    weight_map = _load_safetensors_weight_map(resolved_model_ref)

    artifact_records: list[PreparedArtifactRecord] = []
    artifact_index: dict[str, dict[str, str]] = {}
    for block in selected_blocks:
        tensor_name = _select_tensor_name(block, weight_map)
        shard_name = weight_map[tensor_name]
        shard_path = resolved_model_ref / shard_name
        sample = _load_tensor_head_sample(
            shard_path,
            tensor_name=tensor_name,
            sample_values=sample_values,
        )
        artifact_path = output_path / "artifacts" / f"{block.block_id}.json"
        artifact = create_bitplane_artifact(
            sample.values,
            model_id=adapter.model_id,
            block_id=block.block_id,
            tensor_name=tensor_name,
            max_bit_width=max_bit_width,
            original_dtype=sample.source_dtype,
            checkpoint_ref=str(resolved_model_ref),
            compatibility={
                "block_granularity": "mha_ffn",
                "source_model_ref": str(resolved_model_ref),
                "source_shard": shard_name,
                "source_tensor_name": tensor_name,
                "source_tensor_shape": list(sample.source_shape),
                "source_dtype": sample.source_dtype,
                "artifact_scope": ARTIFACT_SCOPE,
                "sample_policy": SAMPLE_POLICY,
                "sample_count": len(sample.values),
                "full_tensor_values_stored": False,
                "accepted_as_full_quantized_inference_artifact": False,
            },
        )
        save_bitplane_artifact(artifact, artifact_path)
        artifact_records.append(
            PreparedArtifactRecord(
                block_id=block.block_id,
                tensor_name=tensor_name,
                artifact_path=artifact_path,
                source_shard=shard_name,
                source_tensor_shape=sample.source_shape,
                source_dtype=sample.source_dtype,
                sample_count=len(sample.values),
            )
        )
        artifact_index[block.block_id] = {
            str(bit_width): str(artifact_path.resolve())
            for bit_width in precision_candidates
        }

    artifact_index_path = output_path / "artifact_index.json"
    artifact_index_path.write_text(
        json.dumps(artifact_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = BitPlanePreparationResult(
        model_id=adapter.model_id,
        resolved_model_ref=resolved_model_ref,
        output_dir=output_path,
        artifact_index_path=artifact_index_path,
        manifest_path=output_path / "manifest.json",
        precision_candidates=precision_candidates,
        max_bit_width=max_bit_width,
        sample_values=sample_values,
        block_count=len(selected_blocks),
        artifact_records=tuple(artifact_records),
    )
    result.manifest_path.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


@dataclass(frozen=True, slots=True)
class _TensorSample:
    values: list[float]
    source_shape: tuple[int, ...]
    source_dtype: str


def _load_metadata_adapter(
    *,
    model: str,
    tokenizer: str,
    output_dir: Path,
    precision_candidates: tuple[int, ...],
    max_bit_width: int,
) -> Any:
    try:
        return load_model_adapter(
            RunConfig.from_mapping(
                {
                    "model": model,
                    "tokenizer": tokenizer,
                    "dataset": "fake_smoke",
                    "split": "validation",
                    "mode": "fp16",
                    "precision_candidates": list(precision_candidates),
                    "max_bit_width": max_bit_width,
                    "block_granularity": "mha_ffn",
                    "device": "cpu",
                    "gpu_ids": [],
                    "seed": 0,
                    "output_dir": str(output_dir),
                    "overwrite": True,
                    "logging": {"console": False},
                    "prompt_format": "paper_aligned_default",
                    "metric": "artifact_preparation",
                },
                validate_output=False,
            )
        )
    except (ConfigValidationError, ModelAdapterError) as exc:
        raise BitPlanePreparationError("model_metadata_unavailable", str(exc)) from exc


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise BitPlanePreparationError(
            "unsafe_output_reuse",
            "output directory already exists; pass --overwrite to replace metadata/artifacts",
        )
    path.mkdir(parents=True, exist_ok=True)
    artifacts = path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for stale_path in artifacts.glob("*.json"):
            stale_path.unlink()
        for stale_path in (path / "artifact_index.json", path / "manifest.json"):
            if stale_path.exists():
                stale_path.unlink()


def _load_safetensors_weight_map(model_ref: Path) -> dict[str, str]:
    index_path = model_ref / "model.safetensors.index.json"
    if not index_path.is_file():
        single_files = sorted(model_ref.glob("*.safetensors"))
        if len(single_files) == 1:
            return _weight_map_from_single_file(single_files[0])
        raise BitPlanePreparationError(
            "safetensors_index_unavailable",
            f"{index_path} is required when multiple safetensors shards are present",
        )
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BitPlanePreparationError("safetensors_index_read_failed", str(exc)) from exc
    weight_map = raw.get("weight_map") if isinstance(raw, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise BitPlanePreparationError(
            "invalid_safetensors_index",
            "model.safetensors.index.json must contain a non-empty weight_map",
        )
    result: dict[str, str] = {}
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise BitPlanePreparationError(
                "invalid_safetensors_index",
                "weight_map tensor names must be non-empty strings",
            )
        if not isinstance(shard_name, str) or not shard_name:
            raise BitPlanePreparationError(
                "invalid_safetensors_index",
                "weight_map shard names must be non-empty strings",
            )
        if not (model_ref / shard_name).is_file():
            raise BitPlanePreparationError(
                "safetensors_shard_unavailable",
                f"missing safetensors shard {model_ref / shard_name}",
            )
        result[tensor_name] = shard_name
    return result


def _weight_map_from_single_file(path: Path) -> dict[str, str]:
    safe_open = _import_safe_open()
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    except Exception as exc:
        raise BitPlanePreparationError("safetensors_read_failed", str(exc)) from exc
    if not keys:
        raise BitPlanePreparationError(
            "safetensors_empty",
            f"{path} contains no tensors",
        )
    return {key: path.name for key in keys}


def _select_tensor_name(
    block: BlockDescriptor,
    weight_map: Mapping[str, str],
) -> str:
    for tensor_name in block.tensor_names:
        if tensor_name in weight_map:
            return tensor_name
    raise BitPlanePreparationError(
        "block_tensor_unavailable",
        f"no safetensors weight was found for {block.block_id}; expected one of {list(block.tensor_names)}",
    )


def _load_tensor_head_sample(
    shard_path: Path,
    *,
    tensor_name: str,
    sample_values: int,
) -> _TensorSample:
    safe_open = _import_safe_open()
    try:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(tensor_name)
            source_shape = tuple(int(dimension) for dimension in tensor_slice.get_shape())
            source_dtype = str(tensor_slice.get_dtype())
            tensor = _slice_head_values(tensor_slice, source_shape, sample_values)
    except BitPlanePreparationError:
        raise
    except Exception as exc:
        raise BitPlanePreparationError("safetensors_read_failed", str(exc)) from exc
    values = [float(value) for value in tensor.reshape(-1).float().cpu().tolist()]
    if not values:
        raise BitPlanePreparationError(
            "empty_tensor_sample",
            f"{tensor_name} produced an empty sample",
        )
    return _TensorSample(
        values=values[:sample_values],
        source_shape=source_shape,
        source_dtype=source_dtype,
    )


def _slice_head_values(tensor_slice: Any, shape: tuple[int, ...], sample_values: int) -> Any:
    if not shape:
        return tensor_slice[()]
    if len(shape) == 1:
        return tensor_slice[: min(sample_values, shape[0])]
    if len(shape) == 2:
        row_width = max(shape[1], 1)
        rows = min(shape[0], max(1, math.ceil(sample_values / row_width)))
        return tensor_slice[:rows, :]
    raise BitPlanePreparationError(
        "unsupported_tensor_rank",
        f"tensor rank {len(shape)} is not supported by sampled artifact preparation",
    )


def _validate_precision_candidates(
    value: tuple[int, ...],
    *,
    max_bit_width: int,
) -> tuple[int, ...]:
    if (
        isinstance(max_bit_width, bool)
        or not isinstance(max_bit_width, int)
        or max_bit_width <= 0
    ):
        raise BitPlanePreparationError(
            "invalid_max_bit_width",
            "max_bit_width must be a positive integer",
        )
    if not value:
        raise BitPlanePreparationError(
            "invalid_precision_candidates",
            "at least one precision candidate is required",
        )
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise BitPlanePreparationError(
            "invalid_precision_candidates",
            "precision candidates must be positive integers",
        )
    if any(item > max_bit_width for item in value):
        raise BitPlanePreparationError(
            "invalid_precision_candidates",
            "precision candidates cannot exceed max_bit_width",
        )
    if len(set(value)) != len(value):
        raise BitPlanePreparationError(
            "invalid_precision_candidates",
            "precision candidates must be unique",
        )
    return tuple(sorted(value))


def _import_safe_open() -> Any:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise BitPlanePreparationError(
            "safetensors_unavailable",
            "safetensors is required for Hugging Face bit-plane preparation",
        ) from exc
    return safe_open


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare sampled QAQ bit-plane artifacts from local safetensors.",
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local snapshot path.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer ID/path. Defaults to --model.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for artifacts.")
    parser.add_argument(
        "--precision-candidates",
        nargs="+",
        type=int,
        default=[4, 8],
        help="Candidate bit-widths to include in artifact_index.json.",
    )
    parser.add_argument("--max-bit-width", type=int, default=8)
    parser.add_argument(
        "--sample-values",
        type=int,
        default=16,
        help="Number of real source weight values to store per block artifact.",
    )
    parser.add_argument(
        "--block-limit",
        type=int,
        default=None,
        help="Optional development limit on number of discovered blocks to prepare.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = prepare_bitplane_artifacts(
            model=args.model,
            tokenizer=args.tokenizer,
            output_dir=args.output_dir,
            precision_candidates=tuple(args.precision_candidates),
            max_bit_width=args.max_bit_width,
            sample_values=args.sample_values,
            block_limit=args.block_limit,
            overwrite=args.overwrite,
        )
    except BitPlanePreparationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.print_json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
