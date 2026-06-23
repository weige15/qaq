"""Tensor-native QAQ bit-plane artifacts backed by safetensors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qaq.bitplanes import (
    ARTIFACT_VERSION,
    RECONSTRUCTION_POLICY,
    BitPlaneArtifactMetadata,
    BitPlaneError,
    selected_msb_planes,
)
from qaq.quantization import QuantizationParams


TENSOR_ARTIFACT_VERSION = "qaq.tensor_bitplane.v1"
TENSOR_STORAGE_LAYOUT = "packed_uint8_bitplanes"
TENSOR_METADATA_KEY = "qaq_metadata"
TENSOR_VALUES_KEY = "quantized_values"


@dataclass(slots=True)
class TensorBitPlaneError(ValueError):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class TensorReconstructionResult:
    bit_width: int
    selected_planes: tuple[int, ...]
    quantized_values: Any
    metadata: BitPlaneArtifactMetadata

    def as_dict(self) -> dict[str, Any]:
        return {
            "bit_width": self.bit_width,
            "selected_planes": list(self.selected_planes),
            "shape": list(self.quantized_values.shape),
            "dtype": str(self.quantized_values.dtype).removeprefix("torch."),
            "device": str(self.quantized_values.device),
            "metadata": self.metadata.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class TensorBitPlaneArtifact:
    metadata: BitPlaneArtifactMetadata
    quantized_values: Any
    storage_layout: str = TENSOR_STORAGE_LAYOUT

    def validate(self) -> None:
        _validate_tensor_metadata(self.metadata)
        torch = _import_torch()
        if not isinstance(self.quantized_values, torch.Tensor):
            raise TensorBitPlaneError(
                "invalid_tensor_artifact",
                "quantized_values must be a torch.Tensor",
            )
        if self.quantized_values.dtype != torch.uint8:
            raise TensorBitPlaneError(
                "invalid_tensor_artifact",
                "quantized_values must be stored as torch.uint8",
            )
        if tuple(int(value) for value in self.quantized_values.shape) != self.metadata.original_shape:
            raise TensorBitPlaneError(
                "shape_mismatch",
                "quantized tensor shape must match artifact metadata",
            )


def create_tensor_bitplane_artifact(
    tensor: Any,
    *,
    model_id: str,
    block_id: str,
    tensor_name: str,
    max_bit_width: int = 8,
    original_dtype: str | None = None,
    checkpoint_ref: str | None = None,
    compatibility: dict[str, Any] | None = None,
) -> TensorBitPlaneArtifact:
    """Quantize a torch tensor into packed uint8 bit-plane storage."""

    if max_bit_width != 8:
        raise TensorBitPlaneError(
            "unsupported_max_bit_width",
            "tensor-native packed bit-plane artifacts currently support max_bit_width=8",
        )
    torch = _import_torch()
    if not isinstance(tensor, torch.Tensor):
        raise TensorBitPlaneError(
            "invalid_tensor",
            "tensor-native bit-plane artifacts require a torch.Tensor input",
        )
    if tensor.numel() <= 0:
        raise TensorBitPlaneError("empty_tensor", "tensor must contain values")

    source_dtype = original_dtype or str(tensor.dtype).removeprefix("torch.")
    quantized, params = _quantize_tensor_to_uint8(tensor)
    metadata = BitPlaneArtifactMetadata(
        model_id=_non_empty(model_id, "model_id"),
        block_id=_non_empty(block_id, "block_id"),
        tensor_name=_non_empty(tensor_name, "tensor_name"),
        original_shape=tuple(int(value) for value in quantized.shape),
        original_dtype=source_dtype,
        max_bit_width=max_bit_width,
        available_planes=tuple(range(max_bit_width)),
        quantization=params,
        reconstruction_policy=RECONSTRUCTION_POLICY,
        artifact_version=TENSOR_ARTIFACT_VERSION,
        checksum=None,
        validation_status="valid",
        checkpoint_ref=checkpoint_ref,
        compatibility={
            **(compatibility or {}),
            "storage_layout": TENSOR_STORAGE_LAYOUT,
            "tensor_native": True,
        },
    )
    artifact = TensorBitPlaneArtifact(metadata=metadata, quantized_values=quantized)
    artifact.validate()
    return artifact


def save_tensor_bitplane_artifact(
    artifact: TensorBitPlaneArtifact,
    path: str | Path,
) -> Path:
    artifact.validate()
    safetensors_torch = _import_safetensors_torch()
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(
        {TENSOR_VALUES_KEY: artifact.quantized_values.detach().cpu().contiguous()},
        str(artifact_path),
        metadata={
            TENSOR_METADATA_KEY: json.dumps(
                {
                    "metadata": artifact.metadata.as_dict(),
                    "storage_layout": artifact.storage_layout,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        },
    )
    return artifact_path


def load_tensor_bitplane_artifact(path: str | Path) -> TensorBitPlaneArtifact:
    artifact_path = Path(path)
    safe_open = _import_safe_open()
    try:
        with safe_open(artifact_path, framework="pt", device="cpu") as handle:
            metadata_json = (handle.metadata() or {}).get(TENSOR_METADATA_KEY)
            if not metadata_json:
                raise TensorBitPlaneError(
                    "invalid_tensor_artifact",
                    "safetensors metadata is missing qaq_metadata",
                )
            raw = json.loads(metadata_json)
            if not isinstance(raw, dict):
                raise TensorBitPlaneError(
                    "invalid_tensor_artifact",
                    "qaq_metadata must be a JSON object",
                )
            metadata = BitPlaneArtifactMetadata.from_mapping(raw["metadata"])
            storage_layout = raw.get("storage_layout")
            if storage_layout != TENSOR_STORAGE_LAYOUT:
                raise TensorBitPlaneError(
                    "unsupported_storage_layout",
                    f"unsupported tensor artifact layout {storage_layout!r}",
                )
            quantized_values = handle.get_tensor(TENSOR_VALUES_KEY)
    except TensorBitPlaneError:
        raise
    except OSError as exc:
        raise TensorBitPlaneError("tensor_artifact_read_failed", str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise TensorBitPlaneError("tensor_artifact_parse_failed", str(exc)) from exc
    except KeyError as exc:
        raise TensorBitPlaneError(
            "invalid_tensor_artifact",
            f"missing tensor key {TENSOR_VALUES_KEY!r}",
        ) from exc
    artifact = TensorBitPlaneArtifact(
        metadata=metadata,
        quantized_values=quantized_values,
        storage_layout=storage_layout,
    )
    artifact.validate()
    return artifact


def is_tensor_bitplane_artifact_path(path: str | Path) -> bool:
    return str(path).endswith(".qaq.safetensors")


def reconstruct_tensor_weight(
    artifact: TensorBitPlaneArtifact,
    *,
    bit_width: int,
    model_id: str | None = None,
    block_id: str | None = None,
    tensor_name: str | None = None,
) -> TensorReconstructionResult:
    artifact.validate()
    _validate_compatibility(
        artifact.metadata,
        model_id=model_id,
        block_id=block_id,
        tensor_name=tensor_name,
    )
    selected = selected_msb_planes(
        bit_width,
        max_bit_width=artifact.metadata.max_bit_width,
    )
    mask = sum(1 << plane_index for plane_index in selected)
    quantized = artifact.quantized_values.bitwise_and(mask)
    return TensorReconstructionResult(
        bit_width=bit_width,
        selected_planes=selected,
        quantized_values=quantized,
        metadata=artifact.metadata,
    )


def normalized_tensor_reconstruction_delta(
    artifact: TensorBitPlaneArtifact,
    *,
    bit_width: int,
    model_id: str | None = None,
    block_id: str | None = None,
) -> float:
    full = reconstruct_tensor_weight(
        artifact,
        bit_width=artifact.metadata.max_bit_width,
        model_id=model_id,
        block_id=block_id,
    )
    candidate = reconstruct_tensor_weight(
        artifact,
        bit_width=bit_width,
        model_id=model_id,
        block_id=block_id,
    )
    delta = (
        candidate.quantized_values.to(dtype=_import_torch().float32)
        - full.quantized_values.to(dtype=_import_torch().float32)
    ).abs()
    qmax = max(1, artifact.metadata.quantization.qmax)
    return float(delta.mean().item()) / float(qmax)


def pack_selected_msb_planes(
    artifact: TensorBitPlaneArtifact,
    *,
    bit_width: int,
    target_device: str = "cpu",
) -> tuple[Any, tuple[int, ...]]:
    """Pack selected MSB planes into a byte tensor on the target device."""

    reconstruction = reconstruct_tensor_weight(artifact, bit_width=bit_width)
    torch = _import_torch()
    device = torch.device(target_device)
    values = reconstruction.quantized_values.to(device=device, dtype=torch.uint8).reshape(-1)
    plane_indices = torch.tensor(
        list(reconstruction.selected_planes),
        dtype=torch.uint8,
        device=device,
    )
    bits = ((values[:, None] >> plane_indices[None, :]) & 1).reshape(-1).to(torch.uint8)
    pad = (-int(bits.numel())) % 8
    if pad:
        bits = torch.cat([bits, torch.zeros(pad, dtype=torch.uint8, device=device)])
    bit_weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=device)
    packed = (bits.reshape(-1, 8) * bit_weights).sum(dim=1).to(torch.uint8)
    return packed, reconstruction.selected_planes


def _quantize_tensor_to_uint8(tensor: Any) -> tuple[Any, QuantizationParams]:
    torch = _import_torch()
    values = tensor.detach().to(device="cpu", dtype=torch.float32)
    min_value = float(values.min().item())
    max_value = float(values.max().item())
    qmin = 0
    qmax = 255
    if min_value == max_value:
        scale = 1.0
        zero_point = max(qmin, min(qmax, round(-min_value)))
    else:
        scale = (max_value - min_value) / float(qmax - qmin)
        zero_point = max(qmin, min(qmax, round(qmin - min_value / scale)))
    quantized = torch.clamp(torch.round(values / scale + zero_point), qmin, qmax).to(torch.uint8)
    return (
        quantized,
        QuantizationParams(
            scheme="affine_uint8_per_tensor_safetensors",
            max_bit_width=8,
            scale=scale,
            zero_point=int(zero_point),
            qmin=qmin,
            qmax=qmax,
            group_size=None,
        ),
    )


def _validate_tensor_metadata(metadata: BitPlaneArtifactMetadata) -> None:
    if metadata.artifact_version != TENSOR_ARTIFACT_VERSION:
        raise TensorBitPlaneError(
            "unsupported_tensor_artifact",
            f"unsupported tensor artifact version {metadata.artifact_version}",
        )
    if metadata.max_bit_width != 8 or metadata.quantization.max_bit_width != 8:
        raise TensorBitPlaneError(
            "unsupported_max_bit_width",
            "tensor-native artifacts currently require max_bit_width=8",
        )
    if tuple(sorted(metadata.available_planes)) != tuple(range(metadata.max_bit_width)):
        raise TensorBitPlaneError(
            "missing_plane",
            "tensor-native artifact must contain all packed planes",
        )
    if metadata.reconstruction_policy != RECONSTRUCTION_POLICY:
        raise TensorBitPlaneError(
            "unsupported_reconstruction_policy",
            f"unsupported reconstruction policy {metadata.reconstruction_policy}",
        )
    if metadata.validation_status != "valid":
        raise TensorBitPlaneError(
            "invalid_artifact_status",
            f"artifact validation_status is {metadata.validation_status}",
        )


def _validate_compatibility(
    metadata: BitPlaneArtifactMetadata,
    *,
    model_id: str | None,
    block_id: str | None,
    tensor_name: str | None,
) -> None:
    if model_id is not None and metadata.model_id != model_id:
        raise TensorBitPlaneError(
            "artifact_model_mismatch",
            f"artifact model_id {metadata.model_id!r} does not match {model_id!r}",
        )
    if block_id is not None and metadata.block_id != block_id:
        raise TensorBitPlaneError(
            "artifact_block_mismatch",
            f"artifact block_id {metadata.block_id!r} does not match {block_id!r}",
        )
    if tensor_name is not None and metadata.tensor_name != tensor_name:
        raise TensorBitPlaneError(
            "artifact_tensor_mismatch",
            f"artifact tensor_name {metadata.tensor_name!r} does not match {tensor_name!r}",
        )


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise TensorBitPlaneError(
            "torch_unavailable",
            "tensor-native bit-plane artifacts require torch",
        ) from exc
    return torch


def _import_safetensors_torch() -> Any:
    try:
        import safetensors.torch
    except ImportError as exc:
        raise TensorBitPlaneError(
            "safetensors_unavailable",
            "tensor-native bit-plane artifacts require safetensors",
        ) from exc
    return safetensors.torch


def _import_safe_open() -> Any:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise TensorBitPlaneError(
            "safetensors_unavailable",
            "tensor-native bit-plane artifacts require safetensors",
        ) from exc
    return safe_open


def _non_empty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TensorBitPlaneError("invalid_metadata", f"{field} must be a non-empty string")
    return value
