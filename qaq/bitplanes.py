"""Bit-plane artifact creation, validation, and reconstruction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from qaq.quantization import (
    NestedFloat,
    NestedInt,
    QuantizationParams,
    QuantizedTensor,
    dequantize_values,
    flatten_tensor,
    infer_shape,
    quantize_tensor,
    quantized_tensor_from_values,
    unflatten_tensor,
)


ARTIFACT_VERSION = "qaq.bitplane.v1"
RECONSTRUCTION_POLICY = "msb_truncation"


@dataclass(slots=True)
class BitPlaneError(ValueError):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class BitPlaneArtifactMetadata:
    """Metadata needed to safely reuse a bit-plane tensor artifact."""

    model_id: str
    block_id: str
    tensor_name: str
    original_shape: tuple[int, ...]
    original_dtype: str
    max_bit_width: int
    available_planes: tuple[int, ...]
    quantization: QuantizationParams
    reconstruction_policy: str = RECONSTRUCTION_POLICY
    artifact_version: str = ARTIFACT_VERSION
    checksum: str | None = None
    validation_status: str = "valid"
    checkpoint_ref: str | None = None
    compatibility: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "block_id": self.block_id,
            "tensor_name": self.tensor_name,
            "original_shape": list(self.original_shape),
            "original_dtype": self.original_dtype,
            "max_bit_width": self.max_bit_width,
            "available_planes": list(self.available_planes),
            "quantization": self.quantization.as_dict(),
            "reconstruction_policy": self.reconstruction_policy,
            "artifact_version": self.artifact_version,
            "checksum": self.checksum,
            "validation_status": self.validation_status,
            "checkpoint_ref": self.checkpoint_ref,
            "compatibility": dict(self.compatibility or {}),
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "BitPlaneArtifactMetadata":
        return cls(
            model_id=_require_string(value, "model_id"),
            block_id=_require_string(value, "block_id"),
            tensor_name=_require_string(value, "tensor_name"),
            original_shape=tuple(_require_int_list(value, "original_shape")),
            original_dtype=_require_string(value, "original_dtype"),
            max_bit_width=_require_int(value, "max_bit_width"),
            available_planes=tuple(_require_int_list(value, "available_planes")),
            quantization=QuantizationParams.from_mapping(
                _require_mapping(value, "quantization")
            ),
            reconstruction_policy=value.get(
                "reconstruction_policy",
                RECONSTRUCTION_POLICY,
            ),
            artifact_version=value.get("artifact_version", ARTIFACT_VERSION),
            checksum=value.get("checksum"),
            validation_status=value.get("validation_status", "valid"),
            checkpoint_ref=value.get("checkpoint_ref"),
            compatibility=dict(value.get("compatibility") or {}),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    """Reconstruction output for one requested effective bit-width."""

    bit_width: int
    selected_planes: tuple[int, ...]
    quantized_values: NestedInt
    values: NestedFloat
    metadata: BitPlaneArtifactMetadata

    def as_dict(self) -> dict[str, Any]:
        return {
            "bit_width": self.bit_width,
            "selected_planes": list(self.selected_planes),
            "quantized_values": self.quantized_values,
            "values": self.values,
            "metadata": self.metadata.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class BitPlaneArtifact:
    """A single tensor's bit-plane representation."""

    metadata: BitPlaneArtifactMetadata
    planes: dict[int, NestedInt]

    def as_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.as_dict(),
            "planes": {str(index): plane for index, plane in sorted(self.planes.items())},
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "BitPlaneArtifact":
        metadata = BitPlaneArtifactMetadata.from_mapping(
            _require_mapping(value, "metadata")
        )
        raw_planes = _require_mapping(value, "planes")
        planes: dict[int, NestedInt] = {}
        for key, plane in raw_planes.items():
            try:
                plane_index = int(key)
            except ValueError as exc:
                raise BitPlaneError(
                    "invalid_plane_index",
                    f"plane key {key!r} is not an integer",
                ) from exc
            planes[plane_index] = plane
        artifact = cls(metadata=metadata, planes=planes)
        artifact.validate()
        return artifact

    def validate(self) -> None:
        _validate_metadata(self.metadata)
        if set(self.planes) != set(self.metadata.available_planes):
            raise BitPlaneError(
                "plane_metadata_mismatch",
                "available_planes must match stored plane keys",
            )
        for plane_index, plane in self.planes.items():
            _validate_plane_tensor(
                plane,
                plane_index=plane_index,
                max_bit_width=self.metadata.max_bit_width,
                expected_shape=self.metadata.original_shape,
            )
        if self.metadata.checksum:
            expected = artifact_checksum(self.planes, self.metadata)
            if expected != self.metadata.checksum:
                raise BitPlaneError(
                    "checksum_mismatch",
                    "artifact checksum does not match planes and metadata",
                )


def create_bitplane_artifact(
    tensor: Any,
    *,
    model_id: str,
    block_id: str,
    tensor_name: str,
    max_bit_width: int = 8,
    original_dtype: str = "float32",
    checkpoint_ref: str | None = None,
    compatibility: dict[str, Any] | None = None,
) -> BitPlaneArtifact:
    quantized = quantize_tensor(
        tensor,
        max_bit_width=max_bit_width,
        original_dtype=original_dtype,
    )
    return create_bitplane_artifact_from_quantized(
        quantized,
        model_id=model_id,
        block_id=block_id,
        tensor_name=tensor_name,
        checkpoint_ref=checkpoint_ref,
        compatibility=compatibility,
    )


def create_bitplane_artifact_from_quantized_values(
    values: NestedInt,
    *,
    model_id: str,
    block_id: str,
    tensor_name: str,
    max_bit_width: int = 8,
    original_dtype: str = "uint8",
    checkpoint_ref: str | None = None,
    compatibility: dict[str, Any] | None = None,
) -> BitPlaneArtifact:
    quantized = quantized_tensor_from_values(
        values,
        max_bit_width=max_bit_width,
        original_dtype=original_dtype,
    )
    return create_bitplane_artifact_from_quantized(
        quantized,
        model_id=model_id,
        block_id=block_id,
        tensor_name=tensor_name,
        checkpoint_ref=checkpoint_ref,
        compatibility=compatibility,
    )


def create_bitplane_artifact_from_quantized(
    quantized: QuantizedTensor,
    *,
    model_id: str,
    block_id: str,
    tensor_name: str,
    checkpoint_ref: str | None = None,
    compatibility: dict[str, Any] | None = None,
) -> BitPlaneArtifact:
    planes = decompose_to_bitplanes(
        quantized.values,
        max_bit_width=quantized.params.max_bit_width,
    )
    metadata_without_checksum = BitPlaneArtifactMetadata(
        model_id=_non_empty(model_id, "model_id"),
        block_id=_non_empty(block_id, "block_id"),
        tensor_name=_non_empty(tensor_name, "tensor_name"),
        original_shape=quantized.shape,
        original_dtype=quantized.original_dtype,
        max_bit_width=quantized.params.max_bit_width,
        available_planes=tuple(sorted(planes)),
        quantization=quantized.params,
        checkpoint_ref=checkpoint_ref,
        compatibility=compatibility or {},
    )
    checksum = artifact_checksum(planes, metadata_without_checksum)
    metadata = BitPlaneArtifactMetadata(
        **{
            **metadata_without_checksum.as_dict(),
            "original_shape": metadata_without_checksum.original_shape,
            "available_planes": metadata_without_checksum.available_planes,
            "quantization": metadata_without_checksum.quantization,
            "checksum": checksum,
            "compatibility": metadata_without_checksum.compatibility,
        }
    )
    artifact = BitPlaneArtifact(metadata=metadata, planes=planes)
    artifact.validate()
    return artifact


def decompose_to_bitplanes(
    quantized_values: NestedInt,
    *,
    max_bit_width: int,
) -> dict[int, NestedInt]:
    """Split unsigned integer values into LSB-indexed binary bit-planes."""

    _validate_bit_width(max_bit_width)
    shape = infer_shape(quantized_values)
    qmax = (1 << max_bit_width) - 1
    flat = flatten_tensor(quantized_values)
    for value in flat:
        if isinstance(value, bool) or not isinstance(value, int):
            raise BitPlaneError(
                "invalid_quantized_value",
                "bit-plane decomposition requires integer quantized values",
            )
        if value < 0 or value > qmax:
            raise BitPlaneError(
                "invalid_quantized_value",
                f"quantized value {value} is outside [0, {qmax}]",
            )

    planes: dict[int, NestedInt] = {}
    for plane_index in range(max_bit_width):
        plane_flat = [(value >> plane_index) & 1 for value in flat]
        planes[plane_index] = unflatten_tensor(plane_flat, shape)
    return planes


def reconstruct_quantized_from_planes(
    planes: dict[int, NestedInt],
    *,
    bit_width: int,
    max_bit_width: int,
) -> NestedInt:
    """Recombine the selected most-significant bit-planes."""

    selected = selected_msb_planes(bit_width, max_bit_width=max_bit_width)
    missing = [plane_index for plane_index in selected if plane_index not in planes]
    if missing:
        raise BitPlaneError(
            "missing_plane",
            f"missing required bit-planes: {missing}",
        )

    first_shape = infer_shape(planes[selected[0]])
    flat_accumulator = [0 for _ in flatten_tensor(planes[selected[0]])]
    for plane_index in selected:
        plane = planes[plane_index]
        if infer_shape(plane) != first_shape:
            raise BitPlaneError("shape_mismatch", "all planes must have the same shape")
        for value_index, bit in enumerate(flatten_tensor(plane)):
            if bit not in (0, 1):
                raise BitPlaneError("invalid_plane_value", "planes must contain 0 or 1")
            flat_accumulator[value_index] += bit << plane_index

    return unflatten_tensor(flat_accumulator, first_shape)


def reconstruct_weight(
    artifact: BitPlaneArtifact,
    *,
    bit_width: int,
    model_id: str | None = None,
    block_id: str | None = None,
    tensor_name: str | None = None,
) -> ReconstructionResult:
    """Validate and reconstruct a tensor at the requested effective bit-width."""

    artifact.validate()
    metadata = artifact.metadata
    _validate_compatibility(
        metadata,
        model_id=model_id,
        block_id=block_id,
        tensor_name=tensor_name,
    )
    quantized_values = reconstruct_quantized_from_planes(
        artifact.planes,
        bit_width=bit_width,
        max_bit_width=metadata.max_bit_width,
    )
    if infer_shape(quantized_values) != metadata.original_shape:
        raise BitPlaneError(
            "shape_mismatch",
            "reconstructed tensor shape does not match metadata",
        )
    return ReconstructionResult(
        bit_width=bit_width,
        selected_planes=selected_msb_planes(
            bit_width,
            max_bit_width=metadata.max_bit_width,
        ),
        quantized_values=quantized_values,
        values=dequantize_values(quantized_values, metadata.quantization),
        metadata=metadata,
    )


def selected_msb_planes(bit_width: int, *, max_bit_width: int) -> tuple[int, ...]:
    _validate_bit_width(max_bit_width)
    if isinstance(bit_width, bool) or not isinstance(bit_width, int):
        raise BitPlaneError("invalid_bit_width", "bit_width must be an integer")
    if bit_width <= 0 or bit_width > max_bit_width:
        raise BitPlaneError(
            "invalid_bit_width",
            "bit_width must be positive and <= max_bit_width",
        )
    return tuple(range(max_bit_width - bit_width, max_bit_width))


def artifact_checksum(
    planes: dict[int, NestedInt],
    metadata: BitPlaneArtifactMetadata,
) -> str:
    """Return a stable checksum over compatibility-critical artifact content."""

    payload = {
        "metadata": {
            **metadata.as_dict(),
            "checksum": None,
            "validation_status": metadata.validation_status,
        },
        "planes": {str(index): plane for index, plane in sorted(planes.items())},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_metadata(metadata: BitPlaneArtifactMetadata) -> None:
    _non_empty(metadata.model_id, "model_id")
    _non_empty(metadata.block_id, "block_id")
    _non_empty(metadata.tensor_name, "tensor_name")
    _validate_bit_width(metadata.max_bit_width)
    if metadata.quantization.max_bit_width != metadata.max_bit_width:
        raise BitPlaneError(
            "metadata_mismatch",
            "quantization max_bit_width must match artifact max_bit_width",
        )
    if metadata.reconstruction_policy != RECONSTRUCTION_POLICY:
        raise BitPlaneError(
            "unsupported_reconstruction_policy",
            f"unsupported reconstruction policy {metadata.reconstruction_policy}",
        )
    if metadata.artifact_version != ARTIFACT_VERSION:
        raise BitPlaneError(
            "unsupported_artifact_version",
            f"unsupported artifact version {metadata.artifact_version}",
        )
    if metadata.validation_status != "valid":
        raise BitPlaneError(
            "invalid_artifact_status",
            f"artifact validation_status is {metadata.validation_status}",
        )
    if not metadata.available_planes:
        raise BitPlaneError("missing_plane", "artifact must contain bit-planes")
    expected_planes = tuple(range(metadata.max_bit_width))
    if tuple(sorted(metadata.available_planes)) != expected_planes:
        raise BitPlaneError(
            "missing_plane",
            "artifact must contain all planes for its maximum bit-width",
        )


def _validate_plane_tensor(
    plane: NestedInt,
    *,
    plane_index: int,
    max_bit_width: int,
    expected_shape: tuple[int, ...],
) -> None:
    if plane_index < 0 or plane_index >= max_bit_width:
        raise BitPlaneError(
            "invalid_plane_index",
            f"plane index {plane_index} is outside max_bit_width {max_bit_width}",
        )
    if infer_shape(plane) != expected_shape:
        raise BitPlaneError("shape_mismatch", "plane shape does not match metadata")
    for bit in flatten_tensor(plane):
        if bit not in (0, 1):
            raise BitPlaneError("invalid_plane_value", "planes must contain 0 or 1")


def _validate_compatibility(
    metadata: BitPlaneArtifactMetadata,
    *,
    model_id: str | None,
    block_id: str | None,
    tensor_name: str | None,
) -> None:
    expected = {
        "model_id": model_id,
        "block_id": block_id,
        "tensor_name": tensor_name,
    }
    actual = {
        "model_id": metadata.model_id,
        "block_id": metadata.block_id,
        "tensor_name": metadata.tensor_name,
    }
    for key, expected_value in expected.items():
        if expected_value is not None and expected_value != actual[key]:
            raise BitPlaneError(
                "artifact_mismatch",
                f"{key} mismatch: expected {expected_value}, found {actual[key]}",
            )


def _validate_bit_width(max_bit_width: int) -> None:
    if isinstance(max_bit_width, bool) or not isinstance(max_bit_width, int):
        raise BitPlaneError("invalid_max_bit_width", "max_bit_width must be an integer")
    if max_bit_width <= 0:
        raise BitPlaneError("invalid_max_bit_width", "max_bit_width must be positive")


def _non_empty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BitPlaneError("invalid_metadata", f"{field} must be a non-empty string")
    return value


def _require_mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    raw = value.get(key)
    if not isinstance(raw, dict):
        raise BitPlaneError("invalid_artifact", f"{key} must be an object")
    return raw


def _require_string(value: dict[str, Any], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw:
        raise BitPlaneError("invalid_artifact", f"{key} must be a non-empty string")
    return raw


def _require_int(value: dict[str, Any], key: str) -> int:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise BitPlaneError("invalid_artifact", f"{key} must be an integer")
    return raw


def _require_int_list(value: dict[str, Any], key: str) -> list[int]:
    raw = value.get(key)
    if not isinstance(raw, list) or not raw:
        raise BitPlaneError("invalid_artifact", f"{key} must be a non-empty list")
    result: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise BitPlaneError("invalid_artifact", f"{key} must contain integers")
        result.append(item)
    return result
