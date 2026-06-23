"""Stable QAQ controlled-block discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BLOCK_TYPE_MHA = "mha"
BLOCK_TYPE_FFN = "ffn"
BLOCK_TYPES = frozenset({BLOCK_TYPE_MHA, BLOCK_TYPE_FFN})


@dataclass(slots=True)
class BlockRegistryError(ValueError):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class BlockDescriptor:
    """Stable descriptor for one QAQ-controlled transformer block."""

    block_id: str
    layer_index: int
    block_type: str
    module_path: str
    tensor_names: tuple[str, ...]
    supported_bit_widths: tuple[int, ...]
    artifact_refs: dict[str, str] = field(default_factory=dict)
    validation_status: str = "valid"

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "layer_index": self.layer_index,
            "block_type": self.block_type,
            "module_path": self.module_path,
            "tensor_names": list(self.tensor_names),
            "supported_bit_widths": list(self.supported_bit_widths),
            "artifact_refs": dict(self.artifact_refs),
            "validation_status": self.validation_status,
        }


def discover_mha_ffn_blocks(
    model: Any,
    *,
    supported_bit_widths: tuple[int, ...] = (4, 8),
) -> tuple[BlockDescriptor, ...]:
    """Discover MHA and FFN blocks from model metadata or a fake transformer."""

    bit_widths = _validate_supported_bit_widths(supported_bit_widths)
    layers = getattr(model, "layers", None)
    if not isinstance(layers, list | tuple) or not layers:
        raise BlockRegistryError(
            "unsupported_layout",
            "model must expose a non-empty layers sequence",
        )

    blocks: list[BlockDescriptor] = []
    for layer_index, layer in enumerate(layers):
        for block_type in (BLOCK_TYPE_MHA, BLOCK_TYPE_FFN):
            if not hasattr(layer, block_type):
                raise BlockRegistryError(
                    "unsupported_layout",
                    f"layer {layer_index} is missing {block_type} block",
                )
            module_path = f"layers.{layer_index}.{block_type}"
            blocks.append(
                BlockDescriptor(
                    block_id=f"layer_{layer_index:03d}.{block_type}",
                    layer_index=layer_index,
                    block_type=block_type,
                    module_path=module_path,
                    tensor_names=_tensor_names(getattr(layer, block_type), module_path),
                    supported_bit_widths=bit_widths,
                )
            )

    return tuple(blocks)


def block_map(blocks: tuple[BlockDescriptor, ...]) -> dict[str, BlockDescriptor]:
    """Return descriptors keyed by block ID, rejecting duplicates."""

    result: dict[str, BlockDescriptor] = {}
    for block in blocks:
        if block.block_id in result:
            raise BlockRegistryError(
                "duplicate_block_id",
                f"duplicate block_id {block.block_id}",
            )
        result[block.block_id] = block
    return result


def _tensor_names(block: Any, module_path: str) -> tuple[str, ...]:
    names = getattr(block, "tensor_names", None)
    if names is None:
        return (f"{module_path}.weight",)
    if not isinstance(names, list | tuple) or not names:
        raise BlockRegistryError(
            "unsupported_layout",
            f"{module_path} tensor_names must be a non-empty sequence",
        )
    tensor_names = tuple(names)
    if any(not isinstance(name, str) or not name for name in tensor_names):
        raise BlockRegistryError(
            "unsupported_layout",
            f"{module_path} tensor_names must contain non-empty strings",
        )
    return tensor_names


def _validate_supported_bit_widths(bit_widths: tuple[int, ...]) -> tuple[int, ...]:
    if not bit_widths:
        raise BlockRegistryError(
            "invalid_supported_bit_widths",
            "at least one supported bit-width is required",
        )
    if any(isinstance(bit_width, bool) or not isinstance(bit_width, int) for bit_width in bit_widths):
        raise BlockRegistryError(
            "invalid_supported_bit_widths",
            "supported bit-widths must be integers",
        )
    if any(bit_width <= 0 for bit_width in bit_widths):
        raise BlockRegistryError(
            "invalid_supported_bit_widths",
            "supported bit-widths must be positive",
        )
    if len(set(bit_widths)) != len(bit_widths):
        raise BlockRegistryError(
            "invalid_supported_bit_widths",
            "supported bit-widths must be unique",
        )
    return tuple(sorted(bit_widths))
