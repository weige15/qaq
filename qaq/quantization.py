"""Small-tensor quantization helpers for bit-plane artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


NestedNumber = int | float | list["NestedNumber"]
NestedInt = int | list["NestedInt"]
NestedFloat = float | list["NestedFloat"]


@dataclass(slots=True)
class QuantizationError(ValueError):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class QuantizationParams:
    """Per-tensor affine quantization metadata."""

    scheme: str
    max_bit_width: int
    scale: float
    zero_point: int
    qmin: int
    qmax: int
    group_size: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "max_bit_width": self.max_bit_width,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "qmin": self.qmin,
            "qmax": self.qmax,
            "group_size": self.group_size,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "QuantizationParams":
        return cls(
            scheme=_require_string(value, "scheme"),
            max_bit_width=_require_int(value, "max_bit_width"),
            scale=_require_float(value, "scale"),
            zero_point=_require_int(value, "zero_point"),
            qmin=_require_int(value, "qmin"),
            qmax=_require_int(value, "qmax"),
            group_size=(
                _require_int(value, "group_size")
                if value.get("group_size") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class QuantizedTensor:
    """Quantized integer tensor plus metadata needed to dequantize it."""

    values: NestedInt
    shape: tuple[int, ...]
    original_dtype: str
    params: QuantizationParams

    def as_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "shape": list(self.shape),
            "original_dtype": self.original_dtype,
            "params": self.params.as_dict(),
        }


def quantize_tensor(
    tensor: NestedNumber,
    *,
    max_bit_width: int = 8,
    original_dtype: str = "float32",
) -> QuantizedTensor:
    """Quantize a small nested numeric tensor with per-tensor affine uints.

    The QAQ paper does not specify a real-valued quantization scheme. This
    prototype uses deterministic per-tensor affine unsigned quantization and
    records that choice in artifact metadata.
    """

    _validate_bit_width(max_bit_width)
    shape = infer_shape(tensor)
    flat = [float(value) for value in flatten_tensor(tensor)]
    if not flat:
        raise QuantizationError("empty_tensor", "tensor must contain at least one value")

    qmin = 0
    qmax = (1 << max_bit_width) - 1
    min_value = min(flat)
    max_value = max(flat)
    if min_value == max_value:
        scale = 1.0
        zero_point = _clip_int(round(-min_value), qmin, qmax)
    else:
        scale = (max_value - min_value) / float(qmax - qmin)
        zero_point = _clip_int(round(qmin - min_value / scale), qmin, qmax)

    quantized_flat = [
        _clip_int(round(value / scale + zero_point), qmin, qmax) for value in flat
    ]
    params = QuantizationParams(
        scheme="affine_uint_per_tensor",
        max_bit_width=max_bit_width,
        scale=scale,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
        group_size=None,
    )
    return QuantizedTensor(
        values=unflatten_tensor(quantized_flat, shape),
        shape=shape,
        original_dtype=original_dtype,
        params=params,
    )


def quantized_tensor_from_values(
    values: NestedInt,
    *,
    max_bit_width: int = 8,
    original_dtype: str = "uint8",
) -> QuantizedTensor:
    """Wrap an existing unsigned quantized representation for golden tests."""

    _validate_bit_width(max_bit_width)
    shape = infer_shape(values)
    qmin = 0
    qmax = (1 << max_bit_width) - 1
    for value in flatten_tensor(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise QuantizationError(
                "invalid_quantized_value",
                "quantized values must be integers",
            )
        if value < qmin or value > qmax:
            raise QuantizationError(
                "invalid_quantized_value",
                f"quantized value {value} is outside [{qmin}, {qmax}]",
            )

    params = QuantizationParams(
        scheme="uint_identity",
        max_bit_width=max_bit_width,
        scale=1.0,
        zero_point=0,
        qmin=qmin,
        qmax=qmax,
        group_size=None,
    )
    return QuantizedTensor(
        values=values,
        shape=shape,
        original_dtype=original_dtype,
        params=params,
    )


def dequantize_values(values: NestedInt, params: QuantizationParams) -> NestedFloat:
    """Dequantize nested integer values using stored affine parameters."""

    flat = []
    for value in flatten_tensor(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise QuantizationError(
                "invalid_quantized_value",
                "quantized values must be integers",
            )
        flat.append((value - params.zero_point) * params.scale)
    return unflatten_tensor(flat, infer_shape(values))


def infer_shape(tensor: NestedNumber) -> tuple[int, ...]:
    """Infer a rectangular nested-list tensor shape."""

    if isinstance(tensor, bool):
        raise QuantizationError("invalid_tensor", "boolean values are not valid tensors")
    if isinstance(tensor, int | float):
        return ()
    if not isinstance(tensor, list) or not tensor:
        raise QuantizationError(
            "invalid_tensor",
            "tensor must be a number or non-empty nested list",
        )

    child_shapes = [infer_shape(item) for item in tensor]
    first = child_shapes[0]
    if any(shape != first for shape in child_shapes):
        raise QuantizationError(
            "ragged_tensor",
            "nested tensor lists must be rectangular",
        )
    return (len(tensor), *first)


def flatten_tensor(tensor: NestedNumber) -> list[int | float]:
    """Flatten a nested-list tensor in row-major order."""

    if isinstance(tensor, bool):
        raise QuantizationError("invalid_tensor", "boolean values are not valid tensors")
    if isinstance(tensor, int | float):
        return [tensor]
    if not isinstance(tensor, list):
        raise QuantizationError("invalid_tensor", "tensor contains non-numeric value")

    flat: list[int | float] = []
    for item in tensor:
        flat.extend(flatten_tensor(item))
    return flat


def unflatten_tensor(values: list[int] | list[float], shape: tuple[int, ...]) -> Any:
    """Restore row-major flat values to a nested-list tensor shape."""

    expected = 1
    for dimension in shape:
        expected *= dimension
    if expected != len(values):
        raise QuantizationError(
            "shape_mismatch",
            f"shape {shape} expects {expected} values, got {len(values)}",
        )

    iterator = iter(values)

    def build(remaining_shape: tuple[int, ...]) -> Any:
        if not remaining_shape:
            return next(iterator)
        return [build(remaining_shape[1:]) for _ in range(remaining_shape[0])]

    return build(shape)


def _validate_bit_width(max_bit_width: int) -> None:
    if isinstance(max_bit_width, bool) or not isinstance(max_bit_width, int):
        raise QuantizationError(
            "invalid_max_bit_width",
            "max_bit_width must be an integer",
        )
    if max_bit_width <= 0:
        raise QuantizationError(
            "invalid_max_bit_width",
            "max_bit_width must be positive",
        )


def _clip_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _require_string(value: dict[str, Any], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw:
        raise QuantizationError("invalid_quantization_params", f"{key} must be a string")
    return raw


def _require_int(value: dict[str, Any], key: str) -> int:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise QuantizationError("invalid_quantization_params", f"{key} must be an int")
    return raw


def _require_float(value: dict[str, Any], key: str) -> float:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise QuantizationError(
            "invalid_quantization_params",
            f"{key} must be numeric",
        )
    return float(raw)
