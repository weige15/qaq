"""Synchronous on-demand bit-plane loader and residency tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from qaq.artifacts import load_bitplane_artifact
from qaq.bitplanes import BitPlaneArtifact, BitPlaneError, ReconstructionResult, reconstruct_weight
from qaq.quantization import flatten_tensor
from qaq.tensor_bitplanes import (
    TensorBitPlaneArtifact,
    TensorBitPlaneError,
    is_tensor_bitplane_artifact_path,
    load_tensor_bitplane_artifact,
    pack_selected_msb_planes,
    reconstruct_tensor_weight,
)


@dataclass(slots=True)
class LoaderError(ValueError):
    """Raised when a loader request cannot be materialized."""

    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class LoaderRequest:
    """Request to materialize one block's selected precision from a CPU artifact."""

    request_id: str
    block_id: str
    bit_width: int
    artifact_path: Path
    target_device: str = "cpu"
    query_id: str | None = None
    model_id: str | None = None
    tensor_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query_id": self.query_id,
            "block_id": self.block_id,
            "bit_width": self.bit_width,
            "artifact_path": str(self.artifact_path),
            "target_device": self.target_device,
            "model_id": self.model_id,
            "tensor_name": self.tensor_name,
        }


@dataclass(frozen=True, slots=True)
class MaterializedTensor:
    """Selected bit-plane materialization returned by the loader."""

    request_id: str
    block_id: str
    bit_width: int
    target_device: str
    selected_planes: tuple[int, ...]
    plane_values: dict[int | str, Any]
    reconstruction: Any
    bytes_loaded: int
    artifact_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "block_id": self.block_id,
            "bit_width": self.bit_width,
            "target_device": self.target_device,
            "selected_planes": list(self.selected_planes),
            "plane_values": {
                str(index): _serializable_plane_value(value)
                for index, value in self.plane_values.items()
            },
            "reconstruction": self.reconstruction.as_dict(),
            "bytes_loaded": self.bytes_loaded,
            "artifact_path": self.artifact_path,
        }


@dataclass(frozen=True, slots=True)
class ResidencyRecord:
    """Current residency state for one materialized request key."""

    block_id: str
    bit_width: int
    target_device: str
    request_id: str
    artifact_path: str
    selected_planes: tuple[int, ...]
    bytes_resident: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "bit_width": self.bit_width,
            "target_device": self.target_device,
            "request_id": self.request_id,
            "artifact_path": self.artifact_path,
            "selected_planes": list(self.selected_planes),
            "bytes_resident": self.bytes_resident,
        }


@dataclass(frozen=True, slots=True)
class LoaderEvent:
    """Synchronous loader event for loads, cache hits, releases, or failures."""

    request_id: str
    block_id: str
    bit_width: int
    target_device: str
    status: str
    query_id: str | None = None
    selected_planes: tuple[int, ...] = ()
    source_residency: str = "cpu"
    duration_seconds: float = 0.0
    bytes_transferred: int = 0
    error_code: str | None = None
    message: str | None = None
    residency_state: str | None = None
    artifact_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query_id": self.query_id,
            "block_id": self.block_id,
            "bit_width": self.bit_width,
            "target_device": self.target_device,
            "status": self.status,
            "selected_planes": list(self.selected_planes),
            "source_residency": self.source_residency,
            "duration_seconds": self.duration_seconds,
            "bytes_transferred": self.bytes_transferred,
            "error_code": self.error_code,
            "message": self.message,
            "residency_state": self.residency_state,
            "artifact_path": self.artifact_path,
        }


@dataclass(frozen=True, slots=True)
class LoaderSummary:
    """Aggregate loader activity needed by on-demand result artifacts."""

    loads: int
    cache_hits: int
    releases: int
    failures: int
    total_bytes_transferred: int
    current_resident_bytes: int
    peak_resident_bytes: int
    total_transfer_seconds: float
    target_devices: tuple[str, ...]
    resident_entries: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "loads": self.loads,
            "cache_hits": self.cache_hits,
            "releases": self.releases,
            "failures": self.failures,
            "total_bytes_transferred": self.total_bytes_transferred,
            "current_resident_bytes": self.current_resident_bytes,
            "peak_resident_bytes": self.peak_resident_bytes,
            "total_transfer_seconds": self.total_transfer_seconds,
            "target_devices": list(self.target_devices),
            "resident_entries": list(self.resident_entries),
        }


class OnDemandLoader:
    """Synchronous small-tensor loader with explicit residency state."""

    def __init__(
        self,
        *,
        known_block_ids: tuple[str, ...] = (),
        max_resident_bytes: int | None = None,
    ) -> None:
        self.known_block_ids = known_block_ids
        self.max_resident_bytes = max_resident_bytes
        self._resident: dict[tuple[str, int, str, str], MaterializedTensor] = {}
        self._events: list[LoaderEvent] = []
        self._peak_resident_bytes = 0

    @property
    def events(self) -> tuple[LoaderEvent, ...]:
        return tuple(self._events)

    def load(self, request: LoaderRequest) -> MaterializedTensor:
        """Synchronously materialize selected bit-planes for a request."""

        start = perf_counter()
        try:
            validate_loader_request(request, known_block_ids=self.known_block_ids)
            cache_key = _cache_key(request)
            if cache_key in self._resident:
                materialized = self._resident[cache_key]
                self._record_event(
                    request,
                    status="cache_hit",
                    duration_seconds=perf_counter() - start,
                    selected_planes=materialized.selected_planes,
                    bytes_transferred=0,
                    residency_state="resident",
                )
                return materialized

            if is_tensor_bitplane_artifact_path(request.artifact_path):
                artifact = load_tensor_bitplane_artifact(request.artifact_path)
                reconstruction = reconstruct_tensor_weight(
                    artifact,
                    bit_width=request.bit_width,
                    model_id=request.model_id,
                    block_id=request.block_id,
                    tensor_name=request.tensor_name,
                )
                materialized = _materialize_from_tensor_artifact(
                    request,
                    artifact=artifact,
                    reconstruction=reconstruction,
                )
            else:
                artifact = load_bitplane_artifact(request.artifact_path)
                reconstruction = reconstruct_weight(
                    artifact,
                    bit_width=request.bit_width,
                    model_id=request.model_id,
                    block_id=request.block_id,
                    tensor_name=request.tensor_name,
                )
                materialized = _materialize_from_artifact(
                    request,
                    artifact=artifact,
                    reconstruction=reconstruction,
                )
            self._ensure_capacity(materialized.bytes_loaded)
            self._resident[cache_key] = materialized
            self._peak_resident_bytes = max(
                self._peak_resident_bytes,
                self.current_resident_bytes,
            )
            self._record_event(
                request,
                status="loaded",
                duration_seconds=perf_counter() - start,
                selected_planes=materialized.selected_planes,
                bytes_transferred=materialized.bytes_loaded,
                residency_state="resident",
            )
            return materialized
        except LoaderError as exc:
            self._record_failure(request, start=start, code=exc.code, message=exc.message)
            raise
        except BitPlaneError as exc:
            code = "missing_plane" if exc.code in {"missing_plane", "plane_metadata_mismatch"} else exc.code
            self._record_failure(request, start=start, code=code, message=exc.message)
            raise LoaderError(code, exc.message) from exc
        except TensorBitPlaneError as exc:
            self._record_failure(request, start=start, code=exc.code, message=exc.message)
            raise LoaderError(exc.code, exc.message) from exc

    def release(self, request: LoaderRequest) -> LoaderEvent:
        """Release a resident materialized tensor if present."""

        start = perf_counter()
        validate_loader_request(
            request,
            known_block_ids=self.known_block_ids,
            require_artifact_exists=False,
        )
        materialized = self._resident.pop(_cache_key(request), None)
        event = LoaderEvent(
            request_id=request.request_id,
            query_id=request.query_id,
            block_id=request.block_id,
            bit_width=request.bit_width,
            target_device=request.target_device,
            status="released" if materialized else "release_skipped",
            selected_planes=materialized.selected_planes if materialized else (),
            duration_seconds=perf_counter() - start,
            bytes_transferred=0,
            residency_state="released" if materialized else "not_resident",
            artifact_path=str(request.artifact_path),
        )
        self._events.append(event)
        return event

    @property
    def current_resident_bytes(self) -> int:
        return sum(item.bytes_loaded for item in self._resident.values())

    def residency_records(self) -> tuple[ResidencyRecord, ...]:
        return tuple(
            ResidencyRecord(
                block_id=item.block_id,
                bit_width=item.bit_width,
                target_device=item.target_device,
                request_id=item.request_id,
                artifact_path=item.artifact_path,
                selected_planes=item.selected_planes,
                bytes_resident=item.bytes_loaded,
            )
            for item in self._resident.values()
        )

    def summary(self) -> LoaderSummary:
        counts = {
            "loads": sum(event.status == "loaded" for event in self._events),
            "cache_hits": sum(event.status == "cache_hit" for event in self._events),
            "releases": sum(event.status == "released" for event in self._events),
            "failures": sum(event.status == "failed" for event in self._events),
        }
        transfer_events = [event for event in self._events if event.status == "loaded"]
        return LoaderSummary(
            loads=counts["loads"],
            cache_hits=counts["cache_hits"],
            releases=counts["releases"],
            failures=counts["failures"],
            total_bytes_transferred=sum(event.bytes_transferred for event in transfer_events),
            current_resident_bytes=self.current_resident_bytes,
            peak_resident_bytes=self._peak_resident_bytes,
            total_transfer_seconds=sum(event.duration_seconds for event in transfer_events),
            target_devices=tuple(sorted({event.target_device for event in self._events})),
            resident_entries=tuple(record.as_dict() for record in self.residency_records()),
        )

    def _ensure_capacity(self, incoming_bytes: int) -> None:
        if self.max_resident_bytes is None:
            return
        if self.current_resident_bytes + incoming_bytes > self.max_resident_bytes:
            raise LoaderError(
                "insufficient_memory",
                "materialized request would exceed max_resident_bytes",
            )

    def _record_event(
        self,
        request: LoaderRequest,
        *,
        status: str,
        duration_seconds: float,
        selected_planes: tuple[int, ...],
        bytes_transferred: int,
        residency_state: str,
    ) -> None:
        self._events.append(
            LoaderEvent(
                request_id=request.request_id,
                query_id=request.query_id,
                block_id=request.block_id,
                bit_width=request.bit_width,
                target_device=request.target_device,
                status=status,
                selected_planes=selected_planes,
                duration_seconds=duration_seconds,
                bytes_transferred=bytes_transferred,
                residency_state=residency_state,
                artifact_path=str(request.artifact_path),
            )
        )

    def _record_failure(
        self,
        request: LoaderRequest,
        *,
        start: float,
        code: str,
        message: str,
    ) -> None:
        self._events.append(
            LoaderEvent(
                request_id=request.request_id,
                query_id=request.query_id,
                block_id=request.block_id,
                bit_width=request.bit_width,
                target_device=request.target_device,
                status="failed",
                duration_seconds=perf_counter() - start,
                error_code=code,
                message=message,
                residency_state="failed",
                artifact_path=str(request.artifact_path),
            )
        )


def validate_loader_request(
    request: LoaderRequest,
    *,
    known_block_ids: tuple[str, ...] = (),
    require_artifact_exists: bool = True,
) -> None:
    if not isinstance(request.request_id, str) or not request.request_id:
        raise LoaderError("invalid_loader_request", "request_id must be a non-empty string")
    if not isinstance(request.block_id, str) or not request.block_id:
        raise LoaderError("invalid_loader_request", "block_id must be a non-empty string")
    if known_block_ids and request.block_id not in set(known_block_ids):
        raise LoaderError(
            "unknown_block_id",
            f"block_id {request.block_id} is not in the active block registry",
        )
    if isinstance(request.bit_width, bool) or not isinstance(request.bit_width, int):
        raise LoaderError("invalid_bit_width", "bit_width must be an integer")
    if request.bit_width <= 0:
        raise LoaderError("invalid_bit_width", "bit_width must be positive")
    _validate_target_device(request.target_device)
    if request.target_device.startswith("cuda"):
        _validate_cuda_device(request.target_device)
    if require_artifact_exists and not Path(request.artifact_path).is_file():
        raise LoaderError(
            "missing_cpu_artifact",
            f"CPU-resident artifact is missing: {request.artifact_path}",
        )


def _materialize_from_artifact(
    request: LoaderRequest,
    *,
    artifact: BitPlaneArtifact,
    reconstruction: ReconstructionResult,
) -> MaterializedTensor:
    selected_planes = reconstruction.selected_planes
    plane_values = {
        plane_index: _materialize_plane_value(
            artifact.planes[plane_index],
            target_device=request.target_device,
        )
        for plane_index in selected_planes
    }
    bytes_loaded = _estimate_plane_bytes(plane_values)
    return MaterializedTensor(
        request_id=request.request_id,
        block_id=request.block_id,
        bit_width=request.bit_width,
        target_device=request.target_device,
        selected_planes=selected_planes,
        plane_values=plane_values,
        reconstruction=reconstruction,
        bytes_loaded=bytes_loaded,
        artifact_path=str(request.artifact_path),
    )


def _materialize_from_tensor_artifact(
    request: LoaderRequest,
    *,
    artifact: TensorBitPlaneArtifact,
    reconstruction: Any,
) -> MaterializedTensor:
    packed, selected_planes = pack_selected_msb_planes(
        artifact,
        bit_width=request.bit_width,
        target_device=request.target_device,
    )
    plane_values = {"packed_selected_planes": packed}
    bytes_loaded = _estimate_plane_bytes(plane_values)
    return MaterializedTensor(
        request_id=request.request_id,
        block_id=request.block_id,
        bit_width=request.bit_width,
        target_device=request.target_device,
        selected_planes=selected_planes,
        plane_values=plane_values,
        reconstruction=reconstruction,
        bytes_loaded=bytes_loaded,
        artifact_path=str(request.artifact_path),
    )


def _estimate_plane_bytes(plane_values: dict[int | str, Any]) -> int:
    total = 0
    for plane in plane_values.values():
        if _is_torch_tensor(plane):
            total += int(plane.numel()) * int(plane.element_size())
        else:
            # CPU simulation counts each binary plane element as one transferred byte.
            total += len(flatten_tensor(plane))
    return total


def _materialize_plane_value(plane: Any, *, target_device: str) -> Any:
    if target_device == "cpu":
        return plane
    if target_device.startswith("cuda:"):
        torch = _import_torch_for_loader()
        return torch.tensor(plane, dtype=torch.uint8, device=torch.device(target_device))
    raise LoaderError(
        "invalid_target_device",
        "target_device must be 'cpu' or 'cuda:<non-negative-id>'",
    )


def _validate_target_device(target_device: str) -> None:
    if target_device == "cpu":
        return
    if target_device.startswith("cuda:"):
        suffix = target_device.removeprefix("cuda:")
        if suffix.isdigit():
            return
    raise LoaderError(
        "invalid_target_device",
        "target_device must be 'cpu' or 'cuda:<non-negative-id>'",
    )


def _validate_cuda_device(target_device: str) -> None:
    torch = _import_torch_for_loader()
    if not torch.cuda.is_available():
        raise LoaderError(
            "cuda_unavailable",
            "config requested CUDA on-demand loading but torch.cuda.is_available() is false",
        )
    gpu_id = int(target_device.removeprefix("cuda:"))
    if gpu_id >= torch.cuda.device_count():
        raise LoaderError(
            "cuda_unavailable",
            f"CUDA device {target_device} is not available",
        )


def _import_torch_for_loader() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise LoaderError(
            "torch_unavailable",
            "CUDA on-demand loading requires the optional torch package",
        ) from exc
    return torch


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "device") and hasattr(value, "numel")


def _serializable_plane_value(value: Any) -> Any:
    if _is_torch_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _cache_key(request: LoaderRequest) -> tuple[str, int, str, str, str | None, str | None]:
    return (
        request.block_id,
        request.bit_width,
        request.target_device,
        str(request.artifact_path),
        request.model_id,
        request.tensor_name,
    )
