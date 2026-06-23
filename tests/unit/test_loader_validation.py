import json
from pathlib import Path

import pytest

from qaq.artifacts import save_bitplane_artifact
from qaq.bitplanes import create_bitplane_artifact_from_quantized_values
from qaq.loader import LoaderError, LoaderRequest, OnDemandLoader, validate_loader_request


def _artifact(tmp_path: Path, *, block_id: str = "layer_000.mha") -> Path:
    artifact = create_bitplane_artifact_from_quantized_values(
        [[0, 15], [170, 255]],
        model_id="fake-model",
        block_id=block_id,
        tensor_name="layers.0.mha.q_proj.weight",
    )
    return save_bitplane_artifact(artifact, tmp_path / f"{block_id}.json")


def test_validate_loader_request_rejects_unknown_block_and_missing_artifact(tmp_path: Path) -> None:
    missing = LoaderRequest(
        request_id="load-1",
        block_id="layer_000.mha",
        bit_width=4,
        artifact_path=tmp_path / "missing.json",
    )

    with pytest.raises(LoaderError) as block_exc:
        validate_loader_request(missing, known_block_ids=("layer_000.ffn",))
    assert block_exc.value.code == "unknown_block_id"

    with pytest.raises(LoaderError) as artifact_exc:
        validate_loader_request(missing, known_block_ids=("layer_000.mha",))
    assert artifact_exc.value.code == "missing_cpu_artifact"


def test_validate_loader_request_rejects_invalid_bit_width_and_device(tmp_path: Path) -> None:
    path = _artifact(tmp_path)

    with pytest.raises(LoaderError) as bit_exc:
        validate_loader_request(
            LoaderRequest(
                request_id="load-1",
                block_id="layer_000.mha",
                bit_width=0,
                artifact_path=path,
            )
        )
    assert bit_exc.value.code == "invalid_bit_width"

    with pytest.raises(LoaderError) as device_exc:
        validate_loader_request(
            LoaderRequest(
                request_id="load-1",
                block_id="layer_000.mha",
                bit_width=4,
                artifact_path=path,
                target_device="gpu0",
            )
        )
    assert device_exc.value.code == "invalid_target_device"


def test_cuda_request_fails_only_when_cuda_is_unavailable(tmp_path: Path) -> None:
    path = _artifact(tmp_path)
    request = LoaderRequest(
        request_id="load-1",
        block_id="layer_000.mha",
        bit_width=4,
        artifact_path=path,
        target_device="cuda:0",
    )

    if _cuda_available():
        validate_loader_request(request)
        return

    with pytest.raises(LoaderError) as exc:
        validate_loader_request(request)

    assert exc.value.code == "cuda_unavailable"


def test_loader_records_failure_for_insufficient_memory(tmp_path: Path) -> None:
    path = _artifact(tmp_path)
    loader = OnDemandLoader(
        known_block_ids=("layer_000.mha",),
        max_resident_bytes=1,
    )

    with pytest.raises(LoaderError) as exc:
        loader.load(
            LoaderRequest(
                request_id="load-1",
                block_id="layer_000.mha",
                bit_width=4,
                artifact_path=path,
                model_id="fake-model",
            )
        )

    assert exc.value.code == "insufficient_memory"
    assert loader.events[-1].status == "failed"
    assert loader.summary().failures == 1


def test_loader_maps_missing_plane_artifact_to_clear_failure(tmp_path: Path) -> None:
    path = _artifact(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["planes"]["7"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    loader = OnDemandLoader(known_block_ids=("layer_000.mha",))

    with pytest.raises(LoaderError) as exc:
        loader.load(
            LoaderRequest(
                request_id="load-1",
                block_id="layer_000.mha",
                bit_width=1,
                artifact_path=path,
                model_id="fake-model",
            )
        )

    assert exc.value.code == "missing_plane"
    assert loader.events[-1].error_code == "missing_plane"


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
