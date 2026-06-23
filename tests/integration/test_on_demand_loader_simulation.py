from pathlib import Path

import pytest

from qaq.artifacts import save_bitplane_artifact
from qaq.bitplanes import create_bitplane_artifact_from_quantized_values
from qaq.loader import LoaderRequest, OnDemandLoader


def _artifact(tmp_path: Path) -> Path:
    artifact = create_bitplane_artifact_from_quantized_values(
        [[0, 15], [170, 255]],
        model_id="fake-model",
        block_id="layer_000.mha",
        tensor_name="layers.0.mha.q_proj.weight",
    )
    return save_bitplane_artifact(artifact, tmp_path / "layer_000.mha.json")


def test_on_demand_loader_materializes_only_requested_planes_and_records_events(tmp_path: Path) -> None:
    path = _artifact(tmp_path)
    loader = OnDemandLoader(known_block_ids=("layer_000.mha",))
    request = LoaderRequest(
        request_id="query-0.layer_000.mha",
        query_id="query-0",
        block_id="layer_000.mha",
        bit_width=4,
        artifact_path=path,
        model_id="fake-model",
        tensor_name="layers.0.mha.q_proj.weight",
    )

    materialized = loader.load(request)

    assert materialized.selected_planes == (4, 5, 6, 7)
    assert set(materialized.plane_values) == {4, 5, 6, 7}
    assert materialized.reconstruction.quantized_values == [[0, 0], [160, 240]]
    assert materialized.bytes_loaded == 16
    assert loader.events[-1].status == "loaded"
    assert loader.events[-1].bytes_transferred == 16
    assert loader.events[-1].duration_seconds >= 0
    assert loader.summary().loads == 1
    assert loader.summary().current_resident_bytes == 16
    assert loader.summary().peak_resident_bytes == 16


def test_on_demand_loader_cache_hit_release_and_summary(tmp_path: Path) -> None:
    path = _artifact(tmp_path)
    loader = OnDemandLoader(known_block_ids=("layer_000.mha",))
    request = LoaderRequest(
        request_id="query-0.layer_000.mha",
        query_id="query-0",
        block_id="layer_000.mha",
        bit_width=8,
        artifact_path=path,
        model_id="fake-model",
    )

    first = loader.load(request)
    second = loader.load(request)
    release_event = loader.release(request)
    summary = loader.summary()

    assert second is first
    assert loader.events[0].status == "loaded"
    assert loader.events[1].status == "cache_hit"
    assert release_event.status == "released"
    assert summary.loads == 1
    assert summary.cache_hits == 1
    assert summary.releases == 1
    assert summary.failures == 0
    assert summary.total_bytes_transferred == 32
    assert summary.current_resident_bytes == 0
    assert summary.peak_resident_bytes == 32
    assert summary.target_devices == ("cpu",)


def test_loader_summary_is_machine_readable_for_on_demand_results(tmp_path: Path) -> None:
    path = _artifact(tmp_path)
    loader = OnDemandLoader(known_block_ids=("layer_000.mha",))
    loader.load(
        LoaderRequest(
            request_id="query-0.layer_000.mha",
            block_id="layer_000.mha",
            bit_width=4,
            artifact_path=path,
            model_id="fake-model",
        )
    )

    payload = loader.summary().as_dict()

    assert payload["loads"] == 1
    assert payload["total_bytes_transferred"] == 16
    assert payload["current_resident_bytes"] == 16
    assert payload["resident_entries"][0]["selected_planes"] == [4, 5, 6, 7]


def test_on_demand_loader_materializes_requested_planes_on_cuda_when_available(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this test environment")

    path = _artifact(tmp_path)
    loader = OnDemandLoader(known_block_ids=("layer_000.mha",))
    request = LoaderRequest(
        request_id="query-0.layer_000.mha.cuda",
        query_id="query-0",
        block_id="layer_000.mha",
        bit_width=4,
        artifact_path=path,
        target_device="cuda:0",
        model_id="fake-model",
        tensor_name="layers.0.mha.q_proj.weight",
    )

    materialized = loader.load(request)
    summary = loader.summary()

    assert materialized.target_device == "cuda:0"
    assert materialized.selected_planes == (4, 5, 6, 7)
    assert all(value.device.type == "cuda" for value in materialized.plane_values.values())
    assert all(value.dtype == torch.uint8 for value in materialized.plane_values.values())
    assert summary.loads == 1
    assert summary.target_devices == ("cuda:0",)
