from pathlib import Path

import pytest

from qaq.loader import LoaderRequest, OnDemandLoader
from qaq.tensor_bitplanes import (
    create_tensor_bitplane_artifact,
    load_tensor_bitplane_artifact,
    normalized_tensor_reconstruction_delta,
    reconstruct_tensor_weight,
    save_tensor_bitplane_artifact,
)


def test_tensor_native_artifact_roundtrip_and_loader_materialization(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    artifact = create_tensor_bitplane_artifact(
        tensor,
        model_id="fake-qaq-smoke-model",
        block_id="layer_000.mha",
        tensor_name="layers.0.mha.q_proj.weight",
        original_dtype="F32",
    )
    path = save_tensor_bitplane_artifact(
        artifact,
        tmp_path / "layer_000.mha.q_proj.qaq.safetensors",
    )

    loaded = load_tensor_bitplane_artifact(path)
    reconstructed = reconstruct_tensor_weight(
        loaded,
        bit_width=4,
        model_id="fake-qaq-smoke-model",
        block_id="layer_000.mha",
        tensor_name="layers.0.mha.q_proj.weight",
    )
    assert loaded.metadata.artifact_version == "qaq.tensor_bitplane.v1"
    assert loaded.metadata.compatibility["storage_layout"] == "packed_uint8_bitplanes"
    assert tuple(reconstructed.quantized_values.shape) == (4, 4)
    assert reconstructed.selected_planes == (4, 5, 6, 7)
    assert normalized_tensor_reconstruction_delta(
        loaded,
        bit_width=4,
        model_id="fake-qaq-smoke-model",
        block_id="layer_000.mha",
    ) >= 0.0

    loader = OnDemandLoader(known_block_ids=("layer_000.mha",))
    materialized = loader.load(
        LoaderRequest(
            request_id="native-cpu-load",
            block_id="layer_000.mha",
            bit_width=4,
            artifact_path=path,
            target_device="cpu",
            model_id="fake-qaq-smoke-model",
            tensor_name="layers.0.mha.q_proj.weight",
        )
    )

    assert materialized.selected_planes == (4, 5, 6, 7)
    assert set(materialized.plane_values) == {"packed_selected_planes"}
    packed = materialized.plane_values["packed_selected_planes"]
    assert str(packed.device) == "cpu"
    assert int(packed.numel()) == 8
    assert materialized.bytes_loaded == 8
    assert loader.summary().total_bytes_transferred == 8


def test_tensor_native_loader_materializes_packed_planes_on_cuda_when_available(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    tensor = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    artifact = create_tensor_bitplane_artifact(
        tensor,
        model_id="fake-qaq-smoke-model",
        block_id="layer_000.mha",
        tensor_name="layers.0.mha.q_proj.weight",
    )
    path = save_tensor_bitplane_artifact(
        artifact,
        tmp_path / "layer_000.mha.q_proj.qaq.safetensors",
    )

    loader = OnDemandLoader(known_block_ids=("layer_000.mha",))
    materialized = loader.load(
        LoaderRequest(
            request_id="native-cuda-load",
            block_id="layer_000.mha",
            bit_width=4,
            artifact_path=path,
            target_device="cuda:0",
            model_id="fake-qaq-smoke-model",
            tensor_name="layers.0.mha.q_proj.weight",
        )
    )

    packed = materialized.plane_values["packed_selected_planes"]
    assert packed.is_cuda
    assert int(packed.numel()) == 16
    assert materialized.bytes_loaded == 16
