import json

import pytest

from qaq.artifacts import load_bitplane_artifact, save_bitplane_artifact
from qaq.bitplanes import (
    BitPlaneArtifact,
    BitPlaneError,
    create_bitplane_artifact_from_quantized_values,
    reconstruct_weight,
)
from qaq.blocks import discover_mha_ffn_blocks
from tests.fixtures.fake_transformer import make_fake_transformer


def test_bitplane_artifact_save_load_roundtrip_reconstructs_precisions(tmp_path) -> None:
    block = discover_mha_ffn_blocks(make_fake_transformer(num_layers=1))[0]
    artifact = create_bitplane_artifact_from_quantized_values(
        [[0, 15, 16, 255], [170, 85, 240, 15]],
        model_id="fake-model",
        block_id=block.block_id,
        tensor_name=block.tensor_names[0],
        max_bit_width=8,
        checkpoint_ref="fake-checkpoint",
        compatibility={"block_granularity": "mha_ffn"},
    )

    path = save_bitplane_artifact(artifact, tmp_path / "artifact.json")
    loaded = load_bitplane_artifact(path)

    assert loaded.metadata == artifact.metadata
    assert loaded.planes == artifact.planes
    assert reconstruct_weight(loaded, bit_width=8).quantized_values == [
        [0, 15, 16, 255],
        [170, 85, 240, 15],
    ]
    assert reconstruct_weight(loaded, bit_width=4).quantized_values == [
        [0, 0, 16, 240],
        [160, 80, 240, 0],
    ]


def test_artifact_roundtrip_rejects_model_block_and_tensor_mismatches(tmp_path) -> None:
    block = discover_mha_ffn_blocks(make_fake_transformer(num_layers=1))[1]
    artifact = create_bitplane_artifact_from_quantized_values(
        [[0, 255]],
        model_id="fake-model",
        block_id=block.block_id,
        tensor_name=block.tensor_names[0],
    )
    loaded = load_bitplane_artifact(save_bitplane_artifact(artifact, tmp_path / "a.json"))

    for kwargs in (
        {"model_id": "wrong-model"},
        {"block_id": "layer_999.mha"},
        {"tensor_name": "wrong.weight"},
    ):
        with pytest.raises(BitPlaneError) as exc:
            reconstruct_weight(loaded, bit_width=8, **kwargs)
        assert exc.value.code == "artifact_mismatch"


def test_artifact_loader_rejects_checksum_mismatch(tmp_path) -> None:
    artifact = create_bitplane_artifact_from_quantized_values(
        [[0, 255]],
        model_id="fake-model",
        block_id="layer_000.mha",
        tensor_name="weight",
    )
    path = save_bitplane_artifact(artifact, tmp_path / "artifact.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["planes"]["7"] = [[0, 0]]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BitPlaneError) as exc:
        load_bitplane_artifact(path)

    assert exc.value.code == "checksum_mismatch"


def test_artifact_loader_rejects_missing_plane_metadata(tmp_path) -> None:
    artifact = create_bitplane_artifact_from_quantized_values(
        [[0, 255]],
        model_id="fake-model",
        block_id="layer_000.mha",
        tensor_name="weight",
    )
    payload = artifact.as_dict()
    del payload["planes"]["6"]
    path = tmp_path / "missing-plane.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BitPlaneError) as exc:
        BitPlaneArtifact.from_mapping(payload)

    assert exc.value.code == "plane_metadata_mismatch"
