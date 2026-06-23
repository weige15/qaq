import json
from pathlib import Path

import pytest

from qaq.bitplanes import (
    BitPlaneError,
    create_bitplane_artifact,
    create_bitplane_artifact_from_quantized_values,
    decompose_to_bitplanes,
    reconstruct_quantized_from_planes,
    reconstruct_weight,
    selected_msb_planes,
)


GOLDEN = Path("tests/golden/bitplanes_u8.json")


def test_known_tensor_decomposes_and_reconstructs_from_msb_planes() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    values = golden["quantized_values"]

    planes = decompose_to_bitplanes(values, max_bit_width=8)

    assert planes[7] == golden["planes"]["7"]
    assert planes[4] == golden["planes"]["4"]
    assert reconstruct_quantized_from_planes(
        planes,
        bit_width=8,
        max_bit_width=8,
    ) == golden["expected_reconstruction"]["8"]
    assert reconstruct_quantized_from_planes(
        planes,
        bit_width=4,
        max_bit_width=8,
    ) == golden["expected_reconstruction"]["4"]
    assert selected_msb_planes(4, max_bit_width=8) == (4, 5, 6, 7)


def test_artifact_metadata_records_quantization_and_checksum() -> None:
    artifact = create_bitplane_artifact_from_quantized_values(
        [[0, 15], [170, 255]],
        model_id="fake-model",
        block_id="layer_000.mha",
        tensor_name="layers.0.mha.q_proj.weight",
        max_bit_width=8,
    )

    metadata = artifact.metadata
    assert metadata.model_id == "fake-model"
    assert metadata.block_id == "layer_000.mha"
    assert metadata.original_shape == (2, 2)
    assert metadata.original_dtype == "uint8"
    assert metadata.quantization.scheme == "uint_identity"
    assert metadata.reconstruction_policy == "msb_truncation"
    assert metadata.available_planes == tuple(range(8))
    assert metadata.checksum


def test_reconstruct_weight_validates_identity_and_shape() -> None:
    artifact = create_bitplane_artifact_from_quantized_values(
        [[0, 15], [170, 255]],
        model_id="fake-model",
        block_id="layer_000.mha",
        tensor_name="layers.0.mha.q_proj.weight",
        max_bit_width=8,
    )

    result_4 = reconstruct_weight(
        artifact,
        bit_width=4,
        model_id="fake-model",
        block_id="layer_000.mha",
        tensor_name="layers.0.mha.q_proj.weight",
    )
    result_8 = reconstruct_weight(artifact, bit_width=8)

    assert result_4.selected_planes == (4, 5, 6, 7)
    assert result_4.quantized_values == [[0, 0], [160, 240]]
    assert result_4.values == [[0.0, 0.0], [160.0, 240.0]]
    assert result_8.quantized_values == [[0, 15], [170, 255]]

    with pytest.raises(BitPlaneError) as exc:
        reconstruct_weight(artifact, bit_width=4, model_id="other-model")

    assert exc.value.code == "artifact_mismatch"


def test_float_tensor_quantization_preserves_shape_and_full_width_invariant() -> None:
    artifact = create_bitplane_artifact(
        [[-1.0, 0.0], [0.5, 1.0]],
        model_id="fake-model",
        block_id="layer_000.ffn",
        tensor_name="layers.0.ffn.down_proj.weight",
        max_bit_width=8,
    )

    reconstructed = reconstruct_weight(artifact, bit_width=8)

    assert artifact.metadata.original_shape == (2, 2)
    assert reconstructed.metadata.quantization.scheme == "affine_uint_per_tensor"
    assert reconstructed.quantized_values == reconstruct_quantized_from_planes(
        artifact.planes,
        bit_width=8,
        max_bit_width=8,
    )
    assert len(reconstructed.values) == 2
    assert len(reconstructed.values[0]) == 2


def test_invalid_bit_widths_and_missing_planes_fail_clearly() -> None:
    artifact = create_bitplane_artifact_from_quantized_values(
        [[0, 255]],
        model_id="fake-model",
        block_id="layer_000.mha",
        tensor_name="weight",
    )

    for bit_width in (0, 9):
        with pytest.raises(BitPlaneError) as exc:
            reconstruct_weight(artifact, bit_width=bit_width)
        assert exc.value.code == "invalid_bit_width"

    incomplete = {index: plane for index, plane in artifact.planes.items() if index != 7}
    with pytest.raises(BitPlaneError) as exc:
        reconstruct_quantized_from_planes(incomplete, bit_width=1, max_bit_width=8)

    assert exc.value.code == "missing_plane"


def test_invalid_quantized_values_fail_before_artifact_creation() -> None:
    with pytest.raises(BitPlaneError) as exc:
        decompose_to_bitplanes([[256]], max_bit_width=8)

    assert exc.value.code == "invalid_quantized_value"
