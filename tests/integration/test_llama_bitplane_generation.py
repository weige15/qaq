import json
from pathlib import Path

import pytest

from qaq.artifacts import load_bitplane_artifact
from qaq.blocks import BLOCK_TYPE_FFN, BLOCK_TYPE_MHA, BlockDescriptor
from qaq.llama_bitplanes import (
    LlamaBitPlaneGenerationError,
    generate_llama_bitplane_artifacts,
)
from qaq.runtime.weight_overrides import artifact_ref_mode
from qaq.tensor_bitplanes import load_tensor_bitplane_artifact


MHA_TENSORS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
)
FFN_TENSORS = (
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
)


def _write_tiny_llama_snapshot(
    tmp_path: Path,
    tensors: dict[str, object],
) -> tuple[Path, Path]:
    safetensors_torch = pytest.importorskip("safetensors.torch")
    snapshot = tmp_path / "llama-snapshot"
    snapshot.mkdir()
    tokenizer = tmp_path / "tokenizer.json"
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "num_hidden_layers": 1,
                "hidden_size": 4,
                "vocab_size": 16,
            }
        ),
        encoding="utf-8",
    )
    tokenizer.write_text(
        json.dumps(
            {
                "type": "fake_tokenizer",
                "tokenizer_id": "tiny-llama-tokenizer",
                "model_max_length": 16,
            }
        ),
        encoding="utf-8",
    )
    safetensors_torch.save_file(tensors, snapshot / "model.safetensors")
    return snapshot, tokenizer


def _full_layer_tensors() -> dict[str, object]:
    torch = pytest.importorskip("torch")
    tensors: dict[str, object] = {}
    for index, tensor_name in enumerate((*MHA_TENSORS, *FFN_TENSORS)):
        tensors[tensor_name] = torch.arange(
            index * 16,
            (index + 1) * 16,
            dtype=torch.float32,
        ).reshape(4, 4)
    return tensors


def _runtime_blocks(runtime_index: dict[str, dict[str, str]]) -> tuple[BlockDescriptor, ...]:
    return (
        BlockDescriptor(
            block_id="layer_000.mha",
            layer_index=0,
            block_type=BLOCK_TYPE_MHA,
            module_path="layers.0.mha",
            tensor_names=MHA_TENSORS,
            supported_bit_widths=(4, 8),
            artifact_refs=dict(runtime_index["layer_000.mha"]),
        ),
        BlockDescriptor(
            block_id="layer_000.ffn",
            layer_index=0,
            block_type=BLOCK_TYPE_FFN,
            module_path="layers.0.ffn",
            tensor_names=FFN_TENSORS,
            supported_bit_widths=(4, 8),
            artifact_refs=dict(runtime_index["layer_000.ffn"]),
        ),
    )


def test_llama_bitplane_generator_marks_sampled_runtime_index_non_accepted(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    snapshot, tokenizer = _write_tiny_llama_snapshot(
        tmp_path,
        {
            "model.layers.0.self_attn.q_proj.weight": torch.arange(
                16,
                dtype=torch.float32,
            ).reshape(4, 4),
            "model.layers.0.self_attn.k_proj.weight": torch.arange(
                16,
                32,
                dtype=torch.float32,
            ).reshape(4, 4),
        },
    )

    result = generate_llama_bitplane_artifacts(
        model=str(snapshot),
        tokenizer=str(tokenizer),
        output_dir=tmp_path / "artifacts",
        block_limit=1,
        tensor_limit_per_block=2,
        max_elements_per_tensor=8,
        overwrite=False,
    )

    assert len(result.records) == 2
    assert result.manifest_path.is_file()
    assert result.tensor_index_path.is_file()
    assert result.runtime_index_path.is_file()
    assert all(record.truncated for record in result.records)

    tensor_index = json.loads(result.tensor_index_path.read_text(encoding="utf-8"))
    runtime_index = json.loads(result.runtime_index_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    expected_tensors = {
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    }
    assert set(tensor_index) == {"layer_000.mha"}
    assert set(tensor_index["layer_000.mha"]) == expected_tensors
    assert set(runtime_index["layer_000.mha"]) == expected_tensors
    assert "4" not in runtime_index["layer_000.mha"]
    assert "8" not in runtime_index["layer_000.mha"]

    assert manifest["runtime_index_artifact_ref_mode"] == "partial_tensor_index"
    assert manifest["partial_tensor_index"] is True
    assert manifest["full_tensor_runtime_coverage"] is False
    assert manifest["sampled_or_truncated_probe"] is True
    assert manifest["diagnostic_generation_requested"] is True
    assert manifest["accepted_as_full_quantized_inference_artifact"] is False
    assert manifest["artifact_acceptance_status"] == "partial_tensor_index"

    artifact = load_bitplane_artifact(result.records[0].artifact_path)
    compatibility = artifact.metadata.compatibility or {}
    assert artifact.metadata.block_id == "layer_000.mha"
    assert compatibility["framework"] == "transformers_llama"
    assert compatibility["llama_compatible"] is True
    assert compatibility["source_tensor_shape"] == [4, 4]
    assert compatibility["artifact_element_count"] == 8
    assert compatibility["truncated_source_tensor"] is True
    assert compatibility["accepted_as_full_quantized_inference_artifact"] is False


def test_llama_bitplane_generator_writes_full_tensor_native_runtime_index(
    tmp_path: Path,
) -> None:
    snapshot, tokenizer = _write_tiny_llama_snapshot(tmp_path, _full_layer_tensors())

    result = generate_llama_bitplane_artifacts(
        model=str(snapshot),
        tokenizer=str(tokenizer),
        output_dir=tmp_path / "native-artifacts",
        max_elements_per_tensor=None,
        artifact_format="safetensors",
        require_full_runtime_coverage=True,
        overwrite=False,
    )

    assert result.artifact_format == "safetensors"
    assert len(result.records) == 7
    assert all(not record.truncated for record in result.records)
    assert all(record.artifact_path.name.endswith(".qaq.safetensors") for record in result.records)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    runtime_index = json.loads(result.runtime_index_path.read_text(encoding="utf-8"))
    assert set(runtime_index) == {"layer_000.mha", "layer_000.ffn"}
    assert set(runtime_index["layer_000.mha"]) == set(MHA_TENSORS)
    assert set(runtime_index["layer_000.ffn"]) == set(FFN_TENSORS)
    assert all(not key.isdigit() for refs in runtime_index.values() for key in refs)
    assert all(path.endswith(".qaq.safetensors") for refs in runtime_index.values() for path in refs.values())
    assert artifact_ref_mode(_runtime_blocks(runtime_index)) == "full_tensor_index"

    assert manifest["artifact_format"] == "safetensors"
    assert manifest["runtime_index_artifact_ref_mode"] == "full_tensor_index"
    assert manifest["full_tensor_runtime_coverage"] is True
    assert manifest["all_discovered_blocks_covered"] is True
    assert manifest["sampled_or_truncated_probe"] is False
    assert manifest["partial_tensor_index"] is False
    assert manifest["full_tensor_native_runtime_artifacts"] is True
    assert manifest["accepted_as_full_quantized_inference_artifact"] is True
    assert manifest["artifact_acceptance_status"] == "full_tensor_native_accepted_inference_artifact"

    artifact = load_tensor_bitplane_artifact(result.records[0].artifact_path)
    compatibility = artifact.metadata.compatibility or {}
    assert artifact.metadata.block_id in {"layer_000.mha", "layer_000.ffn"}
    assert compatibility["framework"] == "transformers_llama"
    assert compatibility["source_tensor_shape"] == [4, 4]
    assert compatibility["artifact_element_count"] == 16
    assert compatibility["storage_layout"] == "packed_uint8_bitplanes"
    assert compatibility["truncated_source_tensor"] is False
    assert compatibility["accepted_as_full_quantized_inference_artifact"] is True


def test_llama_bitplane_generator_rejects_incomplete_full_runtime_request(
    tmp_path: Path,
) -> None:
    snapshot, tokenizer = _write_tiny_llama_snapshot(tmp_path, _full_layer_tensors())

    with pytest.raises(LlamaBitPlaneGenerationError) as exc:
        generate_llama_bitplane_artifacts(
            model=str(snapshot),
            tokenizer=str(tokenizer),
            output_dir=tmp_path / "incomplete-full-request",
            block_limit=1,
            tensor_limit_per_block=1,
            max_elements_per_tensor=None,
            artifact_format="safetensors",
            require_full_runtime_coverage=True,
            overwrite=False,
        )

    assert exc.value.code == "incomplete_tensor_artifact_index"

    partial = generate_llama_bitplane_artifacts(
        model=str(snapshot),
        tokenizer=str(tokenizer),
        output_dir=tmp_path / "partial-diagnostic-request",
        block_limit=1,
        tensor_limit_per_block=1,
        max_elements_per_tensor=None,
        artifact_format="safetensors",
        overwrite=False,
    )
    manifest = json.loads(partial.manifest_path.read_text(encoding="utf-8"))
    runtime_index = json.loads(partial.runtime_index_path.read_text(encoding="utf-8"))

    assert manifest["runtime_index_artifact_ref_mode"] == "partial_tensor_index"
    assert manifest["partial_tensor_index"] is True
    assert manifest["full_tensor_runtime_coverage"] is False
    assert manifest["diagnostic_generation_requested"] is True
    assert manifest["accepted_as_full_quantized_inference_artifact"] is False
    assert manifest["artifact_acceptance_status"] == "partial_tensor_index"
    assert set(runtime_index["layer_000.mha"]) == {"model.layers.0.self_attn.q_proj.weight"}
    assert "4" not in runtime_index["layer_000.mha"]
    assert "8" not in runtime_index["layer_000.mha"]
