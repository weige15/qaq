import json
from pathlib import Path

import pytest

from qaq.artifacts import load_bitplane_artifact
from qaq.llama_bitplanes import generate_llama_bitplane_artifacts
from qaq.tensor_bitplanes import load_tensor_bitplane_artifact


def test_llama_bitplane_generator_streams_local_safetensors(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
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
    safetensors_torch.save_file(
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
        snapshot / "model.safetensors",
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
    assert set(tensor_index) == {"layer_000.mha"}
    assert set(tensor_index["layer_000.mha"]) == {
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    }
    assert set(runtime_index["layer_000.mha"]) == {"4", "8"}

    artifact = load_bitplane_artifact(result.records[0].artifact_path)
    compatibility = artifact.metadata.compatibility or {}
    assert artifact.metadata.block_id == "layer_000.mha"
    assert compatibility["framework"] == "transformers_llama"
    assert compatibility["llama_compatible"] is True
    assert compatibility["source_tensor_shape"] == [4, 4]
    assert compatibility["artifact_element_count"] == 8
    assert compatibility["truncated_source_tensor"] is True


def test_llama_bitplane_generator_writes_tensor_native_safetensors(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
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
    safetensors_torch.save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.arange(
                16,
                dtype=torch.float32,
            ).reshape(4, 4),
        },
        snapshot / "model.safetensors",
    )

    result = generate_llama_bitplane_artifacts(
        model=str(snapshot),
        tokenizer=str(tokenizer),
        output_dir=tmp_path / "native-artifacts",
        block_limit=1,
        tensor_limit_per_block=1,
        max_elements_per_tensor=8,
        artifact_format="safetensors",
        overwrite=False,
    )

    assert result.artifact_format == "safetensors"
    assert len(result.records) == 1
    record = result.records[0]
    assert record.artifact_path.name.endswith(".qaq.safetensors")
    assert record.storage_layout == "packed_uint8_bitplanes"
    assert record.truncated is True

    artifact = load_tensor_bitplane_artifact(record.artifact_path)
    compatibility = artifact.metadata.compatibility or {}
    assert artifact.metadata.block_id == "layer_000.mha"
    assert compatibility["framework"] == "transformers_llama"
    assert compatibility["source_tensor_shape"] == [4, 4]
    assert compatibility["artifact_element_count"] == 8
    assert compatibility["storage_layout"] == "packed_uint8_bitplanes"
    assert compatibility["truncated_source_tensor"] is True

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    tensor_index = json.loads(result.tensor_index_path.read_text(encoding="utf-8"))
    runtime_index = json.loads(result.runtime_index_path.read_text(encoding="utf-8"))
    assert manifest["artifact_format"] == "safetensors"
    assert manifest["records"][0]["storage_layout"] == "packed_uint8_bitplanes"
    assert tensor_index["layer_000.mha"][record.tensor_name].endswith(".qaq.safetensors")
    assert runtime_index["layer_000.mha"]["4"].endswith(".qaq.safetensors")
