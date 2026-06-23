import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from qaq.artifacts import load_bitplane_artifact
from qaq.prepare_bitplanes import main, prepare_bitplane_artifacts
from qaq.router.train import RouterTrainingConfig, validate_router_training_preflight


def test_prepare_bitplanes_from_local_llama_safetensors_is_router_compatible(
    tmp_path: Path,
) -> None:
    model_dir, tokenizer_path = _write_local_llama_fixture(tmp_path)

    result = prepare_bitplane_artifacts(
        model=str(model_dir),
        tokenizer=str(tokenizer_path),
        output_dir=tmp_path / "prepared",
        sample_values=5,
    )

    assert result.block_count == 4
    assert len(result.artifact_records) == 4
    assert result.artifact_index_path.is_file()
    assert result.manifest_path.is_file()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_scope"] == "sampled_weight_values"
    assert manifest["accepted_as_full_quantized_inference_artifact"] is False

    first_record = result.artifact_records[0]
    artifact = load_bitplane_artifact(first_record.artifact_path)
    assert artifact.metadata.model_id == str(model_dir)
    assert artifact.metadata.block_id == "layer_000.mha"
    assert artifact.metadata.tensor_name == "model.layers.0.self_attn.q_proj.weight"
    assert artifact.metadata.original_shape == (5,)
    assert artifact.metadata.compatibility["source_tensor_shape"] == [4, 4]
    assert artifact.metadata.compatibility["sample_count"] == 5
    assert artifact.metadata.compatibility["full_tensor_values_stored"] is False

    index = json.loads(result.artifact_index_path.read_text(encoding="utf-8"))
    assert set(index) == {
        "layer_000.mha",
        "layer_000.ffn",
        "layer_001.mha",
        "layer_001.ffn",
    }
    assert set(index["layer_000.mha"]) == {"4", "8"}
    assert Path(index["layer_000.mha"]["4"]).is_absolute()

    training_config = RouterTrainingConfig.from_mapping(
        {
            "model": str(model_dir),
            "tokenizer": str(tokenizer_path),
            "data_source": "tests/fixtures/benchmarks/router_training_real.jsonl",
            "split": "train",
            "teacher_model": str(model_dir),
            "student_model": str(model_dir),
            "student_quantized_path": str(result.artifact_index_path),
            "distillation_loss": "router_cost_cross_entropy",
            "precision_candidates": [4, 8],
            "max_bit_width": 8,
            "block_granularity": "mha_ffn",
            "device": "cpu",
            "gpu_ids": [],
            "seed": 0,
            "output_dir": str(tmp_path / "router-train"),
            "prompt_format": "question_answer_v1",
            "training_data_limit": 1,
            "diagnostic": False,
            "logging": {"console": False},
        }
    )
    validate_router_training_preflight(training_config)


def test_prepare_bitplanes_cli_rejects_existing_output_without_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir, tokenizer_path = _write_local_llama_fixture(tmp_path)
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()

    exit_code = main(
        [
            "--model",
            str(model_dir),
            "--tokenizer",
            str(tokenizer_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 2
    assert "unsafe_output_reuse" in capsys.readouterr().err


def _write_local_llama_fixture(tmp_path: Path) -> tuple[Path, Path]:
    model_dir = tmp_path / "llama"
    model_dir.mkdir()
    tokenizer_path = tmp_path / "tokenizer.json"
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "num_hidden_layers": 2,
                "hidden_size": 4,
                "vocab_size": 32,
            }
        ),
        encoding="utf-8",
    )
    tokenizer_path.write_text(
        json.dumps(
            {
                "type": "fake_tokenizer",
                "tokenizer_id": "local-test-tokenizer",
                "model_max_length": 16,
            }
        ),
        encoding="utf-8",
    )

    tensors = {}
    for layer_index in range(2):
        base = float(layer_index * 100)
        tensors[f"model.layers.{layer_index}.self_attn.q_proj.weight"] = (
            torch.arange(16, dtype=torch.float32).reshape(4, 4) + base
        )
        tensors[f"model.layers.{layer_index}.mlp.gate_proj.weight"] = (
            torch.arange(16, dtype=torch.float32).reshape(4, 4) + base + 50.0
        )
    safetensors_torch.save_file(tensors, str(model_dir / "model-00001-of-00001.safetensors"))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 256},
                "weight_map": {
                    name: "model-00001-of-00001.safetensors"
                    for name in sorted(tensors)
                },
            }
        ),
        encoding="utf-8",
    )
    return model_dir, tokenizer_path
