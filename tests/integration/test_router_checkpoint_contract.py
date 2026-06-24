import json
from pathlib import Path

import pytest

from qaq.artifacts import save_bitplane_artifact
from qaq.bitplanes import create_bitplane_artifact_from_quantized_values
from qaq.blocks import discover_mha_ffn_blocks
from qaq.config import RunConfig, load_config_file
from qaq.data import export_router_training_jsonl
from qaq.evaluate import main as evaluate_main
from qaq.model_adapter import load_model_adapter
from qaq.router import train as router_train_module
from qaq.router.checkpoint import (
    RouterCheckpoint,
    load_router_checkpoint,
    save_router_checkpoint,
    validate_checkpoint_compatibility,
)
from qaq.router.train import (
    RouterTrainingConfig,
    RouterTrainingError,
    load_router_training_config,
    run_router_training,
    validate_router_training_preflight,
)
from qaq.router.types import (
    DEFAULT_DECISION_POLICY,
    RouterBlockParameters,
    RouterCheckpointMetadata,
    RouterPolicyError,
)
from qaq.status import RunStatus
from qaq.tensor_bitplanes import create_tensor_bitplane_artifact, save_tensor_bitplane_artifact


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig.from_mapping(
        {
            "model": "fake-qaq-smoke-model",
            "tokenizer": "fake-qaq-smoke-tokenizer",
            "dataset": "fake_smoke",
            "split": "validation",
            "mode": "qaq_on_demand_off",
            "precision_candidates": [4, 8],
            "max_bit_width": 8,
            "block_granularity": "mha_ffn",
            "device": "cpu",
            "gpu_ids": [],
            "seed": 0,
            "output_dir": str(tmp_path / "run"),
            "overwrite": False,
            "logging": {"console": False},
            "router_diagnostic": True,
        },
        validate_output=False,
    )


def _checkpoint(block_ids: tuple[str, ...]) -> RouterCheckpoint:
    return RouterCheckpoint(
        metadata=RouterCheckpointMetadata(
            checkpoint_id="checkpoint-contract",
            model_id="fake-qaq-smoke-model",
            block_ids=block_ids,
            candidate_bit_widths=(4, 8),
            feature_source="block_output_pooled",
            hidden_size=4,
            temperature=1.0,
            decision_policy=DEFAULT_DECISION_POLICY,
            max_bit_width=8,
        ),
        parameters={
            block_id: RouterBlockParameters(
                weights=((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
                bias=(0.0, 0.0),
            )
            for block_id in block_ids
        },
    )


def _write_training_resources(tmp_path: Path) -> dict[str, str]:
    root = tmp_path / "real_router_inputs"
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "router_model.json"
    tokenizer_path = root / "router_tokenizer.json"
    data_path = root / "router_data.jsonl"
    artifact_dir = root / "bitplanes"
    artifact_index = root / "artifact_index.json"

    model_path.write_text(
        json.dumps(
            {
                "type": "fake_causal_lm",
                "model_id": str(model_path),
                "num_layers": 2,
                "hidden_size": 4,
                "vocab_size": 8,
            }
        ),
        encoding="utf-8",
    )
    tokenizer_path.write_text(
        json.dumps(
            {
                "type": "fake_tokenizer",
                "tokenizer_id": str(tokenizer_path),
                "model_max_length": 128,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {"id": f"router-train-{index}", "split": "train", "text": f"Train prompt {index}", "target": f"train-{index}"}
        for index in range(3)
    ] + [
        {"id": f"router-validation-{index}", "split": "validation", "text": f"Validation prompt {index}", "target": f"validation-{index}"}
        for index in range(2)
    ]
    data_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    artifact_refs: dict[str, dict[str, str]] = {}
    for block_id, tensor_name in (
        ("layer_000.mha", "layers.0.mha.q_proj.weight"),
        ("layer_000.ffn", "layers.0.ffn.gate_proj.weight"),
        ("layer_001.mha", "layers.1.mha.q_proj.weight"),
        ("layer_001.ffn", "layers.1.ffn.gate_proj.weight"),
    ):
        artifact = create_bitplane_artifact_from_quantized_values(
            [[0, 255], [16, 240]],
            model_id=str(model_path),
            block_id=block_id,
            tensor_name=tensor_name,
            max_bit_width=8,
            checkpoint_ref="router-test",
            compatibility={"block_granularity": "mha_ffn"},
        )
        artifact_path = artifact_dir / f"{block_id}.json"
        save_bitplane_artifact(artifact, artifact_path)
        artifact_refs[block_id] = {"4": str(artifact_path), "8": str(artifact_path)}
    artifact_index.write_text(json.dumps(artifact_refs, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "model": str(model_path),
        "tokenizer": str(tokenizer_path),
        "data_source": str(data_path),
        "artifact_dir": str(artifact_dir),
        "artifact_index": str(artifact_index),
    }


def _training_mapping(tmp_path: Path) -> dict:
    resources = _write_training_resources(tmp_path)
    return {
        "model": resources["model"],
        "tokenizer": resources["tokenizer"],
        "data_source": resources["data_source"],
        "split": "train",
        "validation_split": "validation",
        "teacher_model": resources["model"],
        "student_model": resources["model"],
        "student_quantized_path": resources["artifact_dir"],
        "distillation_loss": "router_cost_cross_entropy",
        "precision_candidates": [4, 8],
        "max_bit_width": 8,
        "block_granularity": "mha_ffn",
        "device": "cpu",
        "gpu_ids": [],
        "seed": 0,
        "output_dir": str(tmp_path / "router-train"),
        "overwrite": False,
        "prompt_format": "question_answer_v1",
        "training_data_limit": 3,
        "validation_data_limit": 2,
        "checkpoint_interval_steps": 1,
        "diagnostic": False,
        "router": {
            "learning_rate": 0.05,
            "max_steps": 3,
            "temperature": 1.0,
            "target_temperature": 0.2,
            "bit_cost_weight": 0.04,
            "decision_policy": DEFAULT_DECISION_POLICY,
        },
        "logging": {
            "progress_interval_steps": 1,
            "checkpoint_interval_steps": 1,
            "console": False,
            "log_dir": str(tmp_path / "logs"),
        },
    }


def test_router_checkpoint_save_load_roundtrip_and_compatibility(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter = load_model_adapter(config)
    blocks = discover_mha_ffn_blocks(adapter.architecture_metadata)
    checkpoint = _checkpoint(tuple(block.block_id for block in blocks))

    path = save_router_checkpoint(checkpoint, tmp_path / "router.json")
    loaded = load_router_checkpoint(path)

    assert loaded == checkpoint
    validate_checkpoint_compatibility(
        loaded,
        blocks=blocks,
        model_id=config.model,
        candidate_bit_widths=config.precision_candidates,
        feature_source="block_output_pooled",
    )


def test_router_checkpoint_contract_rejects_candidate_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter = load_model_adapter(config)
    blocks = discover_mha_ffn_blocks(adapter.architecture_metadata)
    checkpoint = _checkpoint(tuple(block.block_id for block in blocks))

    with pytest.raises(RouterPolicyError) as exc:
        validate_checkpoint_compatibility(
            checkpoint,
            blocks=blocks,
            model_id=config.model,
            candidate_bit_widths=(4,),
            feature_source="block_output_pooled",
        )

    assert exc.value.code == "router_candidate_mismatch"


def test_router_checkpoint_contract_rejects_feature_source_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter = load_model_adapter(config)
    blocks = discover_mha_ffn_blocks(adapter.architecture_metadata)
    checkpoint = _checkpoint(tuple(block.block_id for block in blocks))

    with pytest.raises(RouterPolicyError) as exc:
        validate_checkpoint_compatibility(
            checkpoint,
            blocks=blocks,
            model_id=config.model,
            candidate_bit_widths=config.precision_candidates,
            feature_source="other_feature_source",
        )

    assert exc.value.code == "router_feature_source_mismatch"


def test_router_checkpoint_rejects_parameter_shape_mismatch() -> None:
    checkpoint = RouterCheckpoint(
        metadata=RouterCheckpointMetadata(
            checkpoint_id="bad",
            model_id="fake-qaq-smoke-model",
            block_ids=("layer_000.mha",),
            candidate_bit_widths=(4, 8),
            feature_source="block_output_pooled",
            hidden_size=4,
        ),
        parameters={
            "layer_000.mha": RouterBlockParameters(
                weights=((0.0, 0.0), (1.0, 0.0)),
                bias=(0.0, 0.0),
            )
        },
    )

    with pytest.raises(RouterPolicyError) as exc:
        checkpoint.validate()

    assert exc.value.code == "router_parameter_mismatch"


def test_router_training_real_data_acceptance_writes_reloadable_checkpoint_and_evaluates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    training_config = RouterTrainingConfig.from_mapping(_training_mapping(tmp_path))

    result = run_router_training(training_config)
    loaded = load_router_checkpoint(result.checkpoint_path)
    adapter = load_model_adapter(
        training_config.to_run_config(model=training_config.model, validate_output=False)
    )
    blocks = discover_mha_ffn_blocks(adapter.architecture_metadata)

    validate_checkpoint_compatibility(
        loaded,
        blocks=blocks,
        model_id=training_config.model,
        candidate_bit_widths=training_config.precision_candidates,
        feature_source="block_output_pooled",
    )

    assert result.manifest.status == "completed"
    manifest_data = json.loads(result.manifest.manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["config"]["mode"] == "fp16"
    assert manifest_data["config"]["router_diagnostic"] is False
    assert manifest_data["artifact_paths"]["router_targets"] == str(result.target_audit_path)
    assert result.progress.status == RunStatus.COMPLETED.value
    assert result.loss_records[-1].objective == "router_cost_cross_entropy"
    assert result.loss_records[-1].loss > 0
    assert result.validation_metrics["validation_loss"] > 0
    assert result.base_parameter_requires_grad
    assert all(requires_grad is False for requires_grad in result.base_parameter_requires_grad.values())
    assert loaded.metadata.training_metadata["completed_step"] == 3
    assert loaded.metadata.training_metadata["training_sample_count"] == 3
    assert loaded.metadata.training_metadata["validation_sample_count"] == 2
    assert loaded.metadata.training_metadata["target_record_count"] == 12
    assert loaded.metadata.training_metadata["distillation_loss"] == "router_cost_cross_entropy"
    assert loaded.metadata.training_metadata["diagnostic_training"] is False
    assert loaded.metadata.training_metadata["shared_teacher_student_reference"] is True
    assert loaded.metadata.training_metadata["reference_execution"] == "shared_teacher_student_forward"
    assert loaded.metadata.training_metadata["parameter_update_l2"] > 0
    assert loaded.metadata.training_metadata["latest_validation_metrics"]["validation_loss"] > 0
    assert any(
        any(value != 0.0 for row in params.weights for value in row)
        or any(bias != 0.0 for bias in params.bias)
        for params in loaded.parameters.values()
    )

    target_audit = json.loads(result.target_audit_path.read_text(encoding="utf-8"))
    assert target_audit["objective"] == "router_cost_cross_entropy"
    assert target_audit["diagnostic_training"] is False
    assert target_audit["shared_teacher_student_reference"] is True
    assert target_audit["training_data_source"] == training_config.data_source
    assert target_audit["training_sample_count"] == 3
    assert target_audit["validation_sample_count"] == 2
    assert target_audit["target_record_count"] == 12
    assert len(target_audit["training_targets"]) == 12
    assert target_audit["training_targets"][0]["example_id"] == "router-train-0"
    probabilities = target_audit["training_targets"][0]["target_probabilities"]
    assert set(probabilities) == {"4", "8"}
    assert sum(probabilities.values()) == pytest.approx(1.0)

    eval_config = RunConfig.from_mapping(
        {
            "model": training_config.model,
            "tokenizer": training_config.tokenizer,
            "dataset": training_config.data_source,
            "split": "validation",
            "mode": "qaq_on_demand_off",
            "precision_candidates": list(training_config.precision_candidates),
            "max_bit_width": training_config.max_bit_width,
            "block_granularity": training_config.block_granularity,
            "device": "cpu",
            "gpu_ids": [],
            "seed": training_config.seed,
            "output_dir": str(tmp_path / "router-eval"),
            "overwrite": True,
            "logging": {"console": False},
            "prompt_format": "question_answer_v1",
            "metric": "router_acceptance",
            "router_checkpoint": str(result.checkpoint_path),
            "router_diagnostic": False,
        },
        validate_output=False,
    )
    eval_config_path = tmp_path / "router-eval.json"
    eval_config_path.write_text(json.dumps(eval_config.as_dict()), encoding="utf-8")

    exit_code = evaluate_main(
        [
            "--config",
            str(eval_config_path),
            "--artifact-index",
            str(tmp_path / "real_router_inputs" / "artifact_index.json"),
            "--skip-output-dir-check",
            "--print-json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["mode"] == "qaq_on_demand_off"
    assert payload["status"] == "completed"
    assert payload["precision_plan"]["decision_source"] == "router"
    assert payload["metadata"]["router_checkpoint_loaded"] is True
    assert payload["metadata"]["router_checkpoint"] == str(result.checkpoint_path)
    assert payload["metadata"]["routing_summary"]["total_decisions"] == 8
    assert len(payload["metadata"]["precision_plans"]) == 2
    assert len(payload["metadata"]["routing_traces"]) == 8
    assert len(payload["reconstruction_records"]) == 8


def test_router_training_accepts_tensor_native_student_artifacts(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    data = _training_mapping(tmp_path)
    model = data["model"]
    config = RunConfig.from_mapping(
        {
            "model": model,
            "tokenizer": data["tokenizer"],
            "dataset": data["data_source"],
            "split": "train",
            "mode": "fp16",
            "precision_candidates": [4, 8],
            "max_bit_width": 8,
            "block_granularity": "mha_ffn",
            "device": "cpu",
            "gpu_ids": [],
            "seed": 0,
            "output_dir": str(tmp_path / "adapter"),
            "overwrite": True,
            "logging": {"console": False},
        },
        validate_output=False,
    )
    adapter = load_model_adapter(config)
    blocks = discover_mha_ffn_blocks(adapter.architecture_metadata)
    native_dir = tmp_path / "native-student-artifacts"
    for block_index, block in enumerate(blocks):
        tensor_name = block.tensor_names[0]
        tensor = (
            torch.arange(16, dtype=torch.float32).reshape(4, 4)
            + float(block_index * 3)
        )
        artifact = create_tensor_bitplane_artifact(
            tensor,
            model_id=model,
            block_id=block.block_id,
            tensor_name=tensor_name,
            original_dtype="F32",
        )
        save_tensor_bitplane_artifact(
            artifact,
            native_dir / f"{block.block_id}.qaq.safetensors",
        )

    data["student_quantized_path"] = str(native_dir)
    data["output_dir"] = str(tmp_path / "router-native-train")
    data["training_data_limit"] = 1
    data["validation_data_limit"] = 1
    data["router"] = {
        **data["router"],
        "max_steps": 1,
    }
    training_config = RouterTrainingConfig.from_mapping(data)

    result = run_router_training(training_config)
    loaded = load_router_checkpoint(result.checkpoint_path)
    target_audit = json.loads(result.target_audit_path.read_text(encoding="utf-8"))

    assert result.loss_records[-1].objective == "router_cost_cross_entropy"
    assert result.loss_records[-1].loss > 0
    assert result.validation_metrics["validation_loss"] > 0
    assert loaded.metadata.training_metadata["training_sample_count"] == 1
    assert loaded.metadata.training_metadata["validation_sample_count"] == 1
    assert loaded.metadata.training_metadata["target_record_count"] == len(blocks)
    assert loaded.metadata.training_metadata["parameter_update_l2"] > 0
    assert target_audit["objective"] == "router_cost_cross_entropy"
    assert target_audit["diagnostic_training"] is False
    assert target_audit["target_record_count"] == len(blocks)


def test_router_training_cuda_preflight_rejects_insufficient_llama_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "llama"
    model_dir.mkdir()
    tokenizer_path = tmp_path / "tokenizer.json"
    model_dir.joinpath("config.json").write_text(
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
    tokenizer_path.write_text(
        json.dumps(
            {
                "type": "fake_tokenizer",
                "tokenizer_id": "tiny-llama-tokenizer",
                "model_max_length": 16,
            }
        ),
        encoding="utf-8",
    )
    model_dir.joinpath("model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 16 * 1024**3}, "weight_map": {}}),
        encoding="utf-8",
    )
    data_path = tmp_path / "router_data.jsonl"
    data_path.write_text(
        json.dumps({"id": "train-0", "split": "train", "text": "Train prompt", "target": "target"}) + "\n",
        encoding="utf-8",
    )
    artifact_index = tmp_path / "artifact_index.json"
    artifact_index.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        router_train_module,
        "_cuda_memory_info",
        lambda gpu_id: router_train_module.CudaMemoryInfo(
            gpu_id=gpu_id,
            free_bytes=6 * 1024**3,
            total_bytes=6 * 1024**3,
        ),
    )
    training_config = RouterTrainingConfig.from_mapping(
        {
            "model": str(model_dir),
            "tokenizer": str(tokenizer_path),
            "data_source": str(data_path),
            "split": "train",
            "teacher_model": str(model_dir),
            "student_model": str(model_dir),
            "student_quantized_path": str(artifact_index),
            "distillation_loss": "router_cost_cross_entropy",
            "precision_candidates": [4, 8],
            "max_bit_width": 8,
            "block_granularity": "mha_ffn",
            "device": "cuda",
            "gpu_ids": [0],
            "seed": 0,
            "output_dir": str(tmp_path / "router-train"),
            "overwrite": True,
            "prompt_format": "question_answer_v1",
            "training_data_limit": 1,
            "diagnostic": False,
            "logging": {"console": False},
        }
    )

    with pytest.raises(RouterTrainingError) as exc:
        validate_router_training_preflight(training_config)

    assert exc.value.code == "insufficient_cuda_memory"
    assert "shared teacher/student model-weight loading path" in exc.value.message
    assert "teacher_model_weight=16.00 GiB" in exc.value.message
    assert "cuda:0 reports 6.00 GiB free" in exc.value.message


def test_router_training_cuda_preflight_counts_shared_llama_reference_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "llama"
    model_dir.mkdir()
    tokenizer_path = tmp_path / "tokenizer.json"
    model_dir.joinpath("config.json").write_text(
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
    tokenizer_path.write_text(
        json.dumps(
            {
                "type": "fake_tokenizer",
                "tokenizer_id": "tiny-llama-tokenizer",
                "model_max_length": 16,
            }
        ),
        encoding="utf-8",
    )
    model_dir.joinpath("model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 16 * 1024**3}, "weight_map": {}}),
        encoding="utf-8",
    )
    data_path = tmp_path / "router_data.jsonl"
    data_path.write_text(
        json.dumps({"id": "train-0", "split": "train", "text": "Train prompt", "target": "target"}) + "\n",
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"
    for block_id, tensor_name in (
        ("layer_000.mha", "model.layers.0.self_attn.q_proj.weight"),
        ("layer_000.ffn", "model.layers.0.mlp.gate_proj.weight"),
    ):
        artifact = create_bitplane_artifact_from_quantized_values(
            [[0, 255], [16, 240]],
            model_id=str(model_dir),
            block_id=block_id,
            tensor_name=tensor_name,
            max_bit_width=8,
            checkpoint_ref="shared-preflight-test",
            compatibility={"block_granularity": "mha_ffn"},
        )
        save_bitplane_artifact(artifact, artifact_dir / f"{block_id}.json")
    monkeypatch.setattr(
        router_train_module,
        "_cuda_memory_info",
        lambda gpu_id: router_train_module.CudaMemoryInfo(
            gpu_id=gpu_id,
            free_bytes=24 * 1024**3,
            total_bytes=24 * 1024**3,
        ),
    )
    training_config = RouterTrainingConfig.from_mapping(
        {
            "model": str(model_dir),
            "tokenizer": str(tokenizer_path),
            "data_source": str(data_path),
            "split": "train",
            "teacher_model": str(model_dir),
            "student_model": str(model_dir),
            "student_quantized_path": str(artifact_dir),
            "distillation_loss": "router_cost_cross_entropy",
            "precision_candidates": [4, 8],
            "max_bit_width": 8,
            "block_granularity": "mha_ffn",
            "device": "cuda",
            "gpu_ids": [0],
            "seed": 0,
            "output_dir": str(tmp_path / "router-train"),
            "overwrite": True,
            "prompt_format": "question_answer_v1",
            "training_data_limit": 1,
            "diagnostic": False,
            "logging": {"console": False},
        }
    )

    validate_router_training_preflight(training_config)


def test_router_training_preflight_rejects_missing_method_before_run(
    tmp_path: Path,
) -> None:
    data = _training_mapping(tmp_path)
    data["distillation_loss"] = None
    training_config = RouterTrainingConfig.from_mapping(data)

    with pytest.raises(RouterTrainingError) as exc:
        validate_router_training_preflight(training_config)

    assert exc.value.code == "missing_distillation_loss"
    assert not training_config.output_dir.exists()


def test_export_router_training_jsonl_writes_real_hellaswag_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_root = tmp_path / "benchmarks"
    hellaswag_dir = benchmark_root / "hellaswag"
    hellaswag_dir.mkdir(parents=True)
    train_row = {
        "ind": 1,
        "split": "train",
        "ctx": "A person opens a box and",
        "endings": ["takes out a tool.", "falls asleep.", "paints the sky.", "eats a shoe."],
        "label": "0",
    }
    validation_row = {
        "ind": 2,
        "split": "validation",
        "ctx": "A person turns on the stove and",
        "endings": ["starts cooking.", "drives away.", "opens an umbrella.", "throws a ball."],
        "label": "0",
    }
    (hellaswag_dir / "train.jsonl").write_text(json.dumps(train_row) + "\n", encoding="utf-8")
    (hellaswag_dir / "validation.jsonl").write_text(
        json.dumps(validation_row) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QAQ_BENCHMARK_DATA_ROOT", str(benchmark_root))
    monkeypatch.setenv("QAQ_DISABLE_HF_DATASETS", "1")

    output = tmp_path / "hellaswag_router_train.jsonl"
    result = export_router_training_jsonl(
        dataset="hellaswag",
        output_path=output,
        overwrite=True,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert result["split_counts"] == {"train": 1, "validation": 1}
    assert [row["split"] for row in rows] == ["train", "validation"]
    assert all({"id", "split", "text", "target"} <= set(row) for row in rows)
    assert rows[0]["target"] == "takes out a tool."
    assert rows[0]["metadata"]["real_benchmark"] is True
    assert rows[0]["metadata"]["benchmark_name"] == "hellaswag"


def test_router_training_non_diagnostic_rejects_prohibited_inputs(tmp_path: Path) -> None:
    base = _training_mapping(tmp_path)
    cases = (
        ("data_source", "fake_smoke"),
        ("data_source", "tests/fixtures/benchmarks/router_training_real.jsonl"),
        ("student_quantized_path", "runs/llama31_8b_bitplanes_sampled/artifact_index.json"),
        ("model", "fake-qaq-smoke-model"),
    )
    for field, value in cases:
        data = {**base, field: value}
        if field == "model":
            data["teacher_model"] = value
            data["student_model"] = value
        config = RouterTrainingConfig.from_mapping(data)
        with pytest.raises(RouterTrainingError) as exc:
            validate_router_training_preflight(config)
        assert exc.value.code == "prohibited_non_diagnostic_training_input"


def test_router_training_non_diagnostic_rejects_nonaccepted_runtime_index(
    tmp_path: Path,
) -> None:
    data = _training_mapping(tmp_path)
    artifact_dir = tmp_path / "full_tensor_probe"
    artifact_dir.mkdir()
    runtime_index = artifact_dir / "runtime_artifact_index.json"
    runtime_index.write_text("{}\n", encoding="utf-8")
    (artifact_dir / "generation_manifest.json").write_text(
        json.dumps(
            {
                "model": data["student_model"],
                "accepted_as_full_quantized_inference_artifact": False,
                "full_tensor_runtime_coverage": False,
                "runtime_index_artifact_ref_mode": "partial_tensor_index",
                "sampled_or_truncated_probe": True,
                "artifact_format": "safetensors",
                "max_elements_per_tensor": 16,
                "tensor_limit_per_block": 1,
            }
        ),
        encoding="utf-8",
    )
    data["student_quantized_path"] = str(runtime_index)
    config = RouterTrainingConfig.from_mapping(data)

    with pytest.raises(RouterTrainingError) as exc:
        validate_router_training_preflight(config)

    assert exc.value.code == "non_diagnostic_requires_full_tensor_artifacts"
    assert "full tensor-native" in exc.value.message


def test_llama31_full_hellaswag_router_config_matches_qaq_checkpoint_path() -> None:
    training_config = load_router_training_config(
        "configs/router_train_llama31_8b_full_hellaswag.yaml"
    )

    assert training_config.model == "meta-llama/Llama-3.1-8B"
    assert training_config.tokenizer == "meta-llama/Llama-3.1-8B"
    assert training_config.teacher_model == "meta-llama/Llama-3.1-8B"
    assert training_config.student_model == "meta-llama/Llama-3.1-8B"
    assert training_config.student_quantized_path == Path(
        "runs/llama31_8b_full_tensor_bitplanes/runtime_artifact_index.json"
    )
    assert training_config.distillation_loss == "router_cost_cross_entropy"
    assert training_config.precision_candidates == (4, 8)
    assert training_config.max_bit_width == 8
    assert training_config.block_granularity == "mha_ffn"
    assert training_config.device == "cuda"
    assert training_config.diagnostic is False
    assert training_config.hf_device_map == "auto"
    assert training_config.resolved_checkpoint_dir / "router_final.json" == Path(
        "runs/llama_first_milestone/router/checkpoints/router_final.json"
    )

    for mode in ("qaq_on_demand_off", "qaq_on_demand_on"):
        run_config = load_config_file(
            f"configs/benchmarks/llama_first_milestone/hellaswag/{mode}.json",
            validate_output=False,
        )
        assert run_config.router_checkpoint == training_config.resolved_checkpoint_dir / "router_final.json"
        assert run_config.precision_candidates == training_config.precision_candidates
        assert run_config.max_bit_width == training_config.max_bit_width
        assert run_config.block_granularity == training_config.block_granularity


def test_router_training_real_yaml_config_loads() -> None:
    training_config = load_router_training_config("configs/router_train_real.yaml")

    assert training_config.data_source == "tests/fixtures/benchmarks/router_training_real.jsonl"
    assert training_config.validation_split == "validation"
    assert training_config.distillation_loss == "router_cost_cross_entropy"
    assert training_config.router.max_steps == 3
    assert training_config.diagnostic is False
