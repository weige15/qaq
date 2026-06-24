import json
from pathlib import Path

import pytest

from qaq.config import RunConfig, load_config_file, main
from qaq.errors import ConfigValidationError
from qaq.manifest import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_STARTED,
    HardwareMetadata,
    create_run_manifest,
)


def base_config(tmp_path: Path) -> dict:
    return {
        "model": "fake-model",
        "tokenizer": "fake-tokenizer",
        "dataset": "toy_prompts",
        "split": "validation",
        "mode": "static_8bit",
        "precision_candidates": [4, 8],
        "max_bit_width": 8,
        "block_granularity": "mha_ffn",
        "gpu_ids": [0],
        "seed": 7,
        "output_dir": str(tmp_path / "run"),
        "overwrite": False,
        "logging": {
            "progress_interval_steps": 5,
            "checkpoint_interval_steps": 50,
            "console": False,
            "log_dir": str(tmp_path / "logs"),
        },
    }


def test_valid_config_creates_manifest(tmp_path: Path) -> None:
    config = RunConfig.from_mapping(base_config(tmp_path), available_gpu_count=8)
    hardware = HardwareMetadata(
        hostname="test-host",
        platform="test-platform",
        python_version="3.12.0",
        selected_gpu_ids=config.gpu_ids,
        detected_gpu_count=8,
    )

    manifest = create_run_manifest(
        config,
        run_id="test-run",
        hardware=hardware,
        artifact_paths={"metrics": "metrics.json"},
        started_at="2026-06-23T00:00:00+00:00",
    )

    assert manifest.status == STATUS_STARTED
    assert manifest.manifest_path.exists()

    data = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))
    assert data["run_id"] == "test-run"
    assert data["config"]["mode"] == "static_8bit"
    assert data["config"]["precision_candidates"] == [4, 8]
    assert data["hardware"]["selected_gpu_ids"] == [0]
    assert data["artifact_paths"] == {"metrics": "metrics.json"}
    assert data["status"] == STATUS_STARTED


def test_manifest_failure_writes_incomplete_marker(tmp_path: Path) -> None:
    config = RunConfig.from_mapping(base_config(tmp_path), available_gpu_count=8)
    manifest = create_run_manifest(config, run_id="test-run")

    manifest.mark_failed(code="controlled_failure", message="boom")

    data = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))
    assert data["status"] == STATUS_FAILED
    assert data["failure"] == {"code": "controlled_failure", "message": "boom"}
    marker = Path(data["incomplete_marker"])
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "controlled_failure: boom\n"


def test_manifest_completion_removes_incomplete_marker(tmp_path: Path) -> None:
    config = RunConfig.from_mapping(base_config(tmp_path), available_gpu_count=8)
    manifest = create_run_manifest(config, run_id="test-run")
    manifest.mark_failed(code="controlled_failure", message="boom")

    manifest.mark_completed(completed_at="2026-06-23T01:00:00+00:00")

    data = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))
    assert data["status"] == STATUS_COMPLETED
    assert data["failure"] is None
    assert data["incomplete_marker"] is None
    assert not (tmp_path / "run" / "INCOMPLETE").exists()


def test_manifest_completion_removes_stale_incomplete_marker(tmp_path: Path) -> None:
    config = RunConfig.from_mapping(base_config(tmp_path), available_gpu_count=8)
    stale_marker = tmp_path / "run" / "INCOMPLETE"
    stale_marker.parent.mkdir(parents=True, exist_ok=True)
    stale_marker.write_text("previous_failure: stale\n", encoding="utf-8")
    manifest = create_run_manifest(config, run_id="test-run")

    manifest.mark_completed(completed_at="2026-06-23T01:00:00+00:00")

    data = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))
    assert data["status"] == STATUS_COMPLETED
    assert data["failure"] is None
    assert data["incomplete_marker"] is None
    assert not stale_marker.exists()


def test_loads_json_config_fixture(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/configs/valid_static_8bit.json")
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    raw["output_dir"] = str(tmp_path / "from-fixture")
    local_fixture = tmp_path / "config.json"
    local_fixture.write_text(json.dumps(raw), encoding="utf-8")

    config = load_config_file(local_fixture, available_gpu_count=8)

    assert config.mode == "static_8bit"
    assert config.precision_candidates == (4, 8)
    assert config.logging.progress_interval_steps == 5


def test_loads_config_with_model_tokenizer_fallback() -> None:
    config = load_config_file(
        "configs/llama31_8b_first_milestone.json",
        validate_output=False,
    )

    assert config.use_model_tokenizer is True
    assert config.tokenizer == "meta-llama/Llama-3.1-8B"
    assert config.device == "cuda"
    assert config.metric == "perplexity"


def test_evaluator_config_fields_validate_and_serialize(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data.update(
        {
            "max_examples": 16,
            "eval_batch_size": 2,
            "collect_hidden_states": False,
            "store_full_logits": False,
            "hf_device_map": "auto",
            "hf_max_memory_per_gpu": "22GiB",
        }
    )

    config = RunConfig.from_mapping(data, available_gpu_count=8)
    payload = config.as_dict()

    assert config.max_examples == 16
    assert config.eval_batch_size == 2
    assert config.collect_hidden_states is False
    assert config.store_full_logits is False
    assert config.hf_device_map == "auto"
    assert config.hf_max_memory_per_gpu == "22GiB"
    assert payload["max_examples"] == 16
    assert payload["eval_batch_size"] == 2
    assert payload["hf_device_map"] == "auto"
    assert payload["hf_max_memory_per_gpu"] == "22GiB"


def test_loads_toml_config_stub(tmp_path: Path) -> None:
    config = load_config_file(
        "configs/smoke.toml",
        validate_output=False,
    )

    assert config.model == "fake-model"
    assert config.mode == "fp16"
    assert config.output_dir == Path("runs/smoke/fp16")


def test_config_cli_validates_config_stub() -> None:
    assert main(["configs/smoke.json", "--skip-output-dir-check"]) == 0


def test_llama_first_milestone_benchmark_configs_validate_structurally() -> None:
    config_paths = sorted(
        Path("configs/benchmarks/llama_first_milestone").glob("*/*.json")
    )

    assert len(config_paths) == 30
    for path in config_paths:
        config = load_config_file(path, validate_output=False)
        assert config.model == "meta-llama/Llama-3.1-8B"
        assert config.dataset != "fake_smoke"
        assert config.mode in {
            "fp16",
            "static_8bit",
            "static_4bit",
            "qaq_on_demand_off",
            "qaq_on_demand_on",
        }
        assert config.device == "cuda"
        assert config.use_model_tokenizer is True


def test_config_cli_returns_validation_exit_code_for_invalid_config() -> None:
    assert (
        main(
            [
                "tests/fixtures/configs/invalid_mode.json",
                "--skip-output-dir-check",
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("mode", "not_a_mode", "invalid_mode"),
        ("precision_candidates", [4, 4], "invalid_precision_candidates"),
        ("precision_candidates", [0, 8], "invalid_precision_candidates"),
        ("precision_candidates", [4, 16], "invalid_precision_candidates"),
        ("gpu_ids", [-1], "invalid_gpu_ids"),
        ("gpu_ids", [0, 0], "invalid_gpu_ids"),
        ("block_granularity", "global", "invalid_block_granularity"),
        ("eval_batch_size", 0, "invalid_eval_batch_size"),
        ("max_examples", 0, "invalid_integer"),
        ("collect_hidden_states", "no", "invalid_boolean"),
        ("store_full_logits", "no", "invalid_boolean"),
        ("hf_device_map", "ddp", "invalid_hf_device_map"),
    ],
)
def test_invalid_config_fields_fail(
    tmp_path: Path,
    field: str,
    value: object,
    code: str,
) -> None:
    data = base_config(tmp_path)
    data[field] = value

    with pytest.raises(ConfigValidationError) as exc:
        RunConfig.from_mapping(data, available_gpu_count=8)

    assert exc.value.code == code


def test_missing_required_field_fails(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    del data["model"]

    with pytest.raises(ConfigValidationError) as exc:
        RunConfig.from_mapping(data, available_gpu_count=8)

    assert exc.value.code == "missing_required_field"
    assert exc.value.field == "model"


def test_gpu_id_above_available_count_fails(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["gpu_ids"] = [8]

    with pytest.raises(ConfigValidationError) as exc:
        RunConfig.from_mapping(data, available_gpu_count=8)

    assert exc.value.code == "invalid_gpu_ids"


def test_existing_output_dir_requires_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    data = base_config(tmp_path)
    data["output_dir"] = str(output_dir)

    with pytest.raises(ConfigValidationError) as exc:
        RunConfig.from_mapping(data, available_gpu_count=8)

    assert exc.value.code == "unsafe_output_reuse"

    data["overwrite"] = True
    config = RunConfig.from_mapping(data, available_gpu_count=8)
    assert config.output_dir == output_dir


def test_static_mode_requires_matching_precision(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["precision_candidates"] = [4]

    with pytest.raises(ConfigValidationError) as exc:
        RunConfig.from_mapping(data, available_gpu_count=8)

    assert exc.value.code == "missing_mode_precision"


def test_qaq_mode_requires_router_checkpoint_or_diagnostic(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["mode"] = "qaq_on_demand_off"

    with pytest.raises(ConfigValidationError) as exc:
        RunConfig.from_mapping(data, available_gpu_count=8)

    assert exc.value.code == "missing_router_checkpoint"

    data["router_diagnostic"] = True
    config = RunConfig.from_mapping(data, available_gpu_count=8)
    assert config.router_diagnostic is True

    data["router_diagnostic"] = False
    data["router_checkpoint"] = str(tmp_path / "router.json")
    config = RunConfig.from_mapping(data, available_gpu_count=8)
    assert config.router_checkpoint == tmp_path / "router.json"


def test_qaq_mode_accepts_diagnostic_router_alias(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["mode"] = "qaq_on_demand_on"
    data["diagnostic_router"] = True

    config = RunConfig.from_mapping(data, available_gpu_count=8)

    assert config.router_diagnostic is True


def test_fixed_mixed_requires_fixed_profile(tmp_path: Path) -> None:
    data = base_config(tmp_path)
    data["mode"] = "fixed_mixed"

    with pytest.raises(ConfigValidationError) as exc:
        RunConfig.from_mapping(data, available_gpu_count=8)

    assert exc.value.code == "missing_fixed_profile"

    data["fixed_precision_by_block"] = {"layer_000.mha": 4, "layer_000.ffn": 8}
    config = RunConfig.from_mapping(data, available_gpu_count=8)
    assert config.fixed_precision_by_block == {"layer_000.mha": 4, "layer_000.ffn": 8}


def test_unsupported_config_format_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mode: fp16\n", encoding="utf-8")

    with pytest.raises(ConfigValidationError) as exc:
        load_config_file(config_path)

    assert exc.value.code == "unsupported_config_format"
