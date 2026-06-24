import json
from pathlib import Path

import pytest

from qaq.config import RunConfig
from qaq.logging import LogEvent, open_run_log, record_completion, record_failure
from qaq.manifest import STATUS_COMPLETED, STATUS_FAILED, create_run_manifest
from qaq.router.train import RouterTrainingConfig, RouterTrainingError, run_router_training
from qaq.router.types import DEFAULT_DECISION_POLICY
from qaq.status import EventType


def base_config(tmp_path: Path) -> RunConfig:
    return RunConfig.from_mapping(
        {
            "model": "fake-model",
            "tokenizer": "fake-tokenizer",
            "dataset": "toy_prompts",
            "split": "validation",
            "mode": "fp16",
            "precision_candidates": [4, 8],
            "max_bit_width": 8,
            "block_granularity": "mha_ffn",
            "device": "cpu",
            "gpu_ids": [],
            "seed": 0,
            "output_dir": str(tmp_path / "run"),
            "overwrite": False,
            "logging": {
                "progress_interval_steps": 1,
                "console": False,
                "log_dir": str(tmp_path / "logs"),
            },
        },
        available_gpu_count=0,
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_open_run_log_records_path_in_manifest_and_persists_events(tmp_path: Path) -> None:
    manifest = create_run_manifest(base_config(tmp_path), run_id="run-log")

    with open_run_log(manifest, name="train") as writer:
        writer.record(
            LogEvent.progress(
                run_id=manifest.run_id,
                module="router_train",
                step=1,
                loss=0.9,
            )
        )
        writer.record(
            LogEvent.warning(
                run_id=manifest.run_id,
                module="router_train",
                message="diagnostic router active",
            )
        )

    manifest_data = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))
    log_path = Path(manifest_data["artifact_paths"]["train_log"])
    events = read_jsonl(log_path)

    assert log_path.exists()
    assert events[0]["event_type"] == EventType.PROGRESS.value
    assert events[0]["step"] == 1
    assert events[0]["loss"] == 0.9
    assert events[1]["event_type"] == EventType.WARNING.value
    assert events[1]["message"] == "diagnostic router active"


def test_open_run_log_truncates_stale_log_for_overwrite_run(tmp_path: Path) -> None:
    config = RunConfig.from_mapping(
        {
            "model": "fake-model",
            "tokenizer": "fake-tokenizer",
            "dataset": "toy_prompts",
            "split": "validation",
            "mode": "fp16",
            "precision_candidates": [4, 8],
            "max_bit_width": 8,
            "block_granularity": "mha_ffn",
            "device": "cpu",
            "gpu_ids": [],
            "seed": 0,
            "output_dir": str(tmp_path / "run"),
            "overwrite": True,
            "logging": {
                "progress_interval_steps": 1,
                "console": False,
                "log_dir": str(tmp_path / "logs"),
            },
        },
        validate_output=False,
    )
    stale_log = tmp_path / "logs" / "train.jsonl"
    stale_log.parent.mkdir(parents=True)
    stale_log.write_text('{"event_type": "stale"}\n', encoding="utf-8")
    manifest = create_run_manifest(config, run_id="run-log")

    with open_run_log(manifest, name="train") as writer:
        writer.record(
            LogEvent.progress(
                run_id=manifest.run_id,
                module="router_train",
                step=1,
                loss=0.9,
            )
        )

    events = read_jsonl(stale_log)
    assert len(events) == 1
    assert events[0]["event_type"] == EventType.PROGRESS.value


def test_record_failure_writes_log_and_incomplete_marker(tmp_path: Path) -> None:
    manifest = create_run_manifest(base_config(tmp_path), run_id="run-failure")

    with open_run_log(manifest, name="eval") as writer:
        writer.record(
            LogEvent.progress(
                run_id=manifest.run_id,
                module="eval",
                processed_examples=2,
                total_examples=5,
            )
        )
        record_failure(
            manifest,
            writer,
            module="eval",
            code="controlled_failure",
            message="boom",
        )

    manifest_data = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))
    log_path = Path(manifest_data["artifact_paths"]["eval_log"])
    events = read_jsonl(log_path)

    assert manifest_data["status"] == STATUS_FAILED
    assert manifest_data["failure"] == {
        "code": "controlled_failure",
        "message": "boom",
    }
    marker = Path(manifest_data["incomplete_marker"])
    assert marker.exists()
    assert events[-1]["event_type"] == EventType.ERROR.value
    assert events[-1]["error_code"] == "controlled_failure"


def test_record_completion_updates_manifest(tmp_path: Path) -> None:
    manifest = create_run_manifest(base_config(tmp_path), run_id="run-complete")

    with open_run_log(manifest, name="eval") as writer:
        record_completion(manifest, writer, module="eval")

    manifest_data = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))
    log_path = Path(manifest_data["artifact_paths"]["eval_log"])
    events = read_jsonl(log_path)

    assert manifest_data["status"] == STATUS_COMPLETED
    assert manifest_data["failure"] is None
    assert manifest_data["incomplete_marker"] is None
    assert events[-1]["event_type"] == EventType.COMPLETION.value


def test_router_training_controlled_failure_writes_incomplete_marker(
    tmp_path: Path,
) -> None:
    config = RouterTrainingConfig.from_mapping(
        {
            "model": "tests/fixtures/models/router_local_model.json",
            "tokenizer": "tests/fixtures/tokenizers/router_local_tokenizer.json",
            "data_source": "tests/fixtures/benchmarks/router_training_real.jsonl",
            "split": "train",
            "validation_split": "validation",
            "teacher_model": "tests/fixtures/models/router_local_model.json",
            "student_model": "tests/fixtures/models/router_local_model.json",
            "student_quantized_path": "tests/fixtures/bitplanes/router_training_real",
            "distillation_loss": "router_cost_cross_entropy",
            "precision_candidates": [4, 8],
            "max_bit_width": 8,
            "block_granularity": "mha_ffn",
            "device": "cpu",
            "gpu_ids": [],
            "seed": 0,
            "output_dir": str(tmp_path / "router-failure"),
            "overwrite": False,
            "prompt_format": "question_answer_v1",
            "training_data_limit": 3,
            "validation_data_limit": 2,
            "checkpoint_interval_steps": 1,
            "diagnostic": True,
            "router": {
                "learning_rate": 0.05,
                "max_steps": 2,
                "temperature": 1.0,
                "target_temperature": 0.2,
                "bit_cost_weight": 0.04,
                "decision_policy": DEFAULT_DECISION_POLICY,
            },
            "logging": {
                "progress_interval_steps": 1,
                "checkpoint_interval_steps": 1,
                "console": False,
                "log_dir": str(tmp_path / "router-logs"),
            },
        }
    )

    with pytest.raises(RouterTrainingError) as exc:
        run_router_training(config, fail_at_step=1)

    manifest_data = json.loads(
        (tmp_path / "router-failure" / "manifest.json").read_text(encoding="utf-8")
    )
    log_path = Path(manifest_data["artifact_paths"]["router_train_log"])
    events = read_jsonl(log_path)

    assert exc.value.code == "controlled_training_failure"
    assert manifest_data["status"] == STATUS_FAILED
    assert manifest_data["config"]["mode"] == "fp16"
    assert manifest_data["config"]["router_diagnostic"] is True
    assert Path(manifest_data["incomplete_marker"]).exists()
    assert events[-1]["event_type"] == EventType.ERROR.value
    assert events[-1]["error_code"] == "controlled_training_failure"
