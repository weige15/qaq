import json
from pathlib import Path

import pytest

from qaq.artifacts import save_bitplane_artifact
from qaq.bitplanes import create_bitplane_artifact_from_quantized_values
from qaq.blocks import discover_mha_ffn_blocks
from qaq.config import RunConfig
from qaq.evaluate import main as evaluate_main
from qaq.model_adapter import load_model_adapter
from qaq.precision_plan import PrecisionPlanError
from qaq.results import build_result_artifact
from qaq.runtime.common import RuntimeError
from qaq.runtime.static import run_static_runtime, validate_required_static_baselines


def _config(tmp_path: Path, mode: str, **overrides: object) -> RunConfig:
    data = {
        "model": "fake-qaq-smoke-model",
        "tokenizer": "fake-qaq-smoke-tokenizer",
        "dataset": "fake_smoke",
        "split": "validation",
        "mode": mode,
        "precision_candidates": [4, 8],
        "max_bit_width": 8,
        "block_granularity": "mha_ffn",
        "device": "cpu",
        "gpu_ids": [],
        "seed": 0,
        "output_dir": str(tmp_path / mode),
        "overwrite": False,
        "logging": {"console": False},
        "prompt_format": "fake_smoke_v1",
        "metric": "exact_match",
    }
    data.update(overrides)
    return RunConfig.from_mapping(data, validate_output=False)


def _artifact_refs(tmp_path: Path, config: RunConfig) -> dict[str, dict[str, str]]:
    adapter = load_model_adapter(config)
    blocks = discover_mha_ffn_blocks(
        adapter.architecture_metadata,
        supported_bit_widths=config.precision_candidates,
    )
    refs: dict[str, dict[str, str]] = {}
    for index, block in enumerate(blocks):
        artifact = create_bitplane_artifact_from_quantized_values(
            [[0, 15 + index], [170, 255 - index]],
            model_id=config.model,
            block_id=block.block_id,
            tensor_name=block.tensor_names[0],
            max_bit_width=config.max_bit_width,
            compatibility={"block_granularity": config.block_granularity},
        )
        path = save_bitplane_artifact(
            artifact,
            tmp_path / "artifacts" / f"{block.block_id}.json",
        )
        refs[block.block_id] = {"4": str(path), "8": str(path)}
    return refs


def test_all_8bit_fixed_profile_matches_static_8bit_outputs(tmp_path: Path) -> None:
    static_config = _config(tmp_path, "static_8bit")
    refs = _artifact_refs(tmp_path, static_config)
    fixed_config = _config(
        tmp_path,
        "fixed_mixed",
        fixed_precision_by_block={block_id: 8 for block_id in refs},
    )

    static_output = run_static_runtime(static_config, artifact_refs=refs)
    fixed_output = run_static_runtime(fixed_config, artifact_refs=refs)

    assert static_output.raw_output.logits == fixed_output.raw_output.logits
    assert static_output.raw_output.predictions == fixed_output.raw_output.predictions
    assert static_output.precision_plan.decisions == fixed_output.precision_plan.decisions
    assert static_output.reconstruction_records == fixed_output.reconstruction_records
    assert static_output.memory_events[0].as_dict()["peak_gpu_memory_gb"] == 0.0
    assert static_output.log_events[0]["processed_examples"] == 2


def test_all_4bit_fixed_profile_matches_static_4bit_outputs(tmp_path: Path) -> None:
    static_config = _config(tmp_path, "static_4bit")
    refs = _artifact_refs(tmp_path, static_config)
    fixed_config = _config(
        tmp_path,
        "fixed_mixed",
        fixed_precision_by_block={block_id: 4 for block_id in refs},
    )

    static_output = run_static_runtime(static_config, artifact_refs=refs)
    fixed_output = run_static_runtime(fixed_config, artifact_refs=refs)

    assert static_output.raw_output.logits == fixed_output.raw_output.logits
    assert static_output.precision_plan.decisions == fixed_output.precision_plan.decisions
    assert [record["selected_planes"] for record in static_output.reconstruction_records] == [
        [4, 5, 6, 7]
    ] * len(refs)


def test_static_quantized_modes_fail_when_artifacts_are_missing(tmp_path: Path) -> None:
    config = _config(tmp_path, "static_8bit")

    with pytest.raises(PrecisionPlanError) as exc:
        run_static_runtime(config)

    assert exc.value.code == "missing_artifact"


def test_static_runtime_rejects_unknown_artifact_blocks(tmp_path: Path) -> None:
    config = _config(tmp_path, "static_8bit")

    with pytest.raises(RuntimeError) as exc:
        run_static_runtime(config, artifact_refs={"layer_999.mha": {"8": "missing.json"}})

    assert exc.value.code == "unknown_artifact_block"


def test_runtime_output_is_json_serializable(tmp_path: Path) -> None:
    config = _config(tmp_path, "static_8bit")
    refs = _artifact_refs(tmp_path, config)

    output = run_static_runtime(config, artifact_refs=refs)

    payload = output.as_dict()
    assert payload["metadata"]["runtime_impl"] == "qaq.runtime.static.fake_cpu"
    assert payload["precision_plan"]["decision_source"] == "static"
    assert payload["latency_events"][0]["name"] == "end_to_end"
    json.dumps(payload, sort_keys=True)


def test_missing_static_baselines_reject_qaq_acceptance() -> None:
    validate_required_static_baselines({"fp16", "static_8bit", "static_4bit"})

    with pytest.raises(RuntimeError) as exc:
        validate_required_static_baselines({"fp16", "static_8bit"})

    assert exc.value.code == "missing_static_baseline"


def test_static_runtime_streams_multiple_micro_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, "fp16", eval_batch_size=1)
    adapter = load_model_adapter(config)
    batch_sizes: list[int] = []
    original_build_batch = adapter.build_batch

    def recording_build_batch(*args, **kwargs):
        batch = original_build_batch(*args, **kwargs)
        batch_sizes.append(batch.metadata.batch_size)
        if batch.metadata.batch_size > 1:
            raise AssertionError("runtime built a full validation batch")
        return batch

    adapter.build_batch = recording_build_batch
    monkeypatch.setattr("qaq.runtime.static.load_model_adapter", lambda _config: adapter)

    output = run_static_runtime(config)

    assert batch_sizes == [1, 1]
    assert output.raw_output.predictions and len(output.raw_output.predictions) == 2
    assert output.raw_output.logits == ((), ())
    assert output.raw_output.hidden_states.by_block == {}
    assert output.metadata["eval_batch_size"] == 1
    assert output.metadata["processed_examples"] == 2
    assert output.metadata["total_examples"] == 2
    assert output.metadata["micro_batch_count"] == 2
    assert output.metadata["peak_gpu_memory_gb"] == 0.0
    assert "model_device_map" in output.metadata


def test_max_examples_marks_subset_run_and_rejects_full_acceptance(tmp_path: Path) -> None:
    config = _config(tmp_path, "fp16", max_examples=1, eval_batch_size=1)

    output = run_static_runtime(config)
    artifact = build_result_artifact(config, output)

    assert output.metadata["processed_examples"] == 1
    assert output.metadata["total_examples"] == 2
    assert output.metadata["max_examples"] == 1
    assert output.metadata["subset_run"] is True
    assert artifact.accepted_as_qaq_result is False
    assert "benchmark_subset_not_full_acceptance" in artifact.rejection_reasons


def test_evaluate_cli_records_streaming_overrides(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path, "fp16")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")

    exit_code = evaluate_main(
        [
            "--config",
            str(config_path),
            "--skip-output-dir-check",
            "--max-examples",
            "1",
            "--eval-batch-size",
            "1",
            "--hf-device-map",
            "single",
            "--print-result-json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    runtime_metadata = payload["metadata"]["runtime_metadata"]
    assert runtime_metadata["cli_overrides"] == {
        "eval_batch_size": 1,
        "hf_device_map": "single",
        "max_examples": 1,
    }
    assert runtime_metadata["processed_examples"] == 1
    assert runtime_metadata["subset_run"] is True
