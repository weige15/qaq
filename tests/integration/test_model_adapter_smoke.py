import json
import math
from pathlib import Path

import pytest

from qaq.benchmark_adapter import build_tokenized_batch
from qaq.config import RunConfig
from qaq.data import BenchmarkDataError, load_benchmark_examples
from qaq.model_adapter import (
    FakeTokenizer,
    HuggingFaceCausalLMAdapter,
    ModelAdapterError,
    load_model_adapter,
    main as model_adapter_main,
    verify_model_adapter_config,
)
from qaq.blocks import discover_mha_ffn_blocks


def _config(tmp_path: Path, **overrides: object) -> RunConfig:
    data = {
        "model": "fake-qaq-smoke-model",
        "tokenizer": "fake-qaq-smoke-tokenizer",
        "dataset": "fake_smoke",
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
        "logging": {"console": False},
        "prompt_format": "fake_smoke_v1",
        "metric": "exact_match",
    }
    data.update(overrides)
    return RunConfig.from_mapping(data, validate_output=False)


def test_fake_adapter_reference_pass_returns_outputs_and_hidden_features(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter = load_model_adapter(config)
    adapter.freeze_base_model()

    examples = adapter.load_examples(config)
    batch = adapter.build_batch(config, examples)
    blocks = discover_mha_ffn_blocks(
        adapter.architecture_metadata,
        supported_bit_widths=config.precision_candidates,
    )
    output = adapter.reference_forward(
        batch,
        block_ids=tuple(block.block_id for block in blocks),
    )

    assert all(parameter.requires_grad is False for parameter in adapter.parameters())
    assert batch.metadata.as_dict() == {
        "dataset": "fake_smoke",
        "split": "validation",
        "prompt_format": "fake_smoke_v1",
        "tokenizer": "fake-qaq-smoke-tokenizer",
        "batch_size": 2,
        "max_length": 128,
        "context_length_policy": "truncate",
        "truncated_examples": [],
        "example_ids": ["fake-smoke-0", "fake-smoke-1"],
    }
    assert len(output.logits) == len(examples)
    assert len(output.losses) == len(examples)
    assert output.metadata["precision"] == "fp16_reference"
    assert output.metadata["feature_source"] == "block_output_pooled"
    assert output.metadata["adapter_kind"] == "fake_adapter"
    assert output.metadata["model_source"] == "fake_adapter"
    assert output.metadata["model_is_fake"] is True
    assert output.metadata["tokenizer_is_fake"] is True
    assert output.metadata["dataset_is_fake"] is True
    assert output.metadata["fixture_only_data"] is False
    assert output.metadata["benchmark_is_real"] is False
    assert output.metadata["diagnostic"] is True
    assert output.metadata["context_length_policy"] == "truncate"
    assert output.metadata["selected_gpu_ids"] == []
    assert set(output.hidden_states.by_block) == {block.block_id for block in blocks}
    assert all(len(vectors) == len(examples) for vectors in output.hidden_states.by_block.values())
    assert all(
        len(feature_vector) == adapter.hidden_size
        for vectors in output.hidden_states.by_block.values()
        for feature_vector in vectors
    )


def test_benchmark_fixture_tokenization_records_context_metadata(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        dataset="tests/fixtures/benchmarks/fake_smoke.jsonl",
        prompt_format="plain",
    )
    adapter = load_model_adapter(config)
    examples = load_benchmark_examples(config.dataset, split=config.split)

    batch = build_tokenized_batch(
        config,
        examples,
        adapter.tokenizer,
        max_length=12,
        context_length_policy="truncate",
    )

    assert batch.example_ids == ("fixture-0", "fixture-1")
    assert batch.metadata.max_length == 12
    assert batch.metadata.context_length_policy == "truncate"
    assert batch.metadata.truncated_examples == ("fixture-0", "fixture-1")
    assert len({len(row) for row in batch.input_ids}) == 1
    assert batch.metadata.as_dict()["dataset"] == "tests/fixtures/benchmarks/fake_smoke.jsonl"

    with pytest.raises(BenchmarkDataError) as length_exc:
        build_tokenized_batch(config, examples, adapter.tokenizer, max_length=0)

    assert length_exc.value.code == "invalid_max_length"


def test_named_real_benchmark_resolves_from_local_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_root = tmp_path / "benchmarks"
    hellaswag_dir = benchmark_root / "hellaswag"
    hellaswag_dir.mkdir(parents=True)
    (hellaswag_dir / "validation.jsonl").write_text(
        json.dumps(
            {
                "ind": 12,
                "split": "validation",
                "ctx": "A person opens the toolbox and",
                "endings": [
                    "finds a hammer.",
                    "turns into a cloud.",
                    "closes the ocean.",
                    "eats the wrench.",
                ],
                "label": "0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QAQ_BENCHMARK_DATA_ROOT", str(benchmark_root))
    monkeypatch.setenv("QAQ_DISABLE_HF_DATASETS", "1")
    config = _config(
        tmp_path,
        dataset="hellaswag",
        split="validation",
        prompt_format="lm_eval:hellaswag",
    )
    adapter = load_model_adapter(config)

    examples = load_benchmark_examples(config.dataset, split=config.split)
    batch = build_tokenized_batch(config, examples, adapter.tokenizer)

    assert len(examples) == 1
    assert examples[0].example_id == "12"
    assert examples[0].target == "finds a hammer."
    assert examples[0].metadata["real_benchmark"] is True
    assert examples[0].metadata["benchmark_name"] == "hellaswag"
    assert "Choices:" in examples[0].text
    assert batch.metadata.prompt_format == "lm_eval:hellaswag"

    output = adapter.reference_forward(batch)

    assert output.metadata["dataset_is_fake"] is False
    assert output.metadata["benchmark_is_real"] is True
    assert output.metadata["fixture_only_data"] is False
    assert output.metadata["dataset_sources"] == [str(hellaswag_dir / "validation.jsonl")]
    assert output.metadata["diagnostic"] is True


def test_named_real_benchmark_without_local_or_cached_data_fails_actionably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QAQ_BENCHMARK_DATA_ROOT", str(tmp_path / "empty-benchmarks"))
    monkeypatch.setenv("QAQ_DISABLE_HF_DATASETS", "1")

    with pytest.raises(BenchmarkDataError) as exc:
        load_benchmark_examples("hellaswag", split="validation")

    assert exc.value.code == "benchmark_dataset_unavailable"
    assert "QAQ_BENCHMARK_DATA_ROOT" in exc.value.message
    assert "hellaswag/validation.jsonl" in exc.value.message


def test_tiny_llama_adapter_verification_is_mechanism_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = tmp_path / "llama-local"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "num_hidden_layers": 2,
                "hidden_size": 6,
                "vocab_size": 128,
            }
        ),
        encoding="utf-8",
    )
    benchmark_root = tmp_path / "benchmarks"
    hellaswag_dir = benchmark_root / "hellaswag"
    hellaswag_dir.mkdir(parents=True)
    (hellaswag_dir / "validation.jsonl").write_text(
        json.dumps(
            {
                "ind": 7,
                "split": "validation",
                "ctx": "The chef sets a skillet on the stove and",
                "endings": [
                    "adds oil before cooking.",
                    "writes a novel underwater.",
                    "folds the moon in half.",
                    "paints the soup blue.",
                ],
                "label": "0",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class LocalHFTokenizer:
        pad_token_id = 0
        eos_token_id = 2
        model_max_length = 64

        @staticmethod
        def encode(text: str, *, add_special_tokens: bool = True) -> list[int]:
            prefix = [1] if add_special_tokens else []
            return prefix + [3 + (ord(character) % 50) for character in text]

    captured: dict[str, object] = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(tokenizer_ref: str, *, local_files_only: bool) -> LocalHFTokenizer:
            captured["tokenizer_ref"] = tokenizer_ref
            captured["local_files_only"] = local_files_only
            return LocalHFTokenizer()

    class FakeTransformers:
        AutoTokenizer = FakeAutoTokenizer

    monkeypatch.setattr("qaq.model_adapter._import_transformers", lambda: FakeTransformers)
    monkeypatch.setenv("QAQ_BENCHMARK_DATA_ROOT", str(benchmark_root))
    monkeypatch.setenv("QAQ_DISABLE_HF_DATASETS", "1")

    config_data = {
        "model": str(model_dir),
        "tokenizer": str(model_dir),
        "dataset": "hellaswag",
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
        "logging": {"console": False},
        "prompt_format": "lm_eval:hellaswag",
        "metric": "target_nll",
    }
    config = RunConfig.from_mapping(config_data, validate_output=False)

    result = verify_model_adapter_config(config, limit=1)

    assert captured == {"tokenizer_ref": str(model_dir), "local_files_only": True}
    assert result["status"] == "completed"
    assert result["adapter_kind"] == "huggingface_llama"
    assert result["model_source"] == "huggingface_local_metadata"
    assert result["model_is_fake"] is False
    assert result["tokenizer_is_fake"] is False
    assert result["dataset_is_fake"] is False
    assert result["fixture_only_data"] is False
    assert result["benchmark_is_real"] is True
    assert result["diagnostic"] is False
    assert result["evidence_level"] == "tiny_real_mechanism_path"
    assert result["accepted_as_real_adapter_verification"] is False
    assert result["accepted_as_benchmark_result"] is False
    assert result["weights_loaded"] is False
    assert result["architecture"]["framework"] == "transformers_llama"
    assert result["controlled_block_count"] == 4
    assert result["batch_metadata"]["prompt_format"] == "lm_eval:hellaswag"
    assert result["batch_metadata"]["context_length_policy"] == "truncate"
    assert result["target_count"] == 1
    assert result["dataset_sources"] == [str(hellaswag_dir / "validation.jsonl")]

    config_path = tmp_path / "verify_config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    output_path = tmp_path / "verify_artifact.json"
    assert model_adapter_main(
        [
            "--config",
            str(config_path),
            "--limit",
            "1",
            "--output",
            str(output_path),
            "--print-json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == written
    assert payload["evidence_level"] == "tiny_real_mechanism_path"
    assert payload["accepted_as_real_adapter_verification"] is False
    assert payload["accepted_as_benchmark_result"] is False
    assert payload["weights_loaded"] is False


def test_local_fake_model_and_tokenizer_metadata_can_be_loaded(tmp_path: Path) -> None:
    model_metadata = tmp_path / "model.json"
    tokenizer_metadata = tmp_path / "tokenizer.json"
    model_metadata.write_text(
        json.dumps(
            {
                "type": "fake_causal_lm",
                "model_id": "local-fake-model",
                "num_layers": 1,
                "hidden_size": 3,
                "vocab_size": 5,
            }
        ),
        encoding="utf-8",
    )
    tokenizer_metadata.write_text(
        json.dumps(
            {
                "type": "fake_tokenizer",
                "tokenizer_id": "local-fake-tokenizer",
                "model_max_length": 16,
            }
        ),
        encoding="utf-8",
    )
    config = _config(
        tmp_path,
        model=str(model_metadata),
        tokenizer=str(tokenizer_metadata),
    )

    adapter = load_model_adapter(config)
    blocks = discover_mha_ffn_blocks(adapter.architecture_metadata)

    assert adapter.model_id == "local-fake-model"
    assert adapter.tokenizer.tokenizer_id == "local-fake-tokenizer"
    assert adapter.hidden_size == 3
    assert [block.block_id for block in blocks] == ["layer_000.mha", "layer_000.ffn"]


def test_local_llama_config_metadata_can_be_loaded_without_weights(tmp_path: Path) -> None:
    model_config = tmp_path / "config.json"
    tokenizer_metadata = tmp_path / "tokenizer.json"
    model_config.write_text(
        json.dumps(
            {
                "model_type": "llama",
                "num_hidden_layers": 2,
                "hidden_size": 6,
                "vocab_size": 32,
            }
        ),
        encoding="utf-8",
    )
    tokenizer_metadata.write_text(
        json.dumps(
            {
                "type": "fake_tokenizer",
                "tokenizer_id": "local-test-tokenizer",
                "model_max_length": 16,
            }
        ),
        encoding="utf-8",
    )
    config = _config(
        tmp_path,
        model=str(model_config),
        tokenizer=str(tokenizer_metadata),
        prompt_format="paper_aligned_default",
    )

    adapter = load_model_adapter(config)
    blocks = discover_mha_ffn_blocks(
        adapter.architecture_metadata,
        supported_bit_widths=config.precision_candidates,
    )

    assert adapter.model_id == str(model_config)
    assert adapter.hidden_size == 6
    assert adapter.feature_source == "layer_output_pooled_shared_mha_ffn"
    assert adapter.architecture_metadata.framework == "transformers_llama"
    assert len(blocks) == 4
    assert blocks[0].tensor_names == (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    )
    assert blocks[1].tensor_names == (
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    )


def test_huggingface_adapter_returns_target_token_losses_without_real_checkpoint(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    class TinyHFOutput:
        def __init__(self, *, logits: object, hidden_states: object) -> None:
            self.logits = logits
            self.hidden_states = hidden_states

    class TinyHFModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.vocab_size = 256
            self.hidden_size = 6
            self.num_layers = 2

        def forward(
            self,
            *,
            input_ids: object,
            attention_mask: object | None = None,
            output_hidden_states: bool = False,
            use_cache: bool = False,
        ) -> TinyHFOutput:
            del attention_mask, output_hidden_states, use_cache
            batch_size, sequence_length = input_ids.shape
            logits = torch.zeros(
                (batch_size, sequence_length, self.vocab_size),
                dtype=torch.float32,
                device=input_ids.device,
            )
            base_hidden = torch.arange(
                self.hidden_size,
                dtype=torch.float32,
                device=input_ids.device,
            ).view(1, 1, self.hidden_size)
            hidden_states = tuple(
                base_hidden.expand(batch_size, sequence_length, self.hidden_size)
                + float(layer_index)
                for layer_index in range(self.num_layers + 1)
            )
            return TinyHFOutput(logits=logits, hidden_states=hidden_states)

    config = _config(
        tmp_path,
        dataset="tests/fixtures/benchmarks/router_training_real.jsonl",
        split="validation",
        prompt_format="question_answer_v1",
    )
    adapter = HuggingFaceCausalLMAdapter(
        model_ref="unused-local-test-model",
        model_id="tiny-llama-test",
        tokenizer=FakeTokenizer("tiny-hf-tokenizer", model_max_length=256),
        hf_config=type(
            "TinyConfig",
            (),
            {"num_hidden_layers": 2, "hidden_size": 6, "vocab_size": 256},
        )(),
        requested_device="cpu",
        gpu_ids=(),
    )
    adapter._model = TinyHFModel()

    examples = adapter.load_examples(config)
    batch = adapter.build_batch(config, examples)
    output = adapter.reference_forward(
        batch,
        block_ids=(
            "layer_000.mha",
            "layer_000.ffn",
            "layer_001.mha",
            "layer_001.ffn",
        ),
    )

    assert len(output.losses) == len(examples)
    assert all(loss is not None and math.isfinite(loss) for loss in output.losses)
    assert output.metadata["loss_source"] == "hf_target_token_nll"
    assert output.metadata["target_loss_count"] == len(examples)
    assert output.metadata["adapter_kind"] == "huggingface_llama"
    assert output.metadata["model_source"] == "injected_model_object"
    assert output.metadata["model_is_fake"] is True
    assert output.metadata["tokenizer_is_fake"] is True
    assert output.metadata["dataset_is_fake"] is False
    assert output.metadata["fixture_only_data"] is True
    assert output.metadata["diagnostic"] is True
    assert output.metadata["selected_gpu_ids"] == []
    assert set(output.hidden_states.by_block) == {
        "layer_000.mha",
        "layer_000.ffn",
        "layer_001.mha",
        "layer_001.ffn",
    }

    compact = adapter.reference_forward(
        batch,
        block_ids=(
            "layer_000.mha",
            "layer_000.ffn",
            "layer_001.mha",
            "layer_001.ffn",
        ),
        collect_hidden_states=False,
        store_full_logits=False,
    )

    assert compact.logits == tuple(() for _ in examples)
    assert compact.hidden_states.by_block == {}
    assert compact.metadata["collect_hidden_states"] is False
    assert compact.metadata["hidden_state_block_count"] == 0
    assert compact.metadata["full_logits_stored"] is False
    assert compact.metadata["logit_row_width"] == 0
    assert compact.metadata["peak_gpu_memory_bytes"] == 0
    assert compact.metadata["model_device_map"] == {"__single_device__": "cpu"}


def test_huggingface_device_map_auto_load_does_not_call_model_to(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

    class FakeTorch:
        cuda = FakeCuda()
        bfloat16 = "bfloat16"
        float16 = "float16"
        float32 = "float32"

        @staticmethod
        def device(value: str) -> str:
            return value

    class FakeParameter:
        requires_grad = False

    class FakeHFModel:
        hf_device_map = {
            "model.embed_tokens": "cuda:0",
            "model.layers.0": "cuda:1",
        }

        def __init__(self) -> None:
            self.parameter = FakeParameter()
            self.to_called = False
            self.eval_called = False

        def to(self, device: object) -> None:
            self.to_called = True
            raise AssertionError(f"model.to({device}) should not be called for device_map=auto")

        def eval(self) -> None:
            self.eval_called = True

        def named_parameters(self):
            return (("anchor", self.parameter),)

        def parameters(self):
            return iter((self.parameter,))

    fake_model = FakeHFModel()
    captured: dict[str, object] = {}

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_ref: str, **kwargs: object) -> FakeHFModel:
            captured["model_ref"] = model_ref
            captured["kwargs"] = kwargs
            return fake_model

    class FakeTransformers:
        AutoModelForCausalLM = FakeAutoModelForCausalLM

    monkeypatch.setattr("qaq.model_adapter._import_torch", lambda: FakeTorch)
    monkeypatch.setattr("qaq.model_adapter._import_transformers", lambda: FakeTransformers)

    adapter = HuggingFaceCausalLMAdapter(
        model_ref="local-llama",
        model_id="meta-llama/Llama-3.1-8B",
        tokenizer=FakeTokenizer("tiny-hf-tokenizer", model_max_length=64),
        hf_config=type(
            "TinyConfig",
            (),
            {
                "num_hidden_layers": 1,
                "hidden_size": 4,
                "vocab_size": 256,
                "torch_dtype": "float16",
            },
        )(),
        requested_device="cuda",
        gpu_ids=(0, 1),
        hf_device_map="auto",
        hf_max_memory_per_gpu="22GiB",
    )

    assert len(adapter.parameters()) == 1

    assert fake_model.to_called is False
    assert fake_model.eval_called is True
    assert captured["model_ref"] == "local-llama"
    assert captured["kwargs"] == {
        "local_files_only": True,
        "dtype": "float16",
        "device_map": "auto",
        "max_memory": {0: "22GiB", 1: "22GiB"},
    }


def test_cuda_weight_load_verification_requires_gpu_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QAQ_GPU_RUN_STATUS", raising=False)
    config = _config(tmp_path, device="cuda", gpu_ids=[0])

    with pytest.raises(ModelAdapterError) as exc:
        verify_model_adapter_config(config, load_weights=True)

    assert exc.value.code == "missing_gpu_selector_record"
    assert "scripts/gpu_run.py" in exc.value.message


def test_unavailable_model_and_dataset_fail_clearly(tmp_path: Path) -> None:
    missing_model_config = _config(
        tmp_path,
        model="definitely-missing-local-model",
        tokenizer="fake-qaq-smoke-tokenizer",
    )
    with pytest.raises(ModelAdapterError) as model_exc:
        load_model_adapter(missing_model_config)

    assert model_exc.value.code == "model_unavailable"

    unsupported_model = tmp_path / "unsupported-model.json"
    unsupported_model.write_text(
        json.dumps({"type": "not_a_supported_architecture"}),
        encoding="utf-8",
    )
    unsupported_model_config = _config(
        tmp_path,
        model=str(unsupported_model),
        tokenizer="fake-qaq-smoke-tokenizer",
    )
    with pytest.raises(ModelAdapterError) as unsupported_exc:
        load_model_adapter(unsupported_model_config)

    assert unsupported_exc.value.code == "unsupported_model"

    missing_dataset_config = _config(tmp_path, dataset="missing-dataset")
    with pytest.raises(BenchmarkDataError) as dataset_exc:
        load_benchmark_examples(
            missing_dataset_config.dataset,
            split=missing_dataset_config.split,
        )

    assert dataset_exc.value.code == "dataset_unavailable"
