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
    assert set(output.hidden_states.by_block) == {
        "layer_000.mha",
        "layer_000.ffn",
        "layer_001.mha",
        "layer_001.ffn",
    }


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
