from pathlib import Path

import pytest

from qaq.config import RunConfig
from qaq.model_adapter import FakeTokenizer, HuggingFaceCausalLMAdapter
from qaq.router.checkpoint import RouterCheckpoint, save_router_checkpoint
from qaq.router.types import (
    DEFAULT_DECISION_POLICY,
    RouterBlockParameters,
    RouterCheckpointMetadata,
)
from qaq.runtime.adaptive import run_adaptive_runtime
from qaq.tensor_bitplanes import create_tensor_bitplane_artifact, save_tensor_bitplane_artifact


def _config(tmp_path: Path, checkpoint: Path) -> RunConfig:
    return RunConfig.from_mapping(
        {
            "model": "tiny-llama-real-weight-test",
            "tokenizer": "tiny-hf-tokenizer",
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
            "prompt_format": "fake_smoke_v1",
            "metric": "exact_match",
            "router_checkpoint": str(checkpoint),
            "router_diagnostic": True,
        },
        validate_output=False,
    )


def test_adaptive_runtime_applies_reconstructed_tensor_weights_per_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")

    class TinyProjection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))

    class TinySelfAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = TinyProjection()
            self.k_proj = TinyProjection()
            self.v_proj = TinyProjection()
            self.o_proj = TinyProjection()

    class TinyMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = TinyProjection()
            self.up_proj = TinyProjection()
            self.down_proj = TinyProjection()

    class TinyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = TinySelfAttention()
            self.mlp = TinyMLP()

    class TinyBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([TinyLayer()])

    class TinyOutput:
        def __init__(self, *, logits: object, hidden_states: object) -> None:
            self.logits = logits
            self.hidden_states = hidden_states

    class TinyHFModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.model = TinyBackbone()
            self.vocab_size = 256
            self.hidden_size = 4

        def forward(
            self,
            *,
            input_ids: object,
            attention_mask: object | None = None,
            output_hidden_states: bool = False,
            use_cache: bool = False,
        ) -> TinyOutput:
            del output_hidden_states, use_cache
            batch_size, sequence_length = input_ids.shape
            signal = self.model.layers[0].self_attn.q_proj.weight.float().mean()
            logits = torch.zeros(
                (batch_size, sequence_length, self.vocab_size),
                dtype=torch.float32,
                device=input_ids.device,
            )
            logits[..., 1] = signal
            hidden_base = torch.arange(
                self.hidden_size,
                dtype=torch.float32,
                device=input_ids.device,
            ).view(1, 1, self.hidden_size)
            mask_term = (attention_mask is not None) * 0.0
            hidden_states = (
                hidden_base.expand(batch_size, sequence_length, self.hidden_size) + mask_term,
                hidden_base.expand(batch_size, sequence_length, self.hidden_size) + 1.0,
            )
            return TinyOutput(logits=logits, hidden_states=hidden_states)

    adapter = HuggingFaceCausalLMAdapter(
        model_ref="unused-local-test-model",
        model_id="tiny-llama-real-weight-test",
        tokenizer=FakeTokenizer("tiny-hf-tokenizer", model_max_length=64),
        hf_config=type(
            "TinyConfig",
            (),
            {"num_hidden_layers": 1, "hidden_size": 4, "vocab_size": 256},
        )(),
        requested_device="cpu",
        gpu_ids=(),
    )
    model = TinyHFModel()
    adapter._model = model

    block_ids = ("layer_000.mha", "layer_000.ffn")
    checkpoint = RouterCheckpoint(
        metadata=RouterCheckpointMetadata(
            checkpoint_id="tiny-real-weight-router",
            model_id="tiny-llama-real-weight-test",
            block_ids=block_ids,
            candidate_bit_widths=(4, 8),
            feature_source=adapter.feature_source,
            hidden_size=adapter.hidden_size,
            temperature=1.0,
            decision_policy=DEFAULT_DECISION_POLICY,
            max_bit_width=8,
            diagnostic=True,
        ),
        parameters={
            block_id: RouterBlockParameters(
                weights=((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
                bias=(1.0, 0.0),
            )
            for block_id in block_ids
        },
    )
    checkpoint_path = save_router_checkpoint(checkpoint, tmp_path / "router.json")
    config = _config(tmp_path, checkpoint_path)

    tensor_refs: dict[str, dict[str, str]] = {"layer_000.mha": {}, "layer_000.ffn": {}}
    for block_id, tensor_names in {
        "layer_000.mha": (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
        ),
        "layer_000.ffn": (
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ),
    }.items():
        for tensor_name in tensor_names:
            source = (
                torch.full((2, 2), 200.0)
                if tensor_name.endswith("q_proj.weight")
                else torch.zeros(2, 2)
            )
            artifact = create_tensor_bitplane_artifact(
                source,
                model_id=config.model,
                block_id=block_id,
                tensor_name=tensor_name,
                original_dtype="F32",
                compatibility={"block_granularity": "mha_ffn"},
            )
            path = save_tensor_bitplane_artifact(
                artifact,
                tmp_path / "artifacts" / block_id / f"{tensor_name.replace('.', '_')}.qaq.safetensors",
            )
            tensor_refs[block_id][tensor_name] = str(path)

    monkeypatch.setattr("qaq.runtime.adaptive.load_model_adapter", lambda _config: adapter)

    examples = adapter.load_examples(config)
    base_batch = adapter.build_batch(config, examples)
    base_output = adapter.reference_forward(base_batch, block_ids=block_ids)
    output = run_adaptive_runtime(config, artifact_refs=tensor_refs)

    assert base_output.predictions == (0, 0)
    assert output.raw_output.predictions == (1, 1)
    assert output.metadata["artifact_ref_mode"] == "full_tensor_index"
    assert output.metadata["mixed_precision_forward_applied"] is True
    assert output.metadata["weight_override_tensor_count"] == 14
    assert output.raw_output.metadata["execution_granularity"] == "per_query"
    assert model.model.layers[0].self_attn.q_proj.weight.detach().eq(0).all()
