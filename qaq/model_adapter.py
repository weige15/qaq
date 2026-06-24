"""Model adapter contracts and local Hugging Face/LLaMA verification support."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from qaq.benchmark_adapter import TokenizedBenchmarkBatch, build_tokenized_batch
from qaq.config import RunConfig, load_config_file
from qaq.data import BenchmarkDataError, BenchmarkExample, load_benchmark_examples


@dataclass(slots=True)
class ModelAdapterError(ValueError):
    """Raised when a model or tokenizer cannot be adapted for QAQ."""

    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(slots=True)
class FakeParameter:
    name: str
    requires_grad: bool = True


@dataclass(frozen=True, slots=True)
class HuggingFaceParameterView:
    """Named view over a real model parameter."""

    name: str
    parameter: Any

    @property
    def requires_grad(self) -> bool:
        return bool(getattr(self.parameter, "requires_grad", False))


@dataclass(frozen=True, slots=True)
class FakeArchitectureBlock:
    tensor_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FakeArchitectureLayer:
    mha: FakeArchitectureBlock
    ffn: FakeArchitectureBlock


@dataclass(frozen=True, slots=True)
class FakeModelArchitectureMetadata:
    model_id: str
    layers: tuple[FakeArchitectureLayer, ...]
    hidden_size: int
    vocab_size: int
    framework: str = "qaq_fake"

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "framework": self.framework,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_layers": len(self.layers),
        }


@dataclass(frozen=True, slots=True)
class HiddenStateBundle:
    """Router feature tensors keyed by controlled block ID."""

    feature_source: str
    by_block: dict[str, tuple[tuple[float, ...], ...]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_source": self.feature_source,
            "by_block": {
                block_id: [list(vector) for vector in vectors]
                for block_id, vectors in self.by_block.items()
            },
        }


@dataclass(frozen=True, slots=True)
class ReferenceBatchOutput:
    """Reference execution output consumed by later runtimes and metrics."""

    logits: tuple[tuple[float, ...], ...]
    losses: tuple[float | None, ...]
    predictions: tuple[int, ...]
    hidden_states: HiddenStateBundle
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "logits": [list(row) for row in self.logits],
            "losses": list(self.losses),
            "predictions": list(self.predictions),
            "hidden_states": self.hidden_states.as_dict(),
            "metadata": dict(self.metadata),
        }


class FakeTokenizer:
    """Deterministic character tokenizer for CPU-only smoke tests."""

    pad_token_id = 0

    def __init__(self, tokenizer_id: str, *, model_max_length: int = 128) -> None:
        if model_max_length <= 0:
            raise ModelAdapterError(
                "invalid_tokenizer_metadata",
                "model_max_length must be positive",
            )
        self.tokenizer_id = tokenizer_id
        self.model_max_length = model_max_length

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise ModelAdapterError("invalid_input", "tokenizer input must be text")
        if not text:
            return (1,)
        return tuple((ord(character) % 251) + 1 for character in text)


class FakeCausalLMAdapter:
    """Small deterministic causal-LM-shaped adapter for local tests."""

    feature_source = "block_output_pooled"
    supports_weight_overrides = False

    def __init__(
        self,
        *,
        model_id: str,
        tokenizer: FakeTokenizer,
        num_layers: int = 2,
        hidden_size: int = 4,
        vocab_size: int = 8,
    ) -> None:
        if num_layers <= 0:
            raise ModelAdapterError(
                "invalid_model_metadata",
                "num_layers must be positive",
            )
        if hidden_size <= 0:
            raise ModelAdapterError(
                "invalid_model_metadata",
                "hidden_size must be positive",
            )
        if vocab_size <= 0:
            raise ModelAdapterError(
                "invalid_model_metadata",
                "vocab_size must be positive",
            )
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self._parameters = tuple(
            FakeParameter(name=f"layers.{layer_index}.weight")
            for layer_index in range(num_layers)
        )
        self.architecture_metadata = _make_fake_architecture_metadata(
            model_id=model_id,
            num_layers=num_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )

    def parameters(self) -> tuple[FakeParameter, ...]:
        return self._parameters

    def freeze_base_model(self) -> None:
        for parameter in self._parameters:
            parameter.requires_grad = False

    def load_examples(
        self,
        config: RunConfig,
        *,
        limit: int | None = None,
    ) -> tuple[BenchmarkExample, ...]:
        return load_benchmark_examples(config.dataset, split=config.split, limit=limit)

    def build_batch(
        self,
        config: RunConfig,
        examples: tuple[BenchmarkExample, ...],
        *,
        max_length: int | None = None,
        context_length_policy: str = "truncate",
    ) -> TokenizedBenchmarkBatch:
        return build_tokenized_batch(
            config,
            examples,
            self.tokenizer,
            max_length=max_length,
            context_length_policy=context_length_policy,
        )

    def reference_forward(
        self,
        batch: TokenizedBenchmarkBatch,
        *,
        block_ids: tuple[str, ...] | None = None,
        weight_overrides: Mapping[str, Any] | None = None,
        precision_label: str | None = None,
        collect_hidden_states: bool = True,
        store_full_logits: bool = True,
    ) -> ReferenceBatchOutput:
        if weight_overrides:
            raise ModelAdapterError(
                "unsupported_weight_overrides",
                "fake adapter does not support real weight override inference",
            )
        del precision_label
        resolved_block_ids = block_ids or _default_block_ids(self.num_layers)
        logits: list[tuple[float, ...]] = []
        losses: list[float | None] = []
        predictions: list[int] = []

        active_lengths = tuple(sum(mask_row) for mask_row in batch.attention_mask)
        token_sums = tuple(
            sum(token for token, mask in zip(row, mask_row, strict=True) if mask)
            for row, mask_row in zip(batch.input_ids, batch.attention_mask, strict=True)
        )
        for token_sum, active_length, target in zip(
            token_sums,
            active_lengths,
            batch.targets,
            strict=True,
        ):
            row = _deterministic_logits(
                token_sum=token_sum,
                active_length=active_length,
                vocab_size=self.vocab_size,
            )
            if store_full_logits:
                logits.append(row)
            else:
                logits.append(())
            prediction = max(range(len(row)), key=row.__getitem__)
            predictions.append(prediction)
            losses.append(_target_loss(target=target, prediction=prediction))

        if collect_hidden_states:
            hidden_states = HiddenStateBundle(
                feature_source=self.feature_source,
                by_block={
                    block_id: tuple(
                        _hidden_feature(
                            block_id=block_id,
                            token_sum=token_sum,
                            active_length=active_length,
                            hidden_size=self.hidden_size,
                        )
                        for token_sum, active_length in zip(
                            token_sums,
                            active_lengths,
                            strict=True,
                        )
                    )
                    for block_id in resolved_block_ids
                },
            )
        else:
            hidden_states = HiddenStateBundle(
                feature_source=self.feature_source,
                by_block={},
            )
        return ReferenceBatchOutput(
            logits=tuple(logits),
            losses=tuple(losses),
            predictions=tuple(predictions),
            hidden_states=hidden_states,
            metadata={
                **_adapter_provenance_metadata(
                    model_id=self.model_id,
                    tokenizer=self.tokenizer,
                    batch=batch,
                    adapter_kind="fake_adapter",
                    selected_gpu_ids=(),
                    model_source="fake_adapter",
                ),
                "model": self.model_id,
                "tokenizer": self.tokenizer.tokenizer_id,
                "dataset": batch.metadata.dataset,
                "split": batch.metadata.split,
                "prompt_format": batch.metadata.prompt_format,
                "context_length_policy": batch.metadata.context_length_policy,
                "feature_source": self.feature_source,
                "precision": "fp16_reference",
                "batch_size": batch.metadata.batch_size,
                "example_ids": list(batch.example_ids),
                "collect_hidden_states": collect_hidden_states,
                "hidden_state_block_count": len(hidden_states.by_block),
                "full_logits_stored": store_full_logits,
                "logit_row_width": self.vocab_size if store_full_logits else 0,
                "peak_gpu_memory_bytes": 0,
                "peak_gpu_memory_gb": 0.0,
                "model_device_map": None,
            },
        )


class HuggingFaceTokenizer:
    """Tokenizer wrapper exposing the local tokenizer protocol."""

    def __init__(
        self,
        tokenizer: Any,
        tokenizer_id: str,
        *,
        tokenizer_ref: str | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self.tokenizer_id = tokenizer_id
        self.tokenizer_ref = tokenizer_ref or tokenizer_id
        self.pad_token_id = _resolve_pad_token_id(tokenizer)
        self.model_max_length = _resolve_model_max_length(tokenizer)

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise ModelAdapterError("invalid_input", "tokenizer input must be text")
        token_ids = self._tokenizer.encode(text, add_special_tokens=True)
        if not isinstance(token_ids, list) or any(not isinstance(item, int) for item in token_ids):
            raise ModelAdapterError(
                "invalid_tokenizer_output",
                "Hugging Face tokenizer must return integer token IDs",
            )
        return tuple(token_ids) or (self.pad_token_id,)


class HuggingFaceCausalLMAdapter:
    """Hugging Face causal-LM adapter for local Llama-family checkpoints."""

    feature_source = "layer_output_pooled_shared_mha_ffn"
    supports_weight_overrides = True

    def __init__(
        self,
        *,
        model_ref: str,
        model_id: str,
        tokenizer: Any,
        hf_config: Any,
        requested_device: str,
        gpu_ids: tuple[int, ...],
        hf_device_map: str | None = None,
        hf_max_memory_per_gpu: str | None = None,
    ) -> None:
        self.model_ref = model_ref
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.hf_config = hf_config
        self.requested_device = requested_device
        self.gpu_ids = gpu_ids
        self.hf_device_map = hf_device_map
        self.hf_max_memory_per_gpu = hf_max_memory_per_gpu
        self._model_loaded_from_pretrained = False
        self.num_layers = _coerce_positive_int(
            getattr(hf_config, "num_hidden_layers", None),
            field="num_hidden_layers",
        )
        self.hidden_size = _coerce_positive_int(
            getattr(hf_config, "hidden_size", None),
            field="hidden_size",
        )
        self.vocab_size = _coerce_positive_int(
            getattr(hf_config, "vocab_size", None),
            field="vocab_size",
        )
        self.architecture_metadata = _make_llama_architecture_metadata(
            model_id=model_id,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
        )
        self._model: Any | None = None
        self._parameter_views: tuple[HuggingFaceParameterView, ...] = ()

    def parameters(self) -> tuple[HuggingFaceParameterView, ...]:
        self._ensure_model_loaded()
        return self._parameter_views

    def freeze_base_model(self) -> None:
        model = self._ensure_model_loaded()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def load_examples(
        self,
        config: RunConfig,
        *,
        limit: int | None = None,
    ) -> tuple[BenchmarkExample, ...]:
        return load_benchmark_examples(config.dataset, split=config.split, limit=limit)

    def build_batch(
        self,
        config: RunConfig,
        examples: tuple[BenchmarkExample, ...],
        *,
        max_length: int | None = None,
        context_length_policy: str = "truncate",
    ) -> TokenizedBenchmarkBatch:
        return build_tokenized_batch(
            config,
            examples,
            self.tokenizer,
            max_length=max_length,
            context_length_policy=context_length_policy,
        )

    def reference_forward(
        self,
        batch: TokenizedBenchmarkBatch,
        *,
        block_ids: tuple[str, ...] | None = None,
        weight_overrides: Mapping[str, Any] | None = None,
        precision_label: str | None = None,
        collect_hidden_states: bool = True,
        store_full_logits: bool = True,
    ) -> ReferenceBatchOutput:
        torch = _import_torch()
        model = self._ensure_model_loaded()
        with _temporary_weight_overrides(model, weight_overrides or {}, torch=torch):
            device = _resolve_model_input_device(model, torch=torch)
            input_ids = torch.tensor(batch.input_ids, dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch.attention_mask, dtype=torch.long, device=device)
            with _inference_context(torch):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=collect_hidden_states,
                    use_cache=False,
                )
            target_losses = _hf_target_language_model_losses(
                model=model,
                tokenizer=self.tokenizer,
                batch=batch,
                torch=torch,
                device=device,
            )

        active_lengths = [max(sum(row), 1) for row in batch.attention_mask]
        logits: list[tuple[float, ...]] = []
        predictions: list[int] = []
        for row_index, active_length in enumerate(active_lengths):
            row_logits = output.logits[row_index, active_length - 1].detach()
            predictions.append(int(torch.argmax(row_logits).item()))
            if store_full_logits:
                logits.append(tuple(float(value) for value in row_logits.float().cpu().tolist()))
            else:
                logits.append(())

        resolved_block_ids = block_ids or _default_block_ids(self.num_layers)
        if collect_hidden_states:
            hidden_state_output = getattr(output, "hidden_states", None)
            if hidden_state_output is None:
                raise ModelAdapterError(
                    "missing_hidden_states",
                    "Hugging Face output did not include hidden states for router feature extraction",
                )
            hidden_by_layer = _pooled_hidden_by_layer(
                hidden_state_output,
                attention_mask=attention_mask,
                active_lengths=active_lengths,
            )
            hidden_states = HiddenStateBundle(
                feature_source=self.feature_source,
                by_block={
                    block_id: tuple(hidden_by_layer[_layer_index_from_block_id(block_id)])
                    for block_id in resolved_block_ids
                },
            )
        else:
            hidden_states = HiddenStateBundle(
                feature_source=self.feature_source,
                by_block={},
            )
        peak_gpu_memory_bytes = _peak_cuda_memory_allocated_bytes(
            torch,
            model=model,
            gpu_ids=self.gpu_ids,
            hf_device_map=self.hf_device_map,
        )
        return ReferenceBatchOutput(
            logits=tuple(logits),
            losses=target_losses,
            predictions=tuple(predictions),
            hidden_states=hidden_states,
            metadata={
                **_adapter_provenance_metadata(
                    model_id=self.model_id,
                    tokenizer=self.tokenizer,
                    batch=batch,
                    adapter_kind="huggingface_llama",
                    selected_gpu_ids=self.gpu_ids,
                    model_source=(
                        "huggingface_local_pretrained"
                        if self._model_loaded_from_pretrained
                        else "injected_model_object"
                    ),
                ),
                "model": self.model_id,
                "tokenizer": self.tokenizer.tokenizer_id,
                "dataset": batch.metadata.dataset,
                "split": batch.metadata.split,
                "prompt_format": batch.metadata.prompt_format,
                "context_length_policy": batch.metadata.context_length_policy,
                "feature_source": self.feature_source,
                "precision": precision_label or "hf_reference",
                "batch_size": batch.metadata.batch_size,
                "example_ids": list(batch.example_ids),
                "weight_override_count": len(weight_overrides or {}),
                "loss_source": (
                    "hf_target_token_nll"
                    if any(loss is not None for loss in target_losses)
                    else "none_no_targets"
                ),
                "target_loss_count": sum(loss is not None for loss in target_losses),
                "collect_hidden_states": collect_hidden_states,
                "hidden_state_block_count": len(hidden_states.by_block),
                "full_logits_stored": store_full_logits,
                "logit_row_width": self.vocab_size if store_full_logits else 0,
                "hf_device_map": self.hf_device_map or "single",
                "hf_max_memory_per_gpu": self.hf_max_memory_per_gpu,
                "model_device_map": _serialized_model_device_map(model),
                "input_device": str(device),
                "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
                "peak_gpu_memory_gb": peak_gpu_memory_bytes / float(1024**3),
            },
        )

    def _ensure_model_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        torch = _import_torch()
        transformers = _import_transformers()
        device_map_mode = self.hf_device_map or "single"
        dtype = _resolve_hf_dtype(
            torch,
            hf_config=self.hf_config,
            requested_device=self.requested_device,
        )
        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "dtype": dtype,
        }
        if device_map_mode == "auto":
            if self.requested_device != "cuda":
                raise ModelAdapterError(
                    "invalid_hf_device_map",
                    "hf_device_map='auto' requires config device='cuda'",
                )
            if not torch.cuda.is_available():
                raise ModelAdapterError(
                    "cuda_unavailable",
                    "hf_device_map='auto' requires visible CUDA devices",
                )
            load_kwargs["device_map"] = "auto"
            max_memory = _hf_max_memory_map(torch, self.hf_max_memory_per_gpu)
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory
            target_device = None
        else:
            target_device = _resolve_torch_device(
                torch,
                requested_device=self.requested_device,
                gpu_ids=self.gpu_ids,
            )
        try:
            try:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.model_ref,
                    **load_kwargs,
                )
            except TypeError as exc:
                if "dtype" not in str(exc):
                    raise
                legacy_kwargs = dict(load_kwargs)
                legacy_kwargs["torch_dtype"] = legacy_kwargs.pop("dtype")
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.model_ref,
                    **legacy_kwargs,
                )
        except Exception as exc:
            raise ModelAdapterError(
                "model_load_failed",
                f"failed to load Hugging Face model {self.model_ref!r} from local files: {exc}",
            ) from exc
        if target_device is not None:
            model.to(target_device)
        model.eval()
        self._model = model
        self._model_loaded_from_pretrained = True
        self._parameter_views = tuple(
            HuggingFaceParameterView(name=name, parameter=parameter)
            for name, parameter in model.named_parameters()
        )
        return model


@contextmanager
def _temporary_weight_overrides(
    model: Any,
    weight_overrides: Mapping[str, Any],
    *,
    torch: Any,
) -> Iterator[None]:
    if not weight_overrides:
        yield
        return

    named_parameters = dict(model.named_parameters())
    original_data: dict[str, Any] = {}
    try:
        for name, override in weight_overrides.items():
            parameter = named_parameters.get(name)
            if parameter is None:
                raise ModelAdapterError(
                    "unknown_weight_override",
                    f"model has no parameter named {name!r}",
                )
            if not isinstance(override, torch.Tensor):
                raise ModelAdapterError(
                    "invalid_weight_override",
                    f"weight override {name!r} must be a torch.Tensor",
                )
            expected_shape = tuple(int(value) for value in parameter.data.shape)
            actual_shape = tuple(int(value) for value in override.shape)
            if actual_shape != expected_shape:
                raise ModelAdapterError(
                    "weight_override_shape_mismatch",
                    f"weight override {name!r} has shape {actual_shape}, expected {expected_shape}",
                )
            original_data[name] = parameter.data
            parameter.data = override.to(
                device=parameter.data.device,
                dtype=parameter.data.dtype,
            ).contiguous()
        yield
    finally:
        for name, data in original_data.items():
            named_parameters[name].data = data


def load_model_adapter(config: RunConfig) -> Any:
    """Load a supported local adapter or fail before expensive model work."""

    model_metadata = _try_load_fake_model_metadata(config.model)
    if model_metadata is not None:
        tokenizer = _load_fake_tokenizer(config.tokenizer)
        return FakeCausalLMAdapter(
            model_id=model_metadata["model_id"],
            tokenizer=tokenizer,
            num_layers=model_metadata["num_layers"],
            hidden_size=model_metadata["hidden_size"],
            vocab_size=model_metadata["vocab_size"],
        )

    resolved_model_ref = _resolve_local_hf_ref(config.model)
    hf_config = _load_hf_config(resolved_model_ref)
    if getattr(hf_config, "model_type", None) != "llama":
        raise ModelAdapterError(
            "unsupported_model",
            f"model {config.model!r} is not a supported fake model or Llama Hugging Face checkpoint",
        )
    tokenizer_ref = (
        resolved_model_ref
        if config.tokenizer == config.model
        else _resolve_local_hf_ref(config.tokenizer)
    )
    tokenizer = _load_any_tokenizer(tokenizer_ref, tokenizer_id=config.tokenizer)
    return HuggingFaceCausalLMAdapter(
        model_ref=resolved_model_ref,
        model_id=config.model,
        tokenizer=tokenizer,
        hf_config=hf_config,
        requested_device=config.device,
        gpu_ids=config.gpu_ids,
        hf_device_map=config.hf_device_map,
        hf_max_memory_per_gpu=config.hf_max_memory_per_gpu,
    )



def verify_model_adapter_config(
    config: RunConfig,
    *,
    limit: int | None = None,
    max_length: int | None = None,
    context_length_policy: str = "truncate",
    load_weights: bool = False,
) -> dict[str, Any]:
    """Verify adapter metadata, tokenizer, and benchmark batching without fake fallback.

    Weight loading is opt-in because LLaMA-sized checkpoints must be launched on
    the lab GPU server through scripts/gpu_run.py under repository policy.
    """

    adapter = load_model_adapter(config)
    examples = adapter.load_examples(config, limit=limit or config.max_examples)
    batch = adapter.build_batch(
        config,
        examples,
        max_length=max_length,
        context_length_policy=context_length_policy,
    )
    adapter_kind = _adapter_kind(adapter)
    gpu_selector_record = _gpu_selector_record_from_env()
    if load_weights:
        if config.device == "cuda" and gpu_selector_record is None:
            raise ModelAdapterError(
                "missing_gpu_selector_record",
                "CUDA weight-load verification must be launched through scripts/gpu_run.py",
            )
        parameter_count = len(adapter.parameters())
        weights_loaded = True
        model_source = _loaded_model_source(adapter)
    else:
        parameter_count = None
        weights_loaded = False
        model_source = _metadata_model_source(adapter)
    provenance = _adapter_provenance_metadata(
        model_id=str(getattr(adapter, "model_id", config.model)),
        tokenizer=adapter.tokenizer,
        batch=batch,
        adapter_kind=adapter_kind,
        selected_gpu_ids=config.gpu_ids,
        model_source=model_source,
    )
    architecture = adapter.architecture_metadata.as_dict()
    model_ref = str(getattr(adapter, "model_ref", config.model))
    tokenizer_ref = str(getattr(adapter.tokenizer, "tokenizer_ref", config.tokenizer))
    result = {
        "status": "completed",
        "verification_type": "model_adapter",
        "model": config.model,
        "tokenizer": config.tokenizer,
        "resolved_model_ref": model_ref,
        "resolved_tokenizer_ref": tokenizer_ref,
        "dataset": config.dataset,
        "split": config.split,
        "prompt_format": config.prompt_format or "plain",
        "metric": config.metric,
        "device": config.device,
        "selected_gpu_ids": list(config.gpu_ids),
        "gpu_selector_record": gpu_selector_record,
        "selected_physical_gpu_ids": _selected_physical_gpu_ids(gpu_selector_record),
        "hf_device_map": config.hf_device_map or "single",
        "hf_max_memory_per_gpu": config.hf_max_memory_per_gpu,
        "adapter_kind": adapter_kind,
        "model_source": model_source,
        "architecture": architecture,
        "controlled_block_count": int(architecture.get("num_layers", 0)) * 2,
        "batch_metadata": batch.metadata.as_dict(),
        "example_count": len(examples),
        "example_ids": list(batch.example_ids),
        "target_count": sum(target is not None for target in batch.targets),
        "weight_load_requested": load_weights,
        "weights_loaded": weights_loaded,
        "parameter_count": parameter_count,
        **provenance,
    }
    evidence_level = _adapter_verification_evidence_level(
        config=config,
        result=result,
        model_ref=model_ref,
        tokenizer_ref=tokenizer_ref,
    )
    result["evidence_level"] = evidence_level
    result["accepted_as_real_adapter_verification"] = evidence_level == "real_subset_path"
    result["accepted_as_benchmark_result"] = False
    result["benchmark_acceptance_reason"] = "adapter_verification_is_not_full_comparison"
    return result


def _gpu_selector_record_from_env() -> dict[str, Any] | None:
    value = os.environ.get("QAQ_GPU_RUN_STATUS")
    if not value:
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(decoded, dict):
        return decoded
    return None


def _selected_physical_gpu_ids(record: Mapping[str, Any] | None) -> list[int]:
    if record is None:
        return []
    selected = record.get("selected_physical_gpu_ids")
    if not isinstance(selected, list | tuple):
        return []
    return [int(item) for item in selected if isinstance(item, int) and not isinstance(item, bool)]


def _adapter_verification_evidence_level(
    *,
    config: RunConfig,
    result: Mapping[str, Any],
    model_ref: str,
    tokenizer_ref: str,
) -> str:
    if result.get("diagnostic") is True:
        return "diagnostic_fake_path"
    if result.get("adapter_kind") != "huggingface_llama":
        return "diagnostic_fake_path"
    if not _is_actual_llama31_real_subset(
        config=config,
        result=result,
        model_ref=model_ref,
        tokenizer_ref=tokenizer_ref,
    ):
        return "tiny_real_mechanism_path"
    return "real_subset_path"


def _is_actual_llama31_real_subset(
    *,
    config: RunConfig,
    result: Mapping[str, Any],
    model_ref: str,
    tokenizer_ref: str,
) -> bool:
    return all(
        (
            config.model == "meta-llama/Llama-3.1-8B",
            config.tokenizer == "meta-llama/Llama-3.1-8B",
            result.get("benchmark_is_real") is True,
            result.get("dataset_is_fake") is False,
            result.get("model_is_fake") is False,
            result.get("tokenizer_is_fake") is False,
            result.get("fixture_only_data") is False,
            _has_local_hf_config(model_ref),
            _has_local_hf_tokenizer_files(tokenizer_ref),
            not _path_has_test_or_tmp_provenance(model_ref),
            not _path_has_test_or_tmp_provenance(tokenizer_ref),
        )
    )


def _has_local_hf_config(model_ref: str) -> bool:
    path = Path(model_ref)
    if path.is_file():
        return path.name == "config.json"
    return path.is_dir() and (path / "config.json").is_file()


def _has_local_hf_tokenizer_files(tokenizer_ref: str) -> bool:
    path = Path(tokenizer_ref)
    if path.is_file():
        return path.name in {
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        }
    if not path.is_dir():
        return False
    return any(
        (path / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    )


def _path_has_test_or_tmp_provenance(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in ("/tmp/", "tests/fixtures", "test_model_adapter"))


def _adapter_kind(adapter: Any) -> str:
    if isinstance(adapter, FakeCausalLMAdapter):
        return "fake_adapter"
    if isinstance(adapter, HuggingFaceCausalLMAdapter):
        return "huggingface_llama"
    return type(adapter).__name__


def _metadata_model_source(adapter: Any) -> str:
    if isinstance(adapter, FakeCausalLMAdapter):
        return "fake_adapter"
    if isinstance(adapter, HuggingFaceCausalLMAdapter):
        return "huggingface_local_metadata"
    return "unknown_adapter_metadata"


def _loaded_model_source(adapter: Any) -> str:
    if isinstance(adapter, FakeCausalLMAdapter):
        return "fake_adapter"
    if isinstance(adapter, HuggingFaceCausalLMAdapter):
        if adapter._model_loaded_from_pretrained:
            return "huggingface_local_pretrained"
        return "injected_model_object"
    return "unknown_adapter_loaded_model"


def _load_fake_tokenizer(tokenizer_id: str) -> FakeTokenizer:
    if _is_fake_identifier(tokenizer_id):
        return FakeTokenizer(tokenizer_id)

    path = Path(tokenizer_id)
    if not path.is_file():
        raise ModelAdapterError(
            "tokenizer_unavailable",
            f"tokenizer {tokenizer_id!r} is not a supported fake tokenizer or readable metadata file",
        )
    metadata = _read_json_metadata(path, kind="tokenizer")
    if metadata.get("type") != "fake_tokenizer":
        raise ModelAdapterError(
            "unsupported_tokenizer",
            f"tokenizer metadata {path} is not type fake_tokenizer",
        )
    return FakeTokenizer(
        str(metadata.get("tokenizer_id", path)),
        model_max_length=_coerce_positive_int(
            metadata.get("model_max_length", 128),
            field="model_max_length",
        ),
    )


def _load_any_tokenizer(tokenizer_ref: str, *, tokenizer_id: str | None = None) -> Any:
    tokenizer_id = tokenizer_id or tokenizer_ref
    try:
        return _load_fake_tokenizer(tokenizer_ref)
    except ModelAdapterError:
        pass
    transformers = _import_transformers()
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_ref,
            local_files_only=True,
        )
    except Exception as exc:
        raise ModelAdapterError(
            "tokenizer_unavailable",
            f"tokenizer {tokenizer_id!r} is not a supported fake tokenizer or local Hugging Face tokenizer: {exc}",
        ) from exc
    return HuggingFaceTokenizer(tokenizer, tokenizer_id, tokenizer_ref=tokenizer_ref)


def _try_load_fake_model_metadata(model_id: str) -> dict[str, Any] | None:
    if _is_fake_identifier(model_id):
        return {
            "model_id": model_id,
            "num_layers": 2,
            "hidden_size": 4,
            "vocab_size": 8,
        }

    path = Path(model_id)
    if not path.is_file():
        return None
    metadata = _read_json_metadata(path, kind="model")
    if metadata.get("type") != "fake_causal_lm":
        return None
    return {
        "model_id": str(metadata.get("model_id", path)),
        "num_layers": _coerce_positive_int(metadata.get("num_layers", 2), field="num_layers"),
        "hidden_size": _coerce_positive_int(metadata.get("hidden_size", 4), field="hidden_size"),
        "vocab_size": _coerce_positive_int(metadata.get("vocab_size", 8), field="vocab_size"),
    }


def _load_hf_config(model_ref: str) -> Any:
    path = Path(model_ref)
    if path.is_file():
        metadata = _read_json_metadata(path, kind="model")
        return _DictConfig(metadata)
    config_path = path / "config.json" if path.is_dir() else None
    if config_path is not None and config_path.is_file():
        metadata = _read_json_metadata(config_path, kind="model")
        return _DictConfig(metadata)
    transformers = _import_transformers()
    try:
        return transformers.AutoConfig.from_pretrained(
            model_ref,
            local_files_only=True,
        )
    except Exception as exc:
        raise ModelAdapterError(
            "model_unavailable",
            f"model {model_ref!r} is not a supported fake model or local Hugging Face config: {exc}",
        ) from exc


def _resolve_local_hf_ref(model_ref: str) -> str:
    path = Path(model_ref)
    if path.exists():
        return str(path)
    if "/" not in model_ref:
        return model_ref
    namespace, name = model_ref.split("/", 1)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_root = cache_root / f"models--{namespace}--{name}"
    refs_main = repo_root / "refs" / "main"
    if refs_main.is_file():
        snapshot = repo_root / "snapshots" / refs_main.read_text(encoding="utf-8").strip()
        if snapshot.is_dir():
            return str(snapshot)
    snapshots = repo_root / "snapshots"
    if snapshots.is_dir():
        candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
        if candidates:
            return str(candidates[-1])
    return model_ref


class _DictConfig:
    def __init__(self, value: dict[str, Any]) -> None:
        self._value = value

    def __getattr__(self, name: str) -> Any:
        try:
            return self._value[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _read_json_metadata(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelAdapterError(f"{kind}_metadata_read_failed", str(exc)) from exc
    if not isinstance(raw, dict):
        raise ModelAdapterError(
            f"invalid_{kind}_metadata",
            f"{kind} metadata must be a JSON object",
        )
    return raw


def _coerce_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelAdapterError(
            "invalid_model_metadata",
            f"{field} must be a positive integer",
        )
    return value


def _is_fake_identifier(value: str) -> bool:
    return value.startswith("fake-") or value.startswith("fake_")


def _make_fake_architecture_metadata(
    *,
    model_id: str,
    num_layers: int,
    hidden_size: int,
    vocab_size: int,
) -> FakeModelArchitectureMetadata:
    return FakeModelArchitectureMetadata(
        model_id=model_id,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        layers=tuple(
            FakeArchitectureLayer(
                mha=FakeArchitectureBlock(
                    tensor_names=(
                        f"layers.{layer_index}.mha.q_proj.weight",
                        f"layers.{layer_index}.mha.o_proj.weight",
                    )
                ),
                ffn=FakeArchitectureBlock(
                    tensor_names=(
                        f"layers.{layer_index}.ffn.gate_proj.weight",
                        f"layers.{layer_index}.ffn.down_proj.weight",
                    )
                ),
            )
            for layer_index in range(num_layers)
        ),
    )


def _make_llama_architecture_metadata(
    *,
    model_id: str,
    num_layers: int,
    hidden_size: int,
    vocab_size: int,
) -> FakeModelArchitectureMetadata:
    return FakeModelArchitectureMetadata(
        model_id=model_id,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        framework="transformers_llama",
        layers=tuple(
            FakeArchitectureLayer(
                mha=FakeArchitectureBlock(
                    tensor_names=(
                        f"model.layers.{layer_index}.self_attn.q_proj.weight",
                        f"model.layers.{layer_index}.self_attn.k_proj.weight",
                        f"model.layers.{layer_index}.self_attn.v_proj.weight",
                        f"model.layers.{layer_index}.self_attn.o_proj.weight",
                    )
                ),
                ffn=FakeArchitectureBlock(
                    tensor_names=(
                        f"model.layers.{layer_index}.mlp.gate_proj.weight",
                        f"model.layers.{layer_index}.mlp.up_proj.weight",
                        f"model.layers.{layer_index}.mlp.down_proj.weight",
                    )
                ),
            )
            for layer_index in range(num_layers)
        ),
    )


def _default_block_ids(num_layers: int) -> tuple[str, ...]:
    return tuple(
        block_id
        for layer_index in range(num_layers)
        for block_id in (f"layer_{layer_index:03d}.mha", f"layer_{layer_index:03d}.ffn")
    )


def _deterministic_logits(
    *,
    token_sum: int,
    active_length: int,
    vocab_size: int,
) -> tuple[float, ...]:
    return tuple(
        round(((token_sum + (index + 1) * active_length) % 997) / 997.0, 6)
        for index in range(vocab_size)
    )


def _adapter_provenance_metadata(
    *,
    model_id: str,
    tokenizer: Any,
    batch: TokenizedBenchmarkBatch,
    adapter_kind: str,
    selected_gpu_ids: tuple[int, ...],
    model_source: str,
) -> dict[str, Any]:
    sources = tuple(
        str(example.metadata.get("source"))
        for example in batch.examples
        if example.metadata.get("source") is not None
    )
    dataset_is_fake = _dataset_is_fake(batch.metadata.dataset)
    tokenizer_id = str(getattr(tokenizer, "tokenizer_id", ""))
    tokenizer_is_fake = _tokenizer_is_fake(tokenizer)
    fixture_only = _fixture_only_dataset(batch.metadata.dataset, sources)
    model_is_fake = (
        _is_fake_identifier(model_id)
        or adapter_kind == "fake_adapter"
        or model_source == "injected_model_object"
    )
    benchmark_is_real = all(
        bool(example.metadata.get("real_benchmark"))
        for example in batch.examples
    )
    diagnostic = model_is_fake or tokenizer_is_fake or dataset_is_fake or fixture_only
    return {
        "adapter_kind": adapter_kind,
        "model_source": model_source,
        "model_is_fake": model_is_fake,
        "tokenizer_is_fake": tokenizer_is_fake,
        "dataset_is_fake": dataset_is_fake,
        "fixture_only_data": fixture_only,
        "benchmark_is_real": benchmark_is_real,
        "diagnostic": diagnostic,
        "selected_gpu_ids": list(selected_gpu_ids),
        "dataset_sources": list(dict.fromkeys(sources)),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_id": tokenizer_id,
    }


def _dataset_is_fake(dataset: str) -> bool:
    lowered = dataset.lower()
    return (
        dataset in _BUILTIN_DATASETS_FOR_PROVENANCE
        or "fake" in lowered
        or "smoke" in lowered
    )


def _tokenizer_is_fake(tokenizer: Any) -> bool:
    tokenizer_id = str(getattr(tokenizer, "tokenizer_id", ""))
    return isinstance(tokenizer, FakeTokenizer) or _is_fake_identifier(tokenizer_id)


def _fixture_only_dataset(dataset: str, sources: tuple[str, ...]) -> bool:
    values = (dataset, *sources)
    return any(
        value == "fixture"
        or "tests/fixtures" in value
        or "tests\\fixtures" in value
        for value in values
    )


_BUILTIN_DATASETS_FOR_PROVENANCE = frozenset({"fake_smoke", "toy_prompts"})


def _target_loss(*, target: str | None, prediction: int) -> float | None:
    if target is None:
        return None
    target_score = sum(ord(character) for character in target) % 17
    return round(abs(target_score - prediction) / 17.0, 6)


def _hf_target_language_model_losses(
    *,
    model: Any,
    tokenizer: Any,
    batch: TokenizedBenchmarkBatch,
    torch: Any,
    device: Any,
) -> tuple[float | None, ...]:
    """Compute target-token NLL for HF reference outputs when targets exist."""

    losses: list[float | None] = [None for _ in batch.examples]
    target_rows: list[tuple[int, ...]] = []
    target_lengths: list[int] = []
    row_indices: list[int] = []
    prompt_lengths: list[int] = []
    max_model_length = int(getattr(tokenizer, "model_max_length", 0) or 0)

    for row_index, (input_row, mask_row, target) in enumerate(
        zip(batch.input_ids, batch.attention_mask, batch.targets, strict=True)
    ):
        if target is None:
            continue
        prompt_ids = tuple(
            token for token, mask_value in zip(input_row, mask_row, strict=True) if mask_value
        )
        if not prompt_ids:
            prompt_ids = (int(getattr(tokenizer, "pad_token_id", 0)),)
        target_ids = _encode_hf_target_tokens(tokenizer, target)
        if max_model_length > 0 and len(prompt_ids) + len(target_ids) > max_model_length:
            target_ids = target_ids[: max(max_model_length - len(prompt_ids), 0)]
        if not target_ids:
            continue
        target_rows.append(prompt_ids + target_ids)
        target_lengths.append(len(target_ids))
        row_indices.append(row_index)
        prompt_lengths.append(len(prompt_ids))

    if not target_rows:
        return tuple(losses)

    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0))
    padded_length = max(len(row) for row in target_rows)
    input_ids = torch.tensor(
        [row + (pad_token_id,) * (padded_length - len(row)) for row in target_rows],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.tensor(
        [(1,) * len(row) + (0,) * (padded_length - len(row)) for row in target_rows],
        dtype=torch.long,
        device=device,
    )
    with _inference_context(torch):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
    for target_batch_index, original_row_index in enumerate(row_indices):
        prompt_length = prompt_lengths[target_batch_index]
        target_length = target_lengths[target_batch_index]
        start = prompt_length - 1
        end = prompt_length + target_length - 1
        target_ids = input_ids[
            target_batch_index,
            prompt_length : prompt_length + target_length,
        ]
        target_logits = output.logits[target_batch_index, start:end].float()
        loss = torch.nn.functional.cross_entropy(
            target_logits,
            target_ids,
            reduction="mean",
        )
        losses[original_row_index] = float(loss.detach().cpu().item())
    return tuple(losses)


def _encode_hf_target_tokens(tokenizer: Any, target: str) -> tuple[int, ...]:
    raw_tokenizer = getattr(tokenizer, "_tokenizer", None)
    if raw_tokenizer is not None and hasattr(raw_tokenizer, "encode"):
        token_ids = raw_tokenizer.encode(target, add_special_tokens=False)
        if isinstance(token_ids, list) and all(isinstance(item, int) for item in token_ids):
            return tuple(token_ids)
    return tuple(tokenizer.encode(target))


def _hidden_feature(
    *,
    block_id: str,
    token_sum: int,
    active_length: int,
    hidden_size: int,
) -> tuple[float, ...]:
    block_score = sum(ord(character) for character in block_id)
    return tuple(
        round(((token_sum + block_score + (offset + 1) * active_length) % 101) / 100.0, 6)
        for offset in range(hidden_size)
    )


def _resolve_pad_token_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    return 0


def _resolve_model_max_length(tokenizer: Any) -> int:
    value = getattr(tokenizer, "model_max_length", 0)
    if not isinstance(value, int) or value <= 0 or value > 1_000_000:
        return 4096
    return value


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise ModelAdapterError(
            "transformers_unavailable",
            "Hugging Face model loading requires the optional transformers package",
        ) from exc
    return transformers


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ModelAdapterError(
            "torch_unavailable",
            "Hugging Face model execution requires the optional torch package",
        ) from exc
    return torch


def _resolve_torch_device(
    torch: Any,
    *,
    requested_device: str,
    gpu_ids: tuple[int, ...],
) -> Any:
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise ModelAdapterError(
                "cuda_unavailable",
                "config requested cuda but torch.cuda.is_available() is false",
            )
        gpu_id = gpu_ids[0] if gpu_ids else 0
        return torch.device(f"cuda:{gpu_id}")
    raise ModelAdapterError(
        "unsupported_device",
        "Hugging Face adapter supports only cpu or cuda devices",
    )


def _resolve_hf_dtype(
    torch: Any,
    *,
    hf_config: Any,
    requested_device: str,
) -> Any:
    if requested_device == "cuda":
        dtype_hint = str(getattr(hf_config, "torch_dtype", "")).lower()
        if "bfloat16" in dtype_hint and hasattr(torch, "bfloat16"):
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _hf_max_memory_map(torch: Any, per_gpu: str | None) -> dict[int, str] | None:
    if per_gpu is None:
        return None
    if not torch.cuda.is_available():
        return None
    return {index: per_gpu for index in range(torch.cuda.device_count())}


def _resolve_model_input_device(model: Any, *, torch: Any) -> Any:
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        embeddings = get_embeddings()
        if embeddings is not None:
            try:
                return next(embeddings.parameters()).device
            except (AttributeError, StopIteration):
                pass

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        for value in device_map.values():
            device = _device_from_map_value(value, torch=torch)
            if device is not None and str(device).startswith("cuda"):
                return device
        for value in device_map.values():
            device = _device_from_map_value(value, torch=torch)
            if device is not None:
                return device

    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ModelAdapterError(
            "model_has_no_parameters",
            "cannot infer a device for Hugging Face model inputs",
        ) from exc


def _device_from_map_value(value: Any, *, torch: Any) -> Any | None:
    if isinstance(value, int):
        return torch.device(f"cuda:{value}")
    if isinstance(value, str):
        if value == "disk":
            return None
        try:
            return torch.device(value)
        except Exception:
            return None
    return None


def _serialized_model_device_map(model: Any) -> dict[str, str] | None:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        return {str(name): str(device) for name, device in device_map.items()}
    try:
        return {"__single_device__": str(next(model.parameters()).device)}
    except StopIteration:
        return None


def _peak_cuda_memory_allocated_bytes(
    torch: Any,
    *,
    model: Any,
    gpu_ids: tuple[int, ...],
    hf_device_map: str | None,
) -> int:
    if not torch.cuda.is_available():
        return 0
    device_count = int(torch.cuda.device_count())
    if device_count <= 0:
        return 0
    if hf_device_map == "auto":
        indices = tuple(range(device_count))
    else:
        indices = _model_cuda_device_indices(model) or tuple(gpu_ids)
    valid_indices = tuple(index for index in indices if 0 <= index < device_count)
    if not valid_indices:
        valid_indices = (0,)
    return max(int(torch.cuda.max_memory_allocated(index)) for index in valid_indices)


def _model_cuda_device_indices(model: Any) -> tuple[int, ...]:
    indices: list[int] = []
    for parameter in model.parameters():
        device = getattr(parameter, "device", None)
        if str(device).startswith("cuda"):
            index = getattr(device, "index", None)
            indices.append(0 if index is None else int(index))
    return tuple(dict.fromkeys(indices))


def _inference_context(torch: Any) -> Any:
    inference_mode = getattr(torch, "inference_mode", None)
    if callable(inference_mode):
        return inference_mode()
    return torch.no_grad()


def _pooled_hidden_by_layer(
    hidden_states: Any,
    *,
    attention_mask: Any,
    active_lengths: list[int],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    # Hidden states are exposed per transformer layer, not separately for MHA and
    # FFN. The router training docs record the shared per-layer feature assumption.
    result: list[tuple[tuple[float, ...], ...]] = []
    for layer_hidden in hidden_states[1:]:
        rows = []
        for row_index, active_length in enumerate(active_lengths):
            values = layer_hidden[row_index, :active_length].detach().float()
            mask = attention_mask[row_index, :active_length].detach().float().unsqueeze(-1)
            pooled = (values * mask).sum(dim=0) / mask.sum().clamp_min(1.0)
            rows.append(tuple(float(value) for value in pooled.cpu().tolist()))
        result.append(tuple(rows))
    return tuple(result)


def _layer_index_from_block_id(block_id: str) -> int:
    try:
        layer_part, _ = block_id.split(".", 1)
        return int(layer_part.removeprefix("layer_"))
    except (ValueError, AttributeError) as exc:
        raise ModelAdapterError(
            "invalid_block_id",
            f"cannot derive layer index from block_id {block_id!r}",
        ) from exc

def _positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify QAQ model and benchmark adapter inputs.")
    parser.add_argument("--config", required=True, help="Run config JSON/TOML to verify.")
    parser.add_argument("--limit", type=_positive_int_arg, default=None)
    parser.add_argument("--max-length", type=_positive_int_arg, default=None)
    parser.add_argument(
        "--context-length-policy",
        choices=("truncate", "reject"),
        default="truncate",
    )
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Also load model weights; use only on the lab GPU server for large checkpoints.",
    )
    parser.add_argument("--print-json", action="store_true", help="Print verification JSON.")
    parser.add_argument("--output", help="Write verification JSON to this path.")
    args = parser.parse_args(argv)
    try:
        config = load_config_file(args.config, validate_output=False)
        result = verify_model_adapter_config(
            config,
            limit=args.limit,
            max_length=args.max_length,
            context_length_policy=args.context_length_policy,
            load_weights=args.load_weights,
        )
    except (BenchmarkDataError, ModelAdapterError, ValueError) as exc:
        code = getattr(exc, "code", "model_adapter_verification_failed")
        message = getattr(exc, "message", str(exc))
        print(
            json.dumps({"status": "failed", "code": code, "message": message}, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.print_json:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
