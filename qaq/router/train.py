"""Router training orchestration."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from qaq.artifacts import load_bitplane_artifact, save_bitplane_artifact
from qaq.bitplanes import (
    BitPlaneArtifact,
    BitPlaneError,
    create_bitplane_artifact_from_quantized_values,
    reconstruct_weight,
)
from qaq.blocks import discover_mha_ffn_blocks
from qaq.config import LoggingConfig, RunConfig
from qaq.data import BenchmarkDataError, load_benchmark_examples
from qaq.logging import JsonlLogWriter, LogEvent, open_run_log
from qaq.manifest import RunManifest, create_run_manifest
from qaq.model_adapter import FakeCausalLMAdapter, load_model_adapter
from qaq.progress import ConsoleProgressMonitor, ProgressState, TimingMeasurement
from qaq.quantization import flatten_tensor
from qaq.router.checkpoint import RouterCheckpoint, save_router_checkpoint
from qaq.router.losses import (
    ROUTER_COST_CROSS_ENTROPY,
    SUPPORTED_DISTILLATION_LOSSES,
    LossRecord,
    RouterObjectiveSample,
    compute_router_objective_loss,
    softmax_from_costs,
)
from qaq.router.policy import normalize_scores, score_block
from qaq.router.types import (
    DEFAULT_DECISION_POLICY,
    RouterBlockParameters,
    RouterCheckpointMetadata,
)
from qaq.status import EventType, RunStatus
from qaq.tensor_bitplanes import (
    TensorBitPlaneArtifact,
    TensorBitPlaneError,
    is_tensor_bitplane_artifact_path,
    load_tensor_bitplane_artifact,
    normalized_tensor_reconstruction_delta,
)


MODULE_NAME = "router_train"
CUDA_MODEL_LOAD_MARGIN_BYTES = 512 * 1024 * 1024


@dataclass(slots=True)
class RouterTrainingError(ValueError):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class RouterHyperparameters:
    learning_rate: float = 0.1
    max_steps: int = 2
    temperature: float = 1.0
    target_temperature: float = 1.0
    bit_cost_weight: float = 0.02
    decision_policy: str = DEFAULT_DECISION_POLICY

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RouterHyperparameters":
        data = dict(value or {})
        learning_rate = _coerce_positive_float(
            data.get("learning_rate", 0.1),
            field="router.learning_rate",
        )
        max_steps = _coerce_positive_int(data.get("max_steps", 2), field="router.max_steps")
        temperature = _coerce_positive_float(
            data.get("temperature", 1.0),
            field="router.temperature",
        )
        target_temperature = _coerce_positive_float(
            data.get("target_temperature", 1.0),
            field="router.target_temperature",
        )
        bit_cost_weight = _coerce_non_negative_float(
            data.get("bit_cost_weight", 0.02),
            field="router.bit_cost_weight",
        )
        decision_policy = _coerce_non_empty_string(
            data.get("decision_policy", DEFAULT_DECISION_POLICY),
            field="router.decision_policy",
        )
        return cls(
            learning_rate=learning_rate,
            max_steps=max_steps,
            temperature=temperature,
            target_temperature=target_temperature,
            bit_cost_weight=bit_cost_weight,
            decision_policy=decision_policy,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "temperature": self.temperature,
            "target_temperature": self.target_temperature,
            "bit_cost_weight": self.bit_cost_weight,
            "decision_policy": self.decision_policy,
        }


@dataclass(frozen=True, slots=True)
class RouterTrainingConfig:
    """Validated settings for a router-training run."""

    model: str
    tokenizer: str
    data_source: str | None
    split: str
    validation_split: str | None
    teacher_model: str
    student_model: str
    student_quantized_path: Path | None
    distillation_loss: str | None
    precision_candidates: tuple[int, ...]
    max_bit_width: int
    block_granularity: str
    output_dir: Path
    seed: int = 0
    overwrite: bool = False
    gpu_ids: tuple[int, ...] = ()
    device: str = "cpu"
    prompt_format: str | None = None
    training_data_limit: int = 2
    validation_data_limit: int = 2
    checkpoint_interval_steps: int = 1
    checkpoint_dir: Path | None = None
    logging: LoggingConfig = LoggingConfig()
    router: RouterHyperparameters = RouterHyperparameters()
    diagnostic: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RouterTrainingConfig":
        data = dict(value)
        router = RouterHyperparameters.from_mapping(_optional_mapping(data.get("router")))
        logging = LoggingConfig.from_mapping(_optional_mapping(data.get("logging")))
        checkpoint_interval = data.get(
            "checkpoint_interval_steps",
            logging.checkpoint_interval_steps or 1,
        )
        diagnostic = data.get("diagnostic", True)
        if not isinstance(diagnostic, bool):
            raise RouterTrainingError(
                "invalid_router_training_config",
                "diagnostic must be a boolean",
            )
        overwrite = data.get("overwrite", False)
        if not isinstance(overwrite, bool):
            raise RouterTrainingError(
                "invalid_router_training_config",
                "overwrite must be a boolean",
            )

        return cls(
            model=_coerce_non_empty_string(data.get("model"), field="model"),
            tokenizer=_coerce_non_empty_string(data.get("tokenizer"), field="tokenizer"),
            data_source=_optional_non_empty_string(data.get("data_source"), "data_source"),
            split=_coerce_non_empty_string(data.get("split"), field="split"),
            validation_split=_optional_non_empty_string(
                data.get("validation_split"),
                "validation_split",
            ),
            teacher_model=_coerce_non_empty_string(
                data.get("teacher_model"),
                field="teacher_model",
            ),
            student_model=_coerce_non_empty_string(
                data.get("student_model"),
                field="student_model",
            ),
            student_quantized_path=_optional_path(
                data.get("student_quantized_path"),
                field="student_quantized_path",
            ),
            distillation_loss=_optional_non_empty_string(
                data.get("distillation_loss"),
                "distillation_loss",
            ),
            precision_candidates=_coerce_precision_candidates(
                data.get("precision_candidates"),
                max_bit_width=_coerce_positive_int(
                    data.get("max_bit_width"),
                    field="max_bit_width",
                ),
            ),
            max_bit_width=_coerce_positive_int(
                data.get("max_bit_width"),
                field="max_bit_width",
            ),
            block_granularity=_coerce_non_empty_string(
                data.get("block_granularity"),
                field="block_granularity",
            ),
            output_dir=Path(
                _coerce_non_empty_string(data.get("output_dir"), field="output_dir")
            ),
            seed=_coerce_int(data.get("seed", 0), field="seed"),
            overwrite=overwrite,
            gpu_ids=_coerce_gpu_ids(data.get("gpu_ids", [])),
            device=_coerce_non_empty_string(data.get("device", "cpu"), field="device"),
            prompt_format=_optional_non_empty_string(
                data.get("prompt_format"),
                "prompt_format",
            ),
            training_data_limit=_coerce_positive_int(
                data.get("training_data_limit", 2),
                field="training_data_limit",
            ),
            validation_data_limit=_coerce_positive_int(
                data.get("validation_data_limit", data.get("training_data_limit", 2)),
                field="validation_data_limit",
            ),
            checkpoint_interval_steps=_coerce_positive_int(
                checkpoint_interval,
                field="checkpoint_interval_steps",
            ),
            checkpoint_dir=_optional_path(data.get("checkpoint_dir"), field="checkpoint_dir"),
            logging=logging,
            router=router,
            diagnostic=diagnostic,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "data_source": self.data_source,
            "split": self.split,
            "validation_split": self.validation_split,
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "student_quantized_path": (
                str(self.student_quantized_path) if self.student_quantized_path else None
            ),
            "distillation_loss": self.distillation_loss,
            "precision_candidates": list(self.precision_candidates),
            "max_bit_width": self.max_bit_width,
            "block_granularity": self.block_granularity,
            "output_dir": str(self.output_dir),
            "seed": self.seed,
            "overwrite": self.overwrite,
            "gpu_ids": list(self.gpu_ids),
            "device": self.device,
            "prompt_format": self.prompt_format,
            "training_data_limit": self.training_data_limit,
            "validation_data_limit": self.validation_data_limit,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
            "logging": self.logging.as_dict(),
            "router": self.router.as_dict(),
            "diagnostic": self.diagnostic,
        }

    @property
    def resolved_checkpoint_dir(self) -> Path:
        return self.checkpoint_dir or self.output_dir / "checkpoints"

    def to_run_config(
        self,
        *,
        model: str | None = None,
        split: str | None = None,
        mode: str = "fp16",
        validate_output: bool = True,
    ) -> RunConfig:
        return RunConfig.from_mapping(
            {
                "model": model or self.model,
                "tokenizer": self.tokenizer,
                "dataset": self.data_source or "",
                "split": split or self.split,
                "mode": mode,
                "precision_candidates": list(self.precision_candidates),
                "max_bit_width": self.max_bit_width,
                "block_granularity": self.block_granularity,
                "gpu_ids": list(self.gpu_ids),
                "seed": self.seed,
                "output_dir": str(self.output_dir),
                "overwrite": self.overwrite,
                "logging": self.logging.as_dict(),
                "router_diagnostic": self.diagnostic,
                "device": self.device,
                "prompt_format": self.prompt_format,
                "metric": "router_distillation_loss",
            },
            validate_output=validate_output,
        )


@dataclass(frozen=True, slots=True)
class RouterTrainingResult:
    manifest: RunManifest
    checkpoint: RouterCheckpoint
    checkpoint_path: Path
    target_audit_path: Path
    log_path: Path
    loss_records: tuple[LossRecord, ...]
    validation_metrics: dict[str, float]
    progress: ProgressState
    base_parameter_requires_grad: dict[str, bool]


@dataclass(frozen=True, slots=True)
class RouterTrainingTarget:
    example_id: str
    block_id: str
    feature: tuple[float, ...]
    target_probabilities: dict[int, float]
    candidate_distillation_costs: dict[int, float]
    candidate_efficiency_penalties: dict[int, float]
    candidate_total_costs: dict[int, float]


@dataclass(frozen=True, slots=True)
class CudaMemoryInfo:
    gpu_id: int
    free_bytes: int
    total_bytes: int


def load_router_training_config(path: str | Path) -> RouterTrainingConfig:
    """Load a JSON, TOML, or small YAML router-training config."""

    config_path = Path(path)
    try:
        if config_path.suffix == ".json":
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        elif config_path.suffix == ".toml":
            with config_path.open("rb") as handle:
                raw = tomllib.load(handle)
        elif config_path.suffix in {".yaml", ".yml"}:
            raw = _parse_small_yaml(config_path.read_text(encoding="utf-8"))
        else:
            raise RouterTrainingError(
                "unsupported_router_training_config",
                "supported config formats are .json, .toml, .yaml, and .yml",
            )
    except RouterTrainingError:
        raise
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise RouterTrainingError("router_training_config_read_failed", str(exc)) from exc

    if not isinstance(raw, Mapping):
        raise RouterTrainingError(
            "invalid_router_training_config",
            "top-level config must be an object",
        )
    return RouterTrainingConfig.from_mapping(raw)


def validate_router_training_preflight(config: RouterTrainingConfig) -> None:
    """Fail before launching a router-training job with missing core choices."""

    if not config.data_source:
        raise RouterTrainingError(
            "missing_training_data",
            "router training requires data_source",
        )
    if not config.distillation_loss:
        raise RouterTrainingError(
            "missing_distillation_loss",
            "router training requires a concrete distillation_loss",
        )
    if config.distillation_loss not in SUPPORTED_DISTILLATION_LOSSES:
        raise RouterTrainingError(
            "unsupported_distillation_loss",
            f"supported distillation losses are {sorted(SUPPORTED_DISTILLATION_LOSSES)}",
        )
    if not config.diagnostic and not Path(config.data_source).is_file():
        raise RouterTrainingError(
            "non_diagnostic_training_requires_file_data",
            "accepted router training requires a file-backed data_source",
        )
    if config.student_quantized_path is None:
        raise RouterTrainingError(
            "missing_student_quantized_path",
            "router training requires student_quantized_path",
        )
    if not config.student_quantized_path.exists():
        raise RouterTrainingError(
            "student_quantized_path_unavailable",
            f"{config.student_quantized_path} does not exist",
        )
    if config.device not in {"cpu", "cuda"}:
        raise RouterTrainingError(
            "invalid_device",
            "router training device must be 'cpu' or 'cuda'",
        )
    if config.device == "cpu" and config.gpu_ids:
        raise RouterTrainingError(
            "invalid_device",
            "cpu router training cannot select gpu_ids",
        )
    if config.device == "cuda" and not config.gpu_ids:
        raise RouterTrainingError(
            "invalid_device",
            "cuda router training requires at least one gpu_id",
        )
    if config.block_granularity != "mha_ffn":
        raise RouterTrainingError(
            "unsupported_block_granularity",
            "router training currently supports only mha_ffn block granularity",
        )
    try:
        load_benchmark_examples(
            config.data_source,
            split=config.split,
            limit=config.training_data_limit,
        )
        if config.validation_split:
            load_benchmark_examples(
                config.data_source,
                split=config.validation_split,
                limit=config.validation_data_limit,
            )
    except BenchmarkDataError as exc:
        raise RouterTrainingError("training_data_unavailable", str(exc)) from exc

    teacher = load_model_adapter(
        config.to_run_config(model=config.teacher_model, validate_output=False)
    )
    student = load_model_adapter(
        config.to_run_config(model=config.student_model, validate_output=False)
    )
    blocks = discover_mha_ffn_blocks(
        teacher.architecture_metadata,
        supported_bit_widths=config.precision_candidates,
    )
    student_blocks = discover_mha_ffn_blocks(
        student.architecture_metadata,
        supported_bit_widths=config.precision_candidates,
    )
    if tuple(block.block_id for block in blocks) != tuple(
        block.block_id for block in student_blocks
    ):
        raise RouterTrainingError(
            "teacher_student_block_mismatch",
            "teacher and student block IDs must match",
        )
    _validate_cuda_training_capacity(config, teacher=teacher, student=student)
    _load_quantized_artifacts(config, blocks)


def run_router_training(
    config: RouterTrainingConfig,
    *,
    fail_at_step: int | None = None,
) -> RouterTrainingResult:
    """Run router training and persist logs/checkpoint metadata."""

    validate_router_training_preflight(config)
    run_config = config.to_run_config()
    manifest = create_run_manifest(run_config, run_id="router-train")
    monitor = ConsoleProgressMonitor(
        run_id=manifest.run_id,
        enabled=config.logging.console,
    )

    with open_run_log(manifest, name="router_train") as writer:
        try:
            return _run_router_training_started(
                config,
                manifest=manifest,
                writer=writer,
                monitor=monitor,
                fail_at_step=fail_at_step,
            )
        except RouterTrainingError as exc:
            _record_training_failure(
                manifest,
                writer,
                monitor,
                code=exc.code,
                message=exc.message,
            )
            raise
        except Exception as exc:
            code = getattr(exc, "code", "router_training_failed")
            if not isinstance(code, str):
                code = "router_training_failed"
            _record_training_failure(
                manifest,
                writer,
                monitor,
                code=code,
                message=str(exc),
            )
            raise RouterTrainingError(code, str(exc)) from exc


def _run_router_training_started(
    config: RouterTrainingConfig,
    *,
    manifest: RunManifest,
    writer: JsonlLogWriter,
    monitor: ConsoleProgressMonitor,
    fail_at_step: int | None,
) -> RouterTrainingResult:
    start_event = LogEvent(
        event_type=EventType.RUN_START.value,
        run_id=manifest.run_id,
        module=MODULE_NAME,
        status=RunStatus.RUNNING.value,
        message="router training started",
        total_examples=config.training_data_limit,
        selected_gpu_ids=config.gpu_ids,
        details={"router_training_config": config.as_dict()},
    )
    writer.record(start_event)
    monitor.handle(start_event)
    if config.diagnostic:
        warning_event = LogEvent.warning(
            run_id=manifest.run_id,
            module=MODULE_NAME,
            message="diagnostic router training is not accepted QAQ evidence",
            details={"diagnostic_training": True},
        )
        writer.record(warning_event)
        monitor.handle(warning_event)

    teacher = load_model_adapter(
        config.to_run_config(model=config.teacher_model, validate_output=False)
    )
    student = load_model_adapter(
        config.to_run_config(model=config.student_model, validate_output=False)
    )
    _freeze_and_validate(teacher)
    _freeze_and_validate(student)

    blocks = discover_mha_ffn_blocks(
        teacher.architecture_metadata,
        supported_bit_widths=config.precision_candidates,
    )
    student_blocks = discover_mha_ffn_blocks(
        student.architecture_metadata,
        supported_bit_widths=config.precision_candidates,
    )
    if tuple(block.block_id for block in blocks) != tuple(
        block.block_id for block in student_blocks
    ):
        raise RouterTrainingError(
            "teacher_student_block_mismatch",
            "teacher and student block IDs must match",
        )
    quantized_artifacts = _load_quantized_artifacts(config, blocks)
    candidate_distortions = _candidate_artifact_distortions(
        config,
        blocks=blocks,
        artifacts_by_block=quantized_artifacts,
    )

    examples = teacher.load_examples(
        config.to_run_config(model=config.teacher_model, validate_output=False),
        limit=config.training_data_limit,
    )
    run_config = config.to_run_config(model=config.teacher_model, validate_output=False)
    batch = teacher.build_batch(run_config, examples)
    block_ids = tuple(block.block_id for block in blocks)
    teacher_output = teacher.reference_forward(batch, block_ids=block_ids)
    student_output = student.reference_forward(batch, block_ids=block_ids)
    training_targets = _build_router_targets(
        config,
        example_ids=batch.example_ids,
        teacher_output=teacher_output,
        student_output=student_output,
        candidate_distortions=candidate_distortions,
        block_ids=block_ids,
    )
    validation_targets: tuple[RouterTrainingTarget, ...] = ()
    validation_sample_count = 0
    if config.validation_split:
        validation_examples = teacher.load_examples(
            config.to_run_config(
                model=config.teacher_model,
                split=config.validation_split,
                validate_output=False,
            ),
            limit=config.validation_data_limit,
        )
        validation_batch = teacher.build_batch(
            config.to_run_config(
                model=config.teacher_model,
                split=config.validation_split,
                validate_output=False,
            ),
            validation_examples,
        )
        validation_teacher_output = teacher.reference_forward(
            validation_batch,
            block_ids=block_ids,
        )
        validation_student_output = student.reference_forward(
            validation_batch,
            block_ids=block_ids,
        )
        validation_targets = _build_router_targets(
            config,
            example_ids=validation_batch.example_ids,
            teacher_output=validation_teacher_output,
            student_output=validation_student_output,
            candidate_distortions=candidate_distortions,
            block_ids=block_ids,
        )
        validation_sample_count = len(validation_examples)

    target_audit_path = _save_target_audit(
        config,
        manifest=manifest,
        training_targets=training_targets,
        validation_targets=validation_targets,
        training_sample_count=len(examples),
        validation_sample_count=validation_sample_count,
    )

    checkpoint = _initial_checkpoint(
        config,
        adapter=teacher,
        block_ids=block_ids,
        completed_step=0,
        latest_loss=None,
        training_sample_count=len(examples),
        validation_sample_count=validation_sample_count,
        target_record_count=len(training_targets),
        validation_metrics={},
        parameter_update_l2=0.0,
    )
    loss_records: list[LossRecord] = []
    validation_metrics: dict[str, float] = {}
    parameter_update_l2 = 0.0

    with TimingMeasurement(name="router_training") as timing:
        for step in range(1, config.router.max_steps + 1):
            if fail_at_step == step:
                raise RouterTrainingError(
                    "controlled_training_failure",
                    f"controlled failure at step {step}",
                )
            objective_samples, gradients = _objective_samples_and_gradients(
                checkpoint,
                training_targets,
                temperature=config.router.temperature,
            )
            loss_record = compute_router_objective_loss(
                samples=objective_samples,
                objective=config.distillation_loss or "",
                step=step,
                learning_rate=config.router.learning_rate,
            )
            loss_records.append(loss_record)
            checkpoint, update_l2 = _apply_router_gradient_update(
                checkpoint,
                gradients=gradients,
                learning_rate=config.router.learning_rate,
                normalizer=len(training_targets),
            )
            parameter_update_l2 += update_l2
            validation_metrics = _compute_validation_metrics(
                checkpoint,
                validation_targets or training_targets,
                objective=config.distillation_loss or "",
                step=step,
                learning_rate=config.router.learning_rate,
            )
            checkpoint = _with_training_metadata(
                checkpoint,
                config,
                completed_step=step,
                latest_loss=loss_record,
                training_sample_count=len(examples),
                validation_sample_count=validation_sample_count,
                target_record_count=len(training_targets),
                validation_metrics=validation_metrics,
                parameter_update_l2=parameter_update_l2,
            )

            progress_event = LogEvent.progress(
                run_id=manifest.run_id,
                module=MODULE_NAME,
                step=step,
                epoch=1,
                processed_examples=len(examples),
                total_examples=len(examples),
                loss=loss_record.loss,
                learning_rate=config.router.learning_rate,
                elapsed_seconds=timing.clock() - (timing.start_seconds or 0.0),
                details={
                    "loss": loss_record.as_dict(),
                    "objective": config.distillation_loss,
                    "training_sample_count": len(examples),
                    "target_record_count": len(training_targets),
                    "validation_metrics": validation_metrics,
                },
            )
            writer.record(progress_event)
            monitor.handle(progress_event)

            if _should_checkpoint(config, step):
                checkpoint_path = _save_checkpoint(
                    checkpoint,
                    config=config,
                    manifest=manifest,
                    writer=writer,
                    monitor=monitor,
                    step=step,
                )

    if "checkpoint_path" not in locals():
        checkpoint_path = _save_checkpoint(
            checkpoint,
            config=config,
            manifest=manifest,
            writer=writer,
            monitor=monitor,
            step=config.router.max_steps,
        )

    completion_event = LogEvent(
        event_type=EventType.COMPLETION.value,
        run_id=manifest.run_id,
        module=MODULE_NAME,
        status=RunStatus.COMPLETED.value,
        message="router training completed",
        step=config.router.max_steps,
        elapsed_seconds=timing.elapsed_seconds,
        checkpoint_path=str(checkpoint_path),
        details={
            "objective": config.distillation_loss,
            "training_sample_count": len(examples),
            "target_record_count": len(training_targets),
            "target_audit_path": str(target_audit_path),
            "validation_metrics": validation_metrics,
            "parameter_update_l2": parameter_update_l2,
        },
    )
    writer.record(completion_event)
    writer.flush()
    monitor.handle(completion_event)
    manifest.mark_completed()
    log_path = Path(manifest.artifact_paths["router_train_log"])
    return RouterTrainingResult(
        manifest=manifest,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        target_audit_path=target_audit_path,
        log_path=log_path,
        loss_records=tuple(loss_records),
        validation_metrics=validation_metrics,
        progress=monitor.state,
        base_parameter_requires_grad={
            parameter.name: parameter.requires_grad for parameter in teacher.parameters()
        },
    )


def _initial_checkpoint(
    config: RouterTrainingConfig,
    *,
    adapter: FakeCausalLMAdapter,
    block_ids: tuple[str, ...],
    completed_step: int,
    latest_loss: LossRecord | None,
    training_sample_count: int,
    validation_sample_count: int,
    target_record_count: int,
    validation_metrics: dict[str, float],
    parameter_update_l2: float,
) -> RouterCheckpoint:
    hidden_size = adapter.hidden_size
    candidate_count = len(config.precision_candidates)
    zero_params = RouterBlockParameters(
        weights=tuple((0.0,) * hidden_size for _ in range(candidate_count)),
        bias=(0.0,) * candidate_count,
    )
    metadata = RouterCheckpointMetadata(
        checkpoint_id="router-train-step-0000",
        model_id=config.model,
        block_ids=block_ids,
        candidate_bit_widths=config.precision_candidates,
        feature_source=adapter.feature_source,
        hidden_size=hidden_size,
        temperature=config.router.temperature,
        decision_policy=config.router.decision_policy,
        max_bit_width=config.max_bit_width,
        diagnostic=config.diagnostic,
        training_metadata=_training_metadata(
            config,
            completed_step=completed_step,
            latest_loss=latest_loss,
            training_sample_count=training_sample_count,
            validation_sample_count=validation_sample_count,
            target_record_count=target_record_count,
            validation_metrics=validation_metrics,
            parameter_update_l2=parameter_update_l2,
        ),
    )
    return RouterCheckpoint(
        metadata=metadata,
        parameters={block_id: zero_params for block_id in block_ids},
    )


def _load_quantized_artifacts(
    config: RouterTrainingConfig,
    blocks: tuple[Any, ...],
) -> dict[str, tuple[BitPlaneArtifact | TensorBitPlaneArtifact, ...]]:
    if config.student_quantized_path is None:
        raise RouterTrainingError(
            "missing_student_quantized_path",
            "router training requires student_quantized_path",
        )
    paths = _artifact_paths(config.student_quantized_path)
    by_block: dict[str, list[BitPlaneArtifact | TensorBitPlaneArtifact]] = {block.block_id: [] for block in blocks}
    known_blocks = set(by_block)
    tensor_names_by_block = {
        block.block_id: set(block.tensor_names)
        for block in blocks
    }

    for path in paths:
        try:
            artifact = (
                load_tensor_bitplane_artifact(path)
                if is_tensor_bitplane_artifact_path(path)
                else load_bitplane_artifact(path)
            )
        except BitPlaneError as exc:
            raise RouterTrainingError("student_artifact_invalid", str(exc)) from exc
        except TensorBitPlaneError as exc:
            raise RouterTrainingError("student_artifact_invalid", str(exc)) from exc
        metadata = artifact.metadata
        if metadata.block_id not in known_blocks:
            continue
        if metadata.model_id != config.student_model:
            raise RouterTrainingError(
                "student_artifact_model_mismatch",
                f"{path} model_id {metadata.model_id!r} does not match student_model {config.student_model!r}",
            )
        if metadata.tensor_name not in tensor_names_by_block[metadata.block_id]:
            raise RouterTrainingError(
                "student_artifact_tensor_mismatch",
                f"{path} tensor {metadata.tensor_name!r} is not owned by {metadata.block_id}",
            )
        if metadata.max_bit_width != config.max_bit_width:
            raise RouterTrainingError(
                "student_artifact_bit_width_mismatch",
                f"{path} max_bit_width {metadata.max_bit_width} does not match config {config.max_bit_width}",
            )
        by_block[metadata.block_id].append(artifact)

    missing = [block_id for block_id, artifacts in by_block.items() if not artifacts]
    if missing:
        raise RouterTrainingError(
            "missing_student_artifacts",
            f"student_quantized_path lacks artifacts for blocks: {missing}",
        )
    return {
        block_id: tuple(artifacts)
        for block_id, artifacts in by_block.items()
    }


def _artifact_paths(path: Path) -> tuple[Path, ...]:
    if path.is_dir():
        paths = tuple(
            sorted(
                item
                for item in (
                    tuple(
                        candidate
                        for candidate in path.rglob("*.json")
                        if _looks_like_json_bitplane_artifact(candidate)
                    )
                    + tuple(path.rglob("*.qaq.safetensors"))
                )
                if item.is_file()
            )
        )
    elif path.is_file():
        if is_tensor_bitplane_artifact_path(path):
            paths = (path,)
        else:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RouterTrainingError("student_artifact_index_read_failed", str(exc)) from exc
            if isinstance(raw, dict) and "metadata" in raw and "planes" in raw:
                paths = (path,)
            elif isinstance(raw, dict):
                paths = _artifact_paths_from_index(raw, base_dir=path.parent)
            else:
                raise RouterTrainingError(
                    "invalid_student_artifact_index",
                    "artifact index must be an object",
                )
    else:
        raise RouterTrainingError(
            "student_quantized_path_unavailable",
            f"{path} does not exist",
        )
    if not paths:
        raise RouterTrainingError(
            "missing_student_artifacts",
            f"{path} does not contain bit-plane artifact files",
        )
    return paths


def _looks_like_json_bitplane_artifact(path: Path) -> bool:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(raw, dict) and "metadata" in raw and "planes" in raw


def _artifact_paths_from_index(
    value: Mapping[str, Any],
    *,
    base_dir: Path,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for block_id, refs in value.items():
        if not isinstance(block_id, str) or not isinstance(refs, Mapping):
            raise RouterTrainingError(
                "invalid_student_artifact_index",
                "artifact index must map block IDs to bit-width maps",
            )
        for raw_path in refs.values():
            if not isinstance(raw_path, str) or not raw_path:
                raise RouterTrainingError(
                    "invalid_student_artifact_index",
                    "artifact paths must be non-empty strings",
                )
            candidate = Path(raw_path)
            paths.append(candidate if candidate.is_absolute() else base_dir / candidate)
    return tuple(sorted(set(paths)))


def _candidate_artifact_distortions(
    config: RouterTrainingConfig,
    *,
    blocks: tuple[Any, ...],
    artifacts_by_block: Mapping[str, tuple[BitPlaneArtifact | TensorBitPlaneArtifact, ...]],
) -> dict[str, dict[int, float]]:
    distortions: dict[str, dict[int, float]] = {}
    for block in blocks:
        block_distortions: dict[int, float] = {}
        artifacts = artifacts_by_block[block.block_id]
        for bit_width in config.precision_candidates:
            values = []
            for artifact in artifacts:
                if isinstance(artifact, TensorBitPlaneArtifact):
                    values.append(
                        normalized_tensor_reconstruction_delta(
                            artifact,
                            bit_width=bit_width,
                            model_id=config.student_model,
                            block_id=block.block_id,
                        )
                    )
                else:
                    full = reconstruct_weight(
                        artifact,
                        bit_width=config.max_bit_width,
                        model_id=config.student_model,
                        block_id=block.block_id,
                    )
                    candidate = reconstruct_weight(
                        artifact,
                        bit_width=bit_width,
                        model_id=config.student_model,
                        block_id=block.block_id,
                    )
                    qmax = max(1, artifact.metadata.quantization.qmax)
                    values.append(
                        _normalized_mean_abs_delta(
                            candidate.quantized_values,
                            full.quantized_values,
                            normalizer=qmax,
                        )
                    )
            block_distortions[bit_width] = sum(values) / len(values)
        distortions[block.block_id] = block_distortions
    return distortions


def _build_router_targets(
    config: RouterTrainingConfig,
    *,
    example_ids: tuple[str, ...],
    teacher_output: Any,
    student_output: Any,
    candidate_distortions: Mapping[str, Mapping[int, float]],
    block_ids: tuple[str, ...],
) -> tuple[RouterTrainingTarget, ...]:
    targets: list[RouterTrainingTarget] = []
    for sample_index, example_id in enumerate(example_ids):
        base_mse = _row_mse(
            teacher_output.logits[sample_index],
            student_output.logits[sample_index],
        )
        for block_id in block_ids:
            feature = teacher_output.hidden_states.by_block[block_id][sample_index]
            sensitivity = _feature_sensitivity(feature)
            distillation_costs: dict[int, float] = {}
            efficiency_penalties: dict[int, float] = {}
            total_costs: dict[int, float] = {}
            for bit_width in config.precision_candidates:
                distortion = candidate_distortions[block_id][bit_width]
                distillation_cost = base_mse + sensitivity * distortion
                efficiency_penalty = (
                    config.router.bit_cost_weight * bit_width / config.max_bit_width
                )
                distillation_costs[bit_width] = distillation_cost
                efficiency_penalties[bit_width] = efficiency_penalty
                total_costs[bit_width] = distillation_cost + efficiency_penalty
            target_probabilities = softmax_from_costs(
                total_costs,
                temperature=config.router.target_temperature,
            )
            targets.append(
                RouterTrainingTarget(
                    example_id=example_id,
                    block_id=block_id,
                    feature=feature,
                    target_probabilities=target_probabilities,
                    candidate_distillation_costs=distillation_costs,
                    candidate_efficiency_penalties=efficiency_penalties,
                    candidate_total_costs=total_costs,
                )
            )
    if not targets:
        raise RouterTrainingError(
            "missing_router_targets",
            "router objective produced no targets",
        )
    return tuple(targets)


def _objective_samples_and_gradients(
    checkpoint: RouterCheckpoint,
    targets: tuple[RouterTrainingTarget, ...],
    *,
    temperature: float,
) -> tuple[tuple[RouterObjectiveSample, ...], dict[str, dict[str, list[list[float]] | list[float]]]]:
    gradients: dict[str, dict[str, list[list[float]] | list[float]]] = {}
    for block_id, params in checkpoint.parameters.items():
        gradients[block_id] = {
            "weights": [
                [0.0 for _ in range(checkpoint.metadata.hidden_size)]
                for _ in params.weights
            ],
            "bias": [0.0 for _ in params.bias],
        }

    samples: list[RouterObjectiveSample] = []
    for target in targets:
        raw_scores = score_block(
            checkpoint,
            block_id=target.block_id,
            feature=target.feature,
        )
        router_probabilities = normalize_scores(raw_scores, temperature=temperature)
        samples.append(
            RouterObjectiveSample(
                target_probabilities=target.target_probabilities,
                router_probabilities=router_probabilities,
                candidate_distillation_costs=target.candidate_distillation_costs,
                candidate_efficiency_penalties=target.candidate_efficiency_penalties,
            )
        )
        weight_gradients = gradients[target.block_id]["weights"]
        bias_gradients = gradients[target.block_id]["bias"]
        assert isinstance(weight_gradients, list)
        assert isinstance(bias_gradients, list)
        for candidate_index, bit_width in enumerate(checkpoint.metadata.candidate_bit_widths):
            grad_score = (
                router_probabilities[bit_width] - target.target_probabilities[bit_width]
            ) / temperature
            bias_gradients[candidate_index] += grad_score
            for feature_index, feature_value in enumerate(target.feature):
                weight_gradients[candidate_index][feature_index] += grad_score * feature_value
    return tuple(samples), gradients


def _save_target_audit(
    config: RouterTrainingConfig,
    *,
    manifest: RunManifest,
    training_targets: tuple[RouterTrainingTarget, ...],
    validation_targets: tuple[RouterTrainingTarget, ...],
    training_sample_count: int,
    validation_sample_count: int,
) -> Path:
    audit_path = config.output_dir / "router_targets.json"
    payload = {
        "objective": config.distillation_loss,
        "training_data_source": config.data_source,
        "training_split": config.split,
        "validation_split": config.validation_split,
        "training_sample_count": training_sample_count,
        "validation_sample_count": validation_sample_count,
        "target_record_count": len(training_targets),
        "validation_target_record_count": len(validation_targets),
        "precision_candidates": list(config.precision_candidates),
        "max_bit_width": config.max_bit_width,
        "bit_cost_weight": config.router.bit_cost_weight,
        "target_temperature": config.router.target_temperature,
        "diagnostic_training": config.diagnostic,
        "training_targets": [_target_as_dict(target) for target in training_targets],
        "validation_targets": [_target_as_dict(target) for target in validation_targets],
    }
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise RouterTrainingError("router_target_audit_write_failed", str(exc)) from exc
    manifest.artifact_paths["router_targets"] = str(audit_path)
    manifest.write()
    return audit_path


def _target_as_dict(target: RouterTrainingTarget) -> dict[str, Any]:
    return {
        "example_id": target.example_id,
        "block_id": target.block_id,
        "feature": list(target.feature),
        "target_probabilities": _float_map(target.target_probabilities),
        "candidate_distillation_costs": _float_map(target.candidate_distillation_costs),
        "candidate_efficiency_penalties": _float_map(
            target.candidate_efficiency_penalties
        ),
        "candidate_total_costs": _float_map(target.candidate_total_costs),
    }


def _float_map(values: Mapping[int, float]) -> dict[str, float]:
    return {str(key): float(values[key]) for key in sorted(values)}


def _apply_router_gradient_update(
    checkpoint: RouterCheckpoint,
    *,
    gradients: Mapping[str, Mapping[str, list[list[float]] | list[float]]],
    learning_rate: float,
    normalizer: int,
) -> tuple[RouterCheckpoint, float]:
    if normalizer <= 0:
        raise RouterTrainingError("invalid_gradient_normalizer", "normalizer must be positive")
    updated: dict[str, RouterBlockParameters] = {}
    update_square_sum = 0.0
    for block_id, params in checkpoint.parameters.items():
        raw_weight_gradients = gradients[block_id]["weights"]
        raw_bias_gradients = gradients[block_id]["bias"]
        assert isinstance(raw_weight_gradients, list)
        assert isinstance(raw_bias_gradients, list)
        new_weights: list[tuple[float, ...]] = []
        for row, gradient_row in zip(params.weights, raw_weight_gradients, strict=True):
            new_row = []
            for value, gradient in zip(row, gradient_row, strict=True):
                delta = learning_rate * gradient / normalizer
                update_square_sum += delta * delta
                new_row.append(value - delta)
            new_weights.append(tuple(new_row))
        new_bias = []
        for value, gradient in zip(params.bias, raw_bias_gradients, strict=True):
            delta = learning_rate * gradient / normalizer
            update_square_sum += delta * delta
            new_bias.append(value - delta)
        updated[block_id] = RouterBlockParameters(
            weights=tuple(new_weights),
            bias=tuple(new_bias),
        )
    return RouterCheckpoint(metadata=checkpoint.metadata, parameters=updated), math.sqrt(
        update_square_sum
    )


def _compute_validation_metrics(
    checkpoint: RouterCheckpoint,
    targets: tuple[RouterTrainingTarget, ...],
    *,
    objective: str,
    step: int,
    learning_rate: float,
) -> dict[str, float]:
    samples, _ = _objective_samples_and_gradients(
        checkpoint,
        targets,
        temperature=checkpoint.metadata.temperature,
    )
    record = compute_router_objective_loss(
        samples=samples,
        objective=objective,
        step=step,
        learning_rate=learning_rate,
    )
    return {
        "validation_loss": record.loss,
        "validation_distillation_cost": record.distillation_loss,
        "validation_efficiency_penalty": record.efficiency_penalty,
    }


def _row_mse(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        raise RouterTrainingError("logit_shape_mismatch", "teacher/student logits must match")
    total = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        delta = left_value - right_value
        total += delta * delta
    return total / len(left)


def _feature_sensitivity(feature: tuple[float, ...]) -> float:
    if not feature:
        raise RouterTrainingError("missing_router_feature", "router feature cannot be empty")
    return max(sum(abs(value) for value in feature) / len(feature), 1e-6)


def _normalized_mean_abs_delta(left: Any, right: Any, *, normalizer: int) -> float:
    left_flat = flatten_tensor(left)
    right_flat = flatten_tensor(right)
    if len(left_flat) != len(right_flat) or not left_flat:
        raise RouterTrainingError(
            "artifact_shape_mismatch",
            "candidate and full-bit reconstructions must have matching shapes",
        )
    total = sum(abs(float(a) - float(b)) for a, b in zip(left_flat, right_flat, strict=True))
    return total / len(left_flat) / normalizer


def _validate_cuda_training_capacity(
    config: RouterTrainingConfig,
    *,
    teacher: Any,
    student: Any,
) -> None:
    if config.device != "cuda":
        return
    if not config.gpu_ids:
        raise RouterTrainingError(
            "invalid_device",
            "cuda router training requires at least one gpu_id",
        )
    gpu_id = config.gpu_ids[0]
    memory = _cuda_memory_info(gpu_id)
    teacher_bytes = _estimated_model_weight_bytes(teacher)
    student_bytes = _estimated_model_weight_bytes(student)
    if _requires_hf_memory_estimate(teacher) and teacher_bytes is None:
        raise RouterTrainingError(
            "cuda_memory_estimate_unavailable",
            f"could not estimate teacher model weight bytes for {config.teacher_model!r}",
        )
    if _requires_hf_memory_estimate(student) and student_bytes is None:
        raise RouterTrainingError(
            "cuda_memory_estimate_unavailable",
            f"could not estimate student model weight bytes for {config.student_model!r}",
        )
    if teacher_bytes is None or student_bytes is None:
        return

    required_bytes = teacher_bytes + student_bytes + CUDA_MODEL_LOAD_MARGIN_BYTES
    if required_bytes > memory.free_bytes:
        raise RouterTrainingError(
            "insufficient_cuda_memory",
            "router training requires at least "
            f"{_bytes_to_gib(required_bytes):.2f} GiB of free CUDA memory for "
            "the current teacher+student model-weight loading path before "
            f"activations, but cuda:{gpu_id} reports "
            f"{_bytes_to_gib(memory.free_bytes):.2f} GiB free "
            f"of {_bytes_to_gib(memory.total_bytes):.2f} GiB total. "
            f"teacher_model_weight={_bytes_to_gib(teacher_bytes):.2f} GiB, "
            f"student_model_weight={_bytes_to_gib(student_bytes):.2f} GiB. "
            "Use a GPU with enough memory or implement shared/sequential "
            "teacher-student execution before running this config.",
        )


def _cuda_memory_info(gpu_id: int) -> CudaMemoryInfo:
    try:
        import torch
    except Exception as exc:
        raise RouterTrainingError(
            "cuda_unavailable",
            f"torch is required for cuda router training: {exc}",
        ) from exc
    try:
        if not torch.cuda.is_available():
            raise RouterTrainingError(
                "cuda_unavailable",
                "torch reports CUDA is unavailable",
            )
        device_count = int(torch.cuda.device_count())
        if gpu_id >= device_count:
            raise RouterTrainingError(
                "cuda_device_unavailable",
                f"cuda:{gpu_id} was requested but only {device_count} CUDA devices are visible",
            )
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
    except RouterTrainingError:
        raise
    except Exception as exc:
        raise RouterTrainingError("cuda_memory_probe_failed", str(exc)) from exc
    return CudaMemoryInfo(
        gpu_id=gpu_id,
        free_bytes=int(free_bytes),
        total_bytes=int(total_bytes),
    )


def _estimated_model_weight_bytes(adapter: Any) -> int | None:
    model_ref = getattr(adapter, "model_ref", None)
    if not isinstance(model_ref, str) or not model_ref:
        return None
    model_path = Path(model_ref)
    if not model_path.is_dir():
        return None

    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        metadata = raw.get("metadata") if isinstance(raw, Mapping) else None
        total_size = metadata.get("total_size") if isinstance(metadata, Mapping) else None
        if isinstance(total_size, int) and total_size > 0:
            return total_size

    safetensor_paths = tuple(sorted(model_path.glob("*.safetensors")))
    if not safetensor_paths:
        return None
    try:
        return sum(path.stat().st_size for path in safetensor_paths)
    except OSError:
        return None


def _requires_hf_memory_estimate(adapter: Any) -> bool:
    model_ref = getattr(adapter, "model_ref", None)
    framework = getattr(getattr(adapter, "architecture_metadata", None), "framework", None)
    return isinstance(model_ref, str) and framework == "transformers_llama"


def _bytes_to_gib(value: int) -> float:
    return float(value) / float(1024**3)


def _with_training_metadata(
    checkpoint: RouterCheckpoint,
    config: RouterTrainingConfig,
    *,
    completed_step: int,
    latest_loss: LossRecord | None,
    training_sample_count: int,
    validation_sample_count: int,
    target_record_count: int,
    validation_metrics: dict[str, float],
    parameter_update_l2: float,
) -> RouterCheckpoint:
    metadata = replace(
        checkpoint.metadata,
        checkpoint_id=f"router-train-step-{completed_step:04d}",
        training_metadata=_training_metadata(
            config,
            completed_step=completed_step,
            latest_loss=latest_loss,
            training_sample_count=training_sample_count,
            validation_sample_count=validation_sample_count,
            target_record_count=target_record_count,
            validation_metrics=validation_metrics,
            parameter_update_l2=parameter_update_l2,
        ),
    )
    return RouterCheckpoint(metadata=metadata, parameters=checkpoint.parameters)


def _training_metadata(
    config: RouterTrainingConfig,
    *,
    completed_step: int,
    latest_loss: LossRecord | None,
    training_sample_count: int,
    validation_sample_count: int,
    target_record_count: int,
    validation_metrics: dict[str, float],
    parameter_update_l2: float,
) -> dict[str, Any]:
    return {
        "training_data_source": config.data_source,
        "training_split": config.split,
        "validation_split": config.validation_split,
        "training_sample_count": training_sample_count,
        "validation_sample_count": validation_sample_count,
        "target_record_count": target_record_count,
        "teacher_model": config.teacher_model,
        "student_model": config.student_model,
        "student_quantized_path": (
            str(config.student_quantized_path) if config.student_quantized_path else None
        ),
        "distillation_loss": config.distillation_loss,
        "objective": config.distillation_loss,
        "objective_assumption": ROUTER_COST_CROSS_ENTROPY,
        "learning_rate": config.router.learning_rate,
        "target_temperature": config.router.target_temperature,
        "bit_cost_weight": config.router.bit_cost_weight,
        "max_steps": config.router.max_steps,
        "checkpoint_interval_steps": config.checkpoint_interval_steps,
        "completed_step": completed_step,
        "latest_loss": latest_loss.as_dict() if latest_loss else None,
        "latest_validation_metrics": dict(validation_metrics),
        "parameter_update_l2": parameter_update_l2,
        "diagnostic_training": config.diagnostic,
    }


def _save_checkpoint(
    checkpoint: RouterCheckpoint,
    *,
    config: RouterTrainingConfig,
    manifest: RunManifest,
    writer: JsonlLogWriter,
    monitor: ConsoleProgressMonitor,
    step: int,
) -> Path:
    try:
        checkpoint_path = save_router_checkpoint(
            checkpoint,
            config.resolved_checkpoint_dir / f"router_step_{step:04d}.json",
        )
    except OSError as exc:
        raise RouterTrainingError("router_checkpoint_write_failed", str(exc)) from exc
    manifest.artifact_paths["router_checkpoint"] = str(checkpoint_path)
    manifest.write()
    event = LogEvent(
        event_type=EventType.CHECKPOINT.value,
        run_id=manifest.run_id,
        module=MODULE_NAME,
        status=RunStatus.RUNNING.value,
        step=step,
        checkpoint_path=str(checkpoint_path),
        message="router checkpoint saved",
        details={"checkpoint_id": checkpoint.metadata.checkpoint_id},
    )
    writer.record(event)
    monitor.handle(event)
    return checkpoint_path


def _freeze_and_validate(adapter: FakeCausalLMAdapter) -> None:
    adapter.freeze_base_model()
    trainable = tuple(
        parameter.name for parameter in adapter.parameters() if parameter.requires_grad
    )
    if trainable:
        raise RouterTrainingError(
            "base_model_not_frozen",
            f"base model parameters remained trainable: {list(trainable)}",
        )


def _should_checkpoint(config: RouterTrainingConfig, step: int) -> bool:
    return (
        step % config.checkpoint_interval_steps == 0
        or step == config.router.max_steps
    )


def _record_training_failure(
    manifest: RunManifest,
    writer: JsonlLogWriter,
    monitor: ConsoleProgressMonitor,
    *,
    code: str,
    message: str,
) -> None:
    event = LogEvent.error(
        run_id=manifest.run_id,
        module=MODULE_NAME,
        code=code,
        message=message,
    )
    writer.record(event)
    writer.flush()
    monitor.handle(event)
    manifest.mark_failed(code=code, message=message)


def _parse_small_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_map: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise RouterTrainingError(
                "router_training_config_parse_failed",
                f"invalid YAML line: {raw_line}",
            )
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise RouterTrainingError(
                "router_training_config_parse_failed",
                f"invalid YAML key: {raw_line}",
            )
        if indent == 0:
            if raw_value.strip():
                result[key] = _parse_scalar(raw_value.strip())
                current_map = None
            else:
                result[key] = {}
                current_map = result[key]
        else:
            if current_map is None:
                raise RouterTrainingError(
                    "router_training_config_parse_failed",
                    f"unexpected nested YAML line: {raw_line}",
                )
            current_map[key] = _parse_scalar(raw_value.strip())
    return result


def _parse_scalar(value: str) -> Any:
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "None"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RouterTrainingError(
            "invalid_router_training_config",
            "nested config sections must be objects",
        )
    return value


def _coerce_non_empty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RouterTrainingError(
            "invalid_router_training_config",
            f"{field} must be a non-empty string",
        )
    return value


def _optional_non_empty_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _coerce_non_empty_string(value, field=field)


def _optional_path(value: Any, *, field: str) -> Path | None:
    raw = _optional_non_empty_string(value, field)
    return Path(raw) if raw is not None else None


def _coerce_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RouterTrainingError(
            "invalid_router_training_config",
            f"{field} must be an integer",
        )
    return value


def _coerce_positive_int(value: Any, *, field: str) -> int:
    coerced = _coerce_int(value, field=field)
    if coerced <= 0:
        raise RouterTrainingError(
            "invalid_router_training_config",
            f"{field} must be positive",
        )
    return coerced


def _coerce_positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RouterTrainingError(
            "invalid_router_training_config",
            f"{field} must be numeric",
        )
    coerced = float(value)
    if coerced <= 0:
        raise RouterTrainingError(
            "invalid_router_training_config",
            f"{field} must be positive",
        )
    return coerced


def _coerce_non_negative_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RouterTrainingError(
            "invalid_router_training_config",
            f"{field} must be numeric",
        )
    coerced = float(value)
    if coerced < 0 or not math.isfinite(coerced):
        raise RouterTrainingError(
            "invalid_router_training_config",
            f"{field} must be finite and non-negative",
        )
    return coerced


def _coerce_gpu_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list | tuple):
        raise RouterTrainingError(
            "invalid_router_training_config",
            "gpu_ids must be a list of integers",
        )
    gpu_ids = tuple(_coerce_int(item, field="gpu_ids") for item in value)
    if len(set(gpu_ids)) != len(gpu_ids) or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise RouterTrainingError(
            "invalid_router_training_config",
            "gpu_ids must be unique non-negative integers",
        )
    return gpu_ids


def _coerce_precision_candidates(value: Any, *, max_bit_width: int) -> tuple[int, ...]:
    if not isinstance(value, list | tuple):
        raise RouterTrainingError(
            "invalid_router_training_config",
            "precision_candidates must be a list of integers",
        )
    candidates = tuple(_coerce_positive_int(item, field="precision_candidates") for item in value)
    if not candidates or len(set(candidates)) != len(candidates):
        raise RouterTrainingError(
            "invalid_router_training_config",
            "precision_candidates must be unique and non-empty",
        )
    if any(candidate > max_bit_width for candidate in candidates):
        raise RouterTrainingError(
            "invalid_router_training_config",
            "precision_candidates cannot exceed max_bit_width",
        )
    return tuple(sorted(candidates))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a QAQ router.")
    parser.add_argument("--config", help="JSON, TOML, or small YAML router training config.")
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run a quick diagnostic router-training health check.",
    )
    args = parser.parse_args(argv)

    if bool(args.config) == bool(args.health_check):
        parser.error("provide exactly one of --config or --health-check")

    try:
        if args.health_check:
            result = _run_router_training_health_check()
        else:
            result = run_router_training(load_router_training_config(args.config))
    except RouterTrainingError as exc:
        print(json.dumps({"status": "failed", "code": exc.code, "message": exc.message}), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": result.manifest.status,
                "checkpoint_path": str(result.checkpoint_path),
                "target_audit_path": str(result.target_audit_path),
                "objective": result.loss_records[-1].objective if result.loss_records else None,
                "training_sample_count": result.checkpoint.metadata.training_metadata.get(
                    "training_sample_count"
                ),
                "target_record_count": result.checkpoint.metadata.training_metadata.get(
                    "target_record_count"
                ),
                "validation_metrics": result.validation_metrics,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_router_training_health_check() -> RouterTrainingResult:
    output_dir = Path("runs/router_train_health")
    artifact_dir = output_dir / "health_artifacts"
    config = RouterTrainingConfig.from_mapping(
        {
            "model": "fake-qaq-smoke-model",
            "tokenizer": "fake-qaq-smoke-tokenizer",
            "data_source": "fake_smoke",
            "split": "validation",
            "teacher_model": "fake-qaq-smoke-model",
            "student_model": "fake-qaq-smoke-model",
            "student_quantized_path": str(artifact_dir),
            "distillation_loss": ROUTER_COST_CROSS_ENTROPY,
            "precision_candidates": [4, 8],
            "max_bit_width": 8,
            "block_granularity": "mha_ffn",
            "device": "cpu",
            "gpu_ids": [],
            "seed": 0,
            "output_dir": str(output_dir),
            "overwrite": True,
            "prompt_format": "fake_smoke_v1",
            "training_data_limit": 2,
            "checkpoint_interval_steps": 1,
            "diagnostic": True,
            "router": {
                "learning_rate": 0.05,
                "max_steps": 1,
                "temperature": 1.0,
                "target_temperature": 0.25,
                "bit_cost_weight": 0.02,
                "decision_policy": DEFAULT_DECISION_POLICY,
            },
            "logging": {
                "progress_interval_steps": 1,
                "checkpoint_interval_steps": 1,
                "console": False,
                "log_dir": str(output_dir / "logs"),
            },
        }
    )
    adapter = load_model_adapter(config.to_run_config(validate_output=False))
    blocks = discover_mha_ffn_blocks(
        adapter.architecture_metadata,
        supported_bit_widths=config.precision_candidates,
    )
    _write_health_artifacts(config, blocks, artifact_dir)
    return run_router_training(config)


def _write_health_artifacts(
    config: RouterTrainingConfig,
    blocks: tuple[Any, ...],
    artifact_dir: Path,
) -> None:
    for index, block in enumerate(blocks):
        low_bits = (index * 3 + 3) % 16
        values = [[low_bits, 255 - low_bits]]
        artifact = create_bitplane_artifact_from_quantized_values(
            values,
            model_id=config.student_model,
            block_id=block.block_id,
            tensor_name=block.tensor_names[0],
            max_bit_width=config.max_bit_width,
            checkpoint_ref="router-health-check",
            compatibility={"block_granularity": config.block_granularity},
        )
        save_bitplane_artifact(
            artifact,
            artifact_dir / f"{block.block_id}.json",
        )


if __name__ == "__main__":
    raise SystemExit(main())
