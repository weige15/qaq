"""Run configuration schema and validation for QAQ experiments."""

from __future__ import annotations

import json
import sys
import tomllib
from argparse import ArgumentParser
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qaq.errors import ConfigValidationError


VALID_MODES = frozenset(
    {
        "fp16",
        "static_8bit",
        "static_4bit",
        "fixed_mixed",
        "qaq_on_demand_off",
        "qaq_on_demand_on",
    }
)
QAQ_MODES = frozenset({"qaq_on_demand_off", "qaq_on_demand_on"})
VALID_BLOCK_GRANULARITIES = frozenset({"mha_ffn", "whole_layer"})


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Logging settings consumed by later progress and event modules."""

    progress_interval_steps: int = 10
    checkpoint_interval_steps: int | None = None
    console: bool = True
    log_dir: Path | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "LoggingConfig":
        data = dict(value or {})
        progress_interval_steps = _coerce_int(
            data.get("progress_interval_steps", 10),
            field="logging.progress_interval_steps",
        )
        if progress_interval_steps <= 0:
            raise ConfigValidationError(
                "invalid_logging",
                "progress interval must be positive",
                "logging.progress_interval_steps",
            )

        checkpoint_raw = data.get("checkpoint_interval_steps")
        checkpoint_interval_steps = None
        if checkpoint_raw is not None:
            checkpoint_interval_steps = _coerce_int(
                checkpoint_raw,
                field="logging.checkpoint_interval_steps",
            )
            if checkpoint_interval_steps <= 0:
                raise ConfigValidationError(
                    "invalid_logging",
                    "checkpoint interval must be positive when provided",
                    "logging.checkpoint_interval_steps",
                )

        console = data.get("console", True)
        if not isinstance(console, bool):
            raise ConfigValidationError(
                "invalid_logging",
                "console must be a boolean",
                "logging.console",
            )

        log_dir_raw = data.get("log_dir")
        log_dir = Path(log_dir_raw) if log_dir_raw else None

        return cls(
            progress_interval_steps=progress_interval_steps,
            checkpoint_interval_steps=checkpoint_interval_steps,
            console=console,
            log_dir=log_dir,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "progress_interval_steps": self.progress_interval_steps,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "console": self.console,
            "log_dir": str(self.log_dir) if self.log_dir else None,
        }


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Validated immutable settings for one QAQ run mode."""

    model: str
    tokenizer: str
    dataset: str
    split: str
    mode: str
    precision_candidates: tuple[int, ...]
    max_bit_width: int
    block_granularity: str
    gpu_ids: tuple[int, ...]
    seed: int
    output_dir: Path
    overwrite: bool = False
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    router_checkpoint: Path | None = None
    router_diagnostic: bool = False
    use_model_tokenizer: bool = False
    device: str = "cpu"
    fixed_precision_by_block: dict[str, int] | None = None
    prompt_format: str | None = None
    metric: str | None = None
    notes: str | None = None
    max_examples: int | None = None
    eval_batch_size: int = 1
    collect_hidden_states: bool = False
    store_full_logits: bool = False
    hf_device_map: str | None = None
    hf_max_memory_per_gpu: str | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        available_gpu_count: int | None = None,
        validate_output: bool = True,
    ) -> "RunConfig":
        required = (
            "model",
            "dataset",
            "split",
            "mode",
            "precision_candidates",
            "max_bit_width",
            "block_granularity",
            "gpu_ids",
            "seed",
            "output_dir",
        )
        data = dict(value)
        for field_name in required:
            if field_name not in data:
                raise ConfigValidationError(
                    "missing_required_field",
                    "field is required",
                    field_name,
            )

        model = _coerce_non_empty_string(data["model"], "model")
        use_model_tokenizer = data.get("use_model_tokenizer", False)
        if not isinstance(use_model_tokenizer, bool):
            raise ConfigValidationError(
                "invalid_use_model_tokenizer",
                "use_model_tokenizer must be a boolean",
                "use_model_tokenizer",
            )
        if "tokenizer" in data:
            tokenizer = _coerce_non_empty_string(data["tokenizer"], "tokenizer")
        elif use_model_tokenizer:
            tokenizer = model
        else:
            raise ConfigValidationError(
                "missing_required_field",
                "field is required unless use_model_tokenizer=true",
                "tokenizer",
            )

        dataset = _coerce_non_empty_string(data["dataset"], "dataset")
        split = _coerce_non_empty_string(data["split"], "split")

        mode = _coerce_non_empty_string(data["mode"], "mode")
        if mode not in VALID_MODES:
            raise ConfigValidationError(
                "invalid_mode",
                f"expected one of {sorted(VALID_MODES)}",
                "mode",
            )

        max_bit_width = _coerce_int(data["max_bit_width"], field="max_bit_width")
        if max_bit_width <= 0:
            raise ConfigValidationError(
                "invalid_max_bit_width",
                "max bit width must be positive",
                "max_bit_width",
            )

        precision_candidates = _coerce_precision_candidates(
            data["precision_candidates"],
            max_bit_width=max_bit_width,
        )
        _validate_mode_precision(mode, precision_candidates)

        block_granularity = _coerce_non_empty_string(
            data["block_granularity"],
            "block_granularity",
        )
        if block_granularity not in VALID_BLOCK_GRANULARITIES:
            raise ConfigValidationError(
                "invalid_block_granularity",
                f"expected one of {sorted(VALID_BLOCK_GRANULARITIES)}",
                "block_granularity",
            )

        gpu_ids = _coerce_gpu_ids(
            data["gpu_ids"],
            available_gpu_count=available_gpu_count,
        )
        seed = _coerce_int(data["seed"], field="seed")
        output_dir = Path(_coerce_non_empty_string(data["output_dir"], "output_dir"))

        overwrite = data.get("overwrite", False)
        if not isinstance(overwrite, bool):
            raise ConfigValidationError(
                "invalid_output_policy",
                "overwrite must be a boolean",
                "overwrite",
            )
        if validate_output and output_dir.exists() and not overwrite:
            raise ConfigValidationError(
                "unsafe_output_reuse",
                "output directory already exists; set overwrite=true to reuse it",
                "output_dir",
            )

        logging = LoggingConfig.from_mapping(data.get("logging"))

        router_checkpoint_raw = data.get("router_checkpoint")
        router_checkpoint = (
            Path(_coerce_non_empty_string(router_checkpoint_raw, "router_checkpoint"))
            if router_checkpoint_raw is not None
            else None
        )

        router_diagnostic = data.get(
            "router_diagnostic",
            data.get("diagnostic_router", False),
        )
        if not isinstance(router_diagnostic, bool):
            raise ConfigValidationError(
                "invalid_router_diagnostic",
                "router_diagnostic must be a boolean",
                "router_diagnostic",
            )
        if mode in QAQ_MODES and router_checkpoint is None and not router_diagnostic:
            raise ConfigValidationError(
                "missing_router_checkpoint",
                "QAQ evaluation requires router_checkpoint unless router_diagnostic=true",
                "router_checkpoint",
            )

        fixed_precision_by_block = _coerce_fixed_precision_by_block(
            data.get("fixed_precision_by_block"),
            mode=mode,
            precision_candidates=precision_candidates,
        )

        device = data.get("device")
        if device is None:
            device = "cuda" if gpu_ids else "cpu"
        device = _coerce_non_empty_string(device, "device")
        if device not in {"cpu", "cuda"}:
            raise ConfigValidationError(
                "invalid_device",
                "expected 'cpu' or 'cuda'",
                "device",
            )
        if device == "cpu" and gpu_ids:
            raise ConfigValidationError(
                "invalid_device",
                "cpu device cannot select gpu_ids",
                "device",
            )
        if device == "cuda" and not gpu_ids:
            raise ConfigValidationError(
                "invalid_device",
                "cuda device requires at least one gpu_id",
                "gpu_ids",
            )

        prompt_format = _optional_string(data.get("prompt_format"), "prompt_format")
        metric = _optional_string(data.get("metric"), "metric")
        notes = _optional_string(data.get("notes"), "notes")
        max_examples = _optional_positive_int(
            data.get("max_examples"),
            field="max_examples",
        )
        eval_batch_size = _coerce_int(
            data.get("eval_batch_size", 1),
            field="eval_batch_size",
        )
        if eval_batch_size <= 0:
            raise ConfigValidationError(
                "invalid_eval_batch_size",
                "eval_batch_size must be positive",
                "eval_batch_size",
            )
        collect_hidden_states = _coerce_bool(
            data.get("collect_hidden_states", False),
            field="collect_hidden_states",
        )
        store_full_logits = _coerce_bool(
            data.get("store_full_logits", False),
            field="store_full_logits",
        )
        hf_device_map = _optional_string(data.get("hf_device_map"), "hf_device_map")
        if hf_device_map not in {None, "single", "auto"}:
            raise ConfigValidationError(
                "invalid_hf_device_map",
                "hf_device_map must be null, 'single', or 'auto'",
                "hf_device_map",
            )
        hf_max_memory_per_gpu = _optional_string(
            data.get("hf_max_memory_per_gpu"),
            "hf_max_memory_per_gpu",
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            split=split,
            mode=mode,
            precision_candidates=precision_candidates,
            max_bit_width=max_bit_width,
            block_granularity=block_granularity,
            gpu_ids=gpu_ids,
            seed=seed,
            output_dir=output_dir,
            overwrite=overwrite,
            logging=logging,
            router_checkpoint=router_checkpoint,
            router_diagnostic=router_diagnostic,
            use_model_tokenizer=use_model_tokenizer,
            device=device,
            fixed_precision_by_block=fixed_precision_by_block,
            prompt_format=prompt_format,
            metric=metric,
            notes=notes,
            max_examples=max_examples,
            eval_batch_size=eval_batch_size,
            collect_hidden_states=collect_hidden_states,
            store_full_logits=store_full_logits,
            hf_device_map=hf_device_map,
            hf_max_memory_per_gpu=hf_max_memory_per_gpu,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "dataset": self.dataset,
            "split": self.split,
            "mode": self.mode,
            "precision_candidates": list(self.precision_candidates),
            "max_bit_width": self.max_bit_width,
            "block_granularity": self.block_granularity,
            "gpu_ids": list(self.gpu_ids),
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "overwrite": self.overwrite,
            "logging": self.logging.as_dict(),
            "router_checkpoint": (
                str(self.router_checkpoint) if self.router_checkpoint else None
            ),
            "router_diagnostic": self.router_diagnostic,
            "use_model_tokenizer": self.use_model_tokenizer,
            "device": self.device,
            "fixed_precision_by_block": self.fixed_precision_by_block,
            "prompt_format": self.prompt_format,
            "metric": self.metric,
            "notes": self.notes,
            "max_examples": self.max_examples,
            "eval_batch_size": self.eval_batch_size,
            "collect_hidden_states": self.collect_hidden_states,
            "store_full_logits": self.store_full_logits,
            "hf_device_map": self.hf_device_map,
            "hf_max_memory_per_gpu": self.hf_max_memory_per_gpu,
        }


def load_config_file(
    path: str | Path,
    *,
    available_gpu_count: int | None = None,
    validate_output: bool = True,
) -> RunConfig:
    """Load a JSON or TOML run config file and return a validated config."""

    config_path = Path(path)
    try:
        if config_path.suffix == ".json":
            with config_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        elif config_path.suffix == ".toml":
            with config_path.open("rb") as handle:
                raw = tomllib.load(handle)
        else:
            raise ConfigValidationError(
                "unsupported_config_format",
                "supported config formats are .json and .toml",
                str(config_path),
            )
    except ConfigValidationError:
        raise
    except OSError as exc:
        raise ConfigValidationError(
            "config_read_failed",
            str(exc),
            str(config_path),
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(
            "config_parse_failed",
            str(exc),
            str(config_path),
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigValidationError(
            "config_parse_failed",
            str(exc),
            str(config_path),
        ) from exc

    if not isinstance(raw, Mapping):
        raise ConfigValidationError(
            "invalid_config",
            "top-level config must be an object",
            str(config_path),
        )

    return RunConfig.from_mapping(
        raw,
        available_gpu_count=available_gpu_count,
        validate_output=validate_output,
    )


def _coerce_precision_candidates(
    value: Any,
    *,
    max_bit_width: int,
) -> tuple[int, ...]:
    if not isinstance(value, list | tuple):
        raise ConfigValidationError(
            "invalid_precision_candidates",
            "precision_candidates must be a list of integers",
            "precision_candidates",
        )

    candidates = tuple(_coerce_int(item, field="precision_candidates") for item in value)
    if not candidates:
        raise ConfigValidationError(
            "invalid_precision_candidates",
            "at least one precision candidate is required",
            "precision_candidates",
        )
    if len(set(candidates)) != len(candidates):
        raise ConfigValidationError(
            "invalid_precision_candidates",
            "precision candidates must be unique",
            "precision_candidates",
        )
    for bit_width in candidates:
        if bit_width <= 0:
            raise ConfigValidationError(
                "invalid_precision_candidates",
                "precision candidates must be positive",
                "precision_candidates",
            )
        if bit_width > max_bit_width:
            raise ConfigValidationError(
                "invalid_precision_candidates",
                "precision candidate exceeds max_bit_width",
                "precision_candidates",
            )
    return tuple(sorted(candidates))


def _validate_mode_precision(mode: str, precision_candidates: tuple[int, ...]) -> None:
    if mode == "static_8bit" and 8 not in precision_candidates:
        raise ConfigValidationError(
            "missing_mode_precision",
            "static_8bit mode requires 8 in precision_candidates",
            "precision_candidates",
        )
    if mode == "static_4bit" and 4 not in precision_candidates:
        raise ConfigValidationError(
            "missing_mode_precision",
            "static_4bit mode requires 4 in precision_candidates",
            "precision_candidates",
        )
    if mode in QAQ_MODES and len(precision_candidates) < 2:
        raise ConfigValidationError(
            "missing_mode_precision",
            "QAQ modes require at least two precision candidates",
            "precision_candidates",
        )


def _coerce_gpu_ids(
    value: Any,
    *,
    available_gpu_count: int | None,
) -> tuple[int, ...]:
    if not isinstance(value, list | tuple):
        raise ConfigValidationError(
            "invalid_gpu_ids",
            "gpu_ids must be a list of non-negative integers",
            "gpu_ids",
        )
    gpu_ids = tuple(_coerce_int(item, field="gpu_ids") for item in value)
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ConfigValidationError(
            "invalid_gpu_ids",
            "gpu_ids must be unique",
            "gpu_ids",
        )
    for gpu_id in gpu_ids:
        if gpu_id < 0:
            raise ConfigValidationError(
                "invalid_gpu_ids",
                "gpu_ids must be non-negative",
                "gpu_ids",
            )
        if available_gpu_count is not None and gpu_id >= available_gpu_count:
            raise ConfigValidationError(
                "invalid_gpu_ids",
                f"gpu_id {gpu_id} is outside available GPU count {available_gpu_count}",
                "gpu_ids",
            )
    return gpu_ids


def _coerce_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(
            "invalid_string",
            "expected a non-empty string",
            field,
        )
    return value


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _coerce_non_empty_string(value, field)


def _coerce_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(
            "invalid_integer",
            "expected an integer",
            field,
        )
    return value


def _optional_positive_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    result = _coerce_int(value, field=field)
    if result <= 0:
        raise ConfigValidationError(
            "invalid_integer",
            "expected a positive integer",
            field,
        )
    return result


def _coerce_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigValidationError(
            "invalid_boolean",
            "expected a boolean",
            field,
        )
    return value


def _coerce_fixed_precision_by_block(
    value: Any,
    *,
    mode: str,
    precision_candidates: tuple[int, ...],
) -> dict[str, int] | None:
    if value is None:
        if mode == "fixed_mixed":
            raise ConfigValidationError(
                "missing_fixed_profile",
                "fixed_mixed mode requires fixed_precision_by_block",
                "fixed_precision_by_block",
            )
        return None

    if not isinstance(value, Mapping):
        raise ConfigValidationError(
            "invalid_fixed_profile",
            "fixed_precision_by_block must be an object",
            "fixed_precision_by_block",
        )

    profile: dict[str, int] = {}
    for block_id, bit_width_raw in value.items():
        if not isinstance(block_id, str) or not block_id:
            raise ConfigValidationError(
                "invalid_fixed_profile",
                "fixed_precision_by_block keys must be block IDs",
                "fixed_precision_by_block",
            )
        bit_width = _coerce_int(
            bit_width_raw,
            field=f"fixed_precision_by_block.{block_id}",
        )
        if bit_width not in precision_candidates:
            raise ConfigValidationError(
                "invalid_fixed_profile",
                "fixed profile bit-widths must be in precision_candidates",
                "fixed_precision_by_block",
            )
        profile[block_id] = bit_width

    if mode == "fixed_mixed" and not profile:
        raise ConfigValidationError(
            "missing_fixed_profile",
            "fixed_mixed mode requires at least one block precision",
            "fixed_precision_by_block",
        )

    return profile


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Validate a QAQ run configuration.")
    parser.add_argument("config", help="Path to a JSON or TOML config file.")
    parser.add_argument(
        "--available-gpu-count",
        type=int,
        default=None,
        help="Optional GPU count used to validate selected gpu_ids.",
    )
    parser.add_argument(
        "--skip-output-dir-check",
        action="store_true",
        help="Skip existing output directory reuse validation.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the resolved config JSON after validation.",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config_file(
            args.config,
            available_gpu_count=args.available_gpu_count,
            validate_output=not args.skip_output_dir_check,
        )
    except ConfigValidationError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code

    if args.print_json:
        print(json.dumps(config.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
