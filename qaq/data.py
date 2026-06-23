"""Benchmark example loading for QAQ smoke and adapter tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BenchmarkDataError(ValueError):
    """Raised when benchmark examples cannot be loaded safely."""

    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    """One benchmark or smoke example before tokenization."""

    example_id: str
    text: str
    target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "text": self.text,
            "target": self.target,
            "metadata": dict(self.metadata),
        }


_BUILTIN_DATASETS: dict[str, dict[str, tuple[BenchmarkExample, ...]]] = {
    "fake_smoke": {
        "validation": (
            BenchmarkExample(
                example_id="fake-smoke-0",
                text="A short QAQ smoke prompt.",
                target="short",
            ),
            BenchmarkExample(
                example_id="fake-smoke-1",
                text="A second prompt that should produce a different trace.",
                target="second",
            ),
        )
    },
    "toy_prompts": {
        "smoke": (
            BenchmarkExample(
                example_id="toy-smoke-0",
                text="Toy prompt for importable smoke execution.",
                target="toy",
            ),
        ),
        "validation": (
            BenchmarkExample(
                example_id="toy-validation-0",
                text="Validation prompt for static adapter checks.",
                target="validation",
            ),
        ),
    },
}


def load_benchmark_examples(
    dataset: str,
    *,
    split: str,
    limit: int | None = None,
) -> tuple[BenchmarkExample, ...]:
    """Load benchmark examples from a built-in smoke set or a JSON/JSONL file."""

    if limit is not None and limit <= 0:
        raise BenchmarkDataError("invalid_limit", "limit must be positive when provided")

    if dataset in _BUILTIN_DATASETS:
        examples = _load_builtin_dataset(dataset, split=split)
    else:
        dataset_path = Path(dataset)
        if dataset_path.is_file():
            examples = _load_file_dataset(dataset_path, split=split)
        else:
            raise BenchmarkDataError(
                "dataset_unavailable",
                f"dataset {dataset!r} is not a built-in smoke set or readable file",
            )

    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise BenchmarkDataError(
            "dataset_split_empty",
            f"dataset {dataset!r} split {split!r} produced no examples",
        )
    return examples


def _load_builtin_dataset(dataset: str, *, split: str) -> tuple[BenchmarkExample, ...]:
    splits = _BUILTIN_DATASETS[dataset]
    if split not in splits:
        raise BenchmarkDataError(
            "dataset_split_unavailable",
            f"dataset {dataset!r} does not provide split {split!r}",
        )
    return splits[split]


def _load_file_dataset(path: Path, *, split: str) -> tuple[BenchmarkExample, ...]:
    try:
        if path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif path.suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            rows = raw["examples"] if isinstance(raw, dict) and "examples" in raw else raw
        else:
            raise BenchmarkDataError(
                "unsupported_dataset_format",
                "benchmark fixtures must be .json or .jsonl",
            )
    except BenchmarkDataError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkDataError("dataset_read_failed", str(exc)) from exc

    if not isinstance(rows, list):
        raise BenchmarkDataError(
            "invalid_dataset",
            "benchmark fixture must contain a list of examples",
        )

    examples: list[BenchmarkExample] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BenchmarkDataError(
                "invalid_dataset",
                f"example {index} must be an object",
            )
        row_split = row.get("split", split)
        if row_split != split:
            continue
        examples.append(_example_from_row(row, fallback_id=f"{path.stem}-{index}"))
    return tuple(examples)


def _example_from_row(row: dict[str, Any], *, fallback_id: str) -> BenchmarkExample:
    text = row.get("text")
    if not isinstance(text, str) or not text:
        raise BenchmarkDataError(
            "invalid_dataset",
            "each example requires non-empty text",
        )
    example_id = row.get("id", row.get("example_id", fallback_id))
    if not isinstance(example_id, str) or not example_id:
        raise BenchmarkDataError(
            "invalid_dataset",
            "example id must be a non-empty string when provided",
        )
    target = row.get("target")
    if target is not None and not isinstance(target, str):
        raise BenchmarkDataError(
            "invalid_dataset",
            "example target must be a string when provided",
        )
    metadata = {
        key: value
        for key, value in row.items()
        if key not in {"id", "example_id", "text", "target", "split"}
    }
    return BenchmarkExample(
        example_id=example_id,
        text=text,
        target=target,
        metadata=metadata,
    )
