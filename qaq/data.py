"""Benchmark example loading for QAQ smoke, file, and real benchmark tests."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
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

_REAL_BENCHMARK_DATASETS: dict[str, dict[str, str | None]] = {
    "hellaswag": {"hf_path": "hellaswag", "hf_name": None},
    "piqa": {"hf_path": "piqa", "hf_name": None},
    "arc_easy": {"hf_path": "ai2_arc", "hf_name": "ARC-Easy"},
    "arc_challenge": {"hf_path": "ai2_arc", "hf_name": "ARC-Challenge"},
    "winogrande": {"hf_path": "winogrande", "hf_name": "winogrande_xl"},
    "wikitext2": {"hf_path": "wikitext", "hf_name": "wikitext-2-raw-v1"},
}
_BENCHMARK_ROOT_ENV_VARS = ("QAQ_BENCHMARK_DATA_ROOT", "QAQ_BENCHMARK_ROOT")
_DISABLE_HF_DATASETS_ENV = "QAQ_DISABLE_HF_DATASETS"


def load_benchmark_examples(
    dataset: str,
    *,
    split: str,
    limit: int | None = None,
) -> tuple[BenchmarkExample, ...]:
    """Load benchmark examples from smoke data, files, or named real benchmarks."""

    if limit is not None and limit <= 0:
        raise BenchmarkDataError("invalid_limit", "limit must be positive when provided")

    if dataset in _BUILTIN_DATASETS:
        examples = _load_builtin_dataset(dataset, split=split)
    else:
        dataset_path = Path(dataset)
        if dataset_path.is_file():
            examples = _load_file_dataset(dataset_path, split=split)
        elif dataset in _REAL_BENCHMARK_DATASETS:
            examples = _load_real_benchmark_dataset(dataset, split=split)
        else:
            raise BenchmarkDataError(
                "dataset_unavailable",
                f"dataset {dataset!r} is not a built-in smoke set, supported real benchmark name, or readable file",
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


def _load_real_benchmark_dataset(dataset: str, *, split: str) -> tuple[BenchmarkExample, ...]:
    searched_paths = tuple(_candidate_real_benchmark_paths(dataset, split=split))
    for candidate in searched_paths:
        if candidate.is_file():
            return _load_file_dataset(candidate, split=split, dataset_name=dataset)

    try:
        return _load_cached_hf_dataset(dataset, split=split)
    except BenchmarkDataError as exc:
        searched = ", ".join(str(path) for path in searched_paths) or "no benchmark root configured"
        raise BenchmarkDataError(
            "benchmark_dataset_unavailable",
            (
                f"real benchmark {dataset!r} split {split!r} is supported by name, "
                "but no local data was found. Set QAQ_BENCHMARK_DATA_ROOT to files "
                f"such as <root>/{dataset}/{split}.jsonl or provide a cached Hugging Face datasets copy. "
                f"searched: {searched}. cached datasets result: {exc}"
            ),
        ) from exc


def _candidate_real_benchmark_paths(dataset: str, *, split: str) -> tuple[Path, ...]:
    roots: list[Path] = []
    for env_name in _BENCHMARK_ROOT_ENV_VARS:
        env_value = os.environ.get(env_name)
        if env_value:
            roots.append(Path(env_value))
    roots.extend(path for path in (Path("benchmarks/data"), Path("data/benchmarks")) if path.exists())

    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            (
                root / dataset / f"{split}.jsonl",
                root / dataset / f"{split}.json",
                root / f"{dataset}_{split}.jsonl",
                root / f"{dataset}_{split}.json",
                root / f"{dataset}-{split}.jsonl",
                root / f"{dataset}-{split}.json",
                root / f"{dataset}.jsonl",
                root / f"{dataset}.json",
            )
        )
    return tuple(dict.fromkeys(candidates))


def _load_cached_hf_dataset(dataset: str, *, split: str) -> tuple[BenchmarkExample, ...]:
    if os.environ.get(_DISABLE_HF_DATASETS_ENV) == "1":
        raise BenchmarkDataError(
            "cached_benchmark_disabled",
            f"cached Hugging Face datasets loading is disabled by {_DISABLE_HF_DATASETS_ENV}=1",
        )
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise BenchmarkDataError(
            "datasets_dependency_missing",
            "install or provide Hugging Face datasets for named real benchmarks",
        ) from exc

    spec = _REAL_BENCHMARK_DATASETS[dataset]
    try:
        rows = load_dataset(
            str(spec["hf_path"]),
            name=spec["hf_name"],
            split=split,
            download_config=DownloadConfig(local_files_only=True),
        )
    except Exception as exc:
        raise BenchmarkDataError(
            "cached_benchmark_unavailable",
            f"failed to load cached Hugging Face dataset {dataset!r}: {exc}",
        ) from exc
    return _examples_from_rows(rows, split=split, dataset_name=dataset, source="hf_datasets_cache")


def _load_file_dataset(
    path: Path,
    *,
    split: str,
    dataset_name: str | None = None,
) -> tuple[BenchmarkExample, ...]:
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

    return _examples_from_rows(
        rows,
        split=split,
        dataset_name=dataset_name,
        source=str(path),
    )


def _examples_from_rows(
    rows: Iterable[Any],
    *,
    split: str,
    dataset_name: str | None,
    source: str,
) -> tuple[BenchmarkExample, ...]:
    examples: list[BenchmarkExample] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise BenchmarkDataError(
                "invalid_dataset",
                f"example {index} must be an object",
            )
        row_dict = dict(row)
        row_split = row_dict.get("split", split)
        if row_split != split:
            continue
        examples.append(
            _example_from_row(
                row_dict,
                fallback_id=f"{dataset_name or 'dataset'}-{index}",
                dataset_name=dataset_name,
                source=source,
            )
        )
    return tuple(examples)


def _example_from_row(
    row: dict[str, Any],
    *,
    fallback_id: str,
    dataset_name: str | None = None,
    source: str | None = None,
) -> BenchmarkExample:
    if dataset_name in _REAL_BENCHMARK_DATASETS and "text" not in row:
        return _real_benchmark_example_from_row(
            dataset_name,
            row,
            fallback_id=fallback_id,
            source=source,
        )

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
        target = str(target)
    metadata = {
        key: value
        for key, value in row.items()
        if key not in {"id", "example_id", "text", "target", "split"}
    }
    if source is not None:
        metadata.setdefault("source", source)
    if dataset_name is not None:
        metadata.setdefault("benchmark_name", dataset_name)
    return BenchmarkExample(
        example_id=example_id,
        text=text,
        target=target,
        metadata=metadata,
    )


def _real_benchmark_example_from_row(
    dataset_name: str,
    row: dict[str, Any],
    *,
    fallback_id: str,
    source: str | None,
) -> BenchmarkExample:
    if dataset_name == "hellaswag":
        return _hellaswag_example(row, fallback_id=fallback_id, source=source)
    if dataset_name == "piqa":
        return _piqa_example(row, fallback_id=fallback_id, source=source)
    if dataset_name in {"arc_easy", "arc_challenge"}:
        return _arc_example(dataset_name, row, fallback_id=fallback_id, source=source)
    if dataset_name == "winogrande":
        return _winogrande_example(row, fallback_id=fallback_id, source=source)
    if dataset_name == "wikitext2":
        return _wikitext2_example(row, fallback_id=fallback_id, source=source)
    raise BenchmarkDataError(
        "unsupported_real_benchmark",
        f"benchmark {dataset_name!r} has no row adapter",
    )


def _hellaswag_example(row: dict[str, Any], *, fallback_id: str, source: str | None) -> BenchmarkExample:
    ctx = _required_string(row, "ctx", fallback=("context", "query"))
    endings = _string_list(row.get("endings"))
    if not endings:
        raise BenchmarkDataError("invalid_dataset", "HellaSwag rows require endings")
    label_index = _label_index(row.get("label"))
    target = endings[label_index] if label_index is not None and label_index < len(endings) else None
    return _real_example(
        row,
        fallback_id=fallback_id,
        text=_multiple_choice_prompt(ctx, endings),
        target=target,
        benchmark_name="hellaswag",
        source=source,
    )


def _piqa_example(row: dict[str, Any], *, fallback_id: str, source: str | None) -> BenchmarkExample:
    goal = _required_string(row, "goal", fallback=("question", "text"))
    choices = (_required_string(row, "sol1"), _required_string(row, "sol2"))
    label_index = _label_index(row.get("label"))
    target = choices[label_index] if label_index is not None and label_index < len(choices) else None
    return _real_example(
        row,
        fallback_id=fallback_id,
        text=_multiple_choice_prompt(goal, choices),
        target=target,
        benchmark_name="piqa",
        source=source,
    )


def _arc_example(
    dataset_name: str,
    row: dict[str, Any],
    *,
    fallback_id: str,
    source: str | None,
) -> BenchmarkExample:
    question = _required_string(row, "question", fallback=("text",))
    choices_value = row.get("choices")
    if isinstance(choices_value, Mapping):
        choice_texts = _string_list(choices_value.get("text"))
        choice_labels = _string_list(choices_value.get("label"))
    else:
        choice_texts = _string_list(choices_value)
        choice_labels = tuple(chr(ord("A") + index) for index in range(len(choice_texts)))
    if not choice_texts:
        raise BenchmarkDataError("invalid_dataset", "ARC rows require choices")
    if len(choice_labels) != len(choice_texts):
        choice_labels = tuple(chr(ord("A") + index) for index in range(len(choice_texts)))
    answer = row.get("answerKey", row.get("answer", row.get("label")))
    target = _target_from_labeled_choices(answer, labels=choice_labels, choices=choice_texts)
    return _real_example(
        row,
        fallback_id=fallback_id,
        text=_multiple_choice_prompt(question, choice_texts, labels=choice_labels),
        target=target,
        benchmark_name=dataset_name,
        source=source,
    )


def _winogrande_example(row: dict[str, Any], *, fallback_id: str, source: str | None) -> BenchmarkExample:
    sentence = _required_string(row, "sentence", fallback=("text",))
    choices = (_required_string(row, "option1"), _required_string(row, "option2"))
    label = _label_index(row.get("answer"))
    target = choices[label - 1] if label in {1, 2} else None
    prompt = f"{sentence}\nChoices:\n1. {choices[0]}\n2. {choices[1]}"
    return _real_example(
        row,
        fallback_id=fallback_id,
        text=prompt,
        target=target,
        benchmark_name="winogrande",
        source=source,
    )


def _wikitext2_example(row: dict[str, Any], *, fallback_id: str, source: str | None) -> BenchmarkExample:
    text = _required_string(row, "text")
    return _real_example(
        row,
        fallback_id=fallback_id,
        text=text,
        target=text,
        benchmark_name="wikitext2",
        source=source,
    )


def _real_example(
    row: dict[str, Any],
    *,
    fallback_id: str,
    text: str,
    target: str | None,
    benchmark_name: str,
    source: str | None,
) -> BenchmarkExample:
    example_id = row.get("id", row.get("example_id", row.get("ind", fallback_id)))
    if not isinstance(example_id, str):
        example_id = str(example_id)
    metadata = {
        key: value
        for key, value in row.items()
        if key not in {"id", "example_id", "text", "target", "split"}
    }
    metadata["benchmark_name"] = benchmark_name
    metadata["real_benchmark"] = True
    if source is not None:
        metadata["source"] = source
    return BenchmarkExample(
        example_id=example_id,
        text=text,
        target=target,
        metadata=metadata,
    )


def _multiple_choice_prompt(
    stem: str,
    choices: Iterable[str],
    *,
    labels: Iterable[str] | None = None,
) -> str:
    choices_tuple = tuple(choices)
    labels_tuple = tuple(labels) if labels is not None else tuple(
        chr(ord("A") + index) for index in range(len(choices_tuple))
    )
    rendered = "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(labels_tuple, choices_tuple, strict=True)
    )
    return f"{stem}\nChoices:\n{rendered}\nAnswer:"


def _required_string(
    row: Mapping[str, Any],
    field: str,
    *,
    fallback: tuple[str, ...] = (),
) -> str:
    for name in (field, *fallback):
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value
    raise BenchmarkDataError("invalid_dataset", f"real benchmark row requires {field!r}")


def _string_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    return ()


def _label_index(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _target_from_labeled_choices(
    value: Any,
    *,
    labels: tuple[str, ...],
    choices: tuple[str, ...],
) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        for label, choice in zip(labels, choices, strict=True):
            if stripped == label:
                return choice
        if stripped.isdigit():
            index = int(stripped)
            if 0 <= index < len(choices):
                return choices[index]
    index = _label_index(value)
    if index is not None and 0 <= index < len(choices):
        return choices[index]
    return None
