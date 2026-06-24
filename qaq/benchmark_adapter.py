"""Benchmark prompt formatting and tokenization contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from qaq.config import RunConfig
from qaq.data import BenchmarkDataError, BenchmarkExample


class TokenizerLike(Protocol):
    tokenizer_id: str
    pad_token_id: int
    model_max_length: int

    def encode(self, text: str) -> tuple[int, ...]:
        """Return token IDs for text without padding."""


@dataclass(frozen=True, slots=True)
class BenchmarkBatchMetadata:
    dataset: str
    split: str
    prompt_format: str
    tokenizer: str
    batch_size: int
    max_length: int
    context_length_policy: str
    truncated_examples: tuple[str, ...]
    example_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "prompt_format": self.prompt_format,
            "tokenizer": self.tokenizer,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "context_length_policy": self.context_length_policy,
            "truncated_examples": list(self.truncated_examples),
            "example_ids": list(self.example_ids),
        }


@dataclass(frozen=True, slots=True)
class TokenizedBenchmarkBatch:
    """Padded benchmark inputs plus metadata needed for comparison checks."""

    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    targets: tuple[str | None, ...]
    examples: tuple[BenchmarkExample, ...]
    metadata: BenchmarkBatchMetadata

    @property
    def example_ids(self) -> tuple[str, ...]:
        return self.metadata.example_ids

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_ids": [list(row) for row in self.input_ids],
            "attention_mask": [list(row) for row in self.attention_mask],
            "targets": list(self.targets),
            "metadata": self.metadata.as_dict(),
        }


def build_tokenized_batch(
    config: RunConfig,
    examples: tuple[BenchmarkExample, ...],
    tokenizer: TokenizerLike,
    *,
    max_length: int | None = None,
    context_length_policy: str = "truncate",
) -> TokenizedBenchmarkBatch:
    """Format and tokenize examples with recorded comparability metadata."""

    if not examples:
        raise BenchmarkDataError("empty_dataset", "at least one example is required")
    if context_length_policy not in {"truncate", "reject"}:
        raise BenchmarkDataError(
            "invalid_context_policy",
            "context_length_policy must be 'truncate' or 'reject'",
        )

    resolved_max_length = tokenizer.model_max_length if max_length is None else max_length
    if resolved_max_length <= 0:
        raise BenchmarkDataError("invalid_max_length", "max_length must be positive")

    prompt_format = config.prompt_format or "plain"
    encoded_rows: list[tuple[int, ...]] = []
    truncated_examples: list[str] = []

    for example in examples:
        prompt = format_prompt(example, prompt_format=prompt_format)
        token_ids = tokenizer.encode(prompt)
        if len(token_ids) > resolved_max_length:
            if context_length_policy == "reject":
                raise BenchmarkDataError(
                    "context_length_exceeded",
                    f"{example.example_id} exceeds max_length={resolved_max_length}",
                )
            token_ids = token_ids[:resolved_max_length]
            truncated_examples.append(example.example_id)
        encoded_rows.append(token_ids)

    padded_length = max(len(row) for row in encoded_rows)
    input_ids = tuple(
        row + (tokenizer.pad_token_id,) * (padded_length - len(row))
        for row in encoded_rows
    )
    attention_mask = tuple(
        (1,) * len(row) + (0,) * (padded_length - len(row))
        for row in encoded_rows
    )
    metadata = BenchmarkBatchMetadata(
        dataset=config.dataset,
        split=config.split,
        prompt_format=prompt_format,
        tokenizer=config.tokenizer,
        batch_size=len(examples),
        max_length=resolved_max_length,
        context_length_policy=context_length_policy,
        truncated_examples=tuple(truncated_examples),
        example_ids=tuple(example.example_id for example in examples),
    )
    return TokenizedBenchmarkBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        targets=tuple(example.target for example in examples),
        examples=examples,
        metadata=metadata,
    )


def format_prompt(example: BenchmarkExample, *, prompt_format: str) -> str:
    """Apply explicit prompt formats for smoke and benchmark adapter tests."""

    if prompt_format in {"plain", "fake_smoke_v1", "paper_aligned_default"}:
        return example.text
    if prompt_format.startswith("lm_eval:"):
        return example.text
    if prompt_format == "question_answer_v1":
        return f"Question: {example.text}\nAnswer:"
    raise BenchmarkDataError(
        "unsupported_prompt_format",
        f"prompt format {prompt_format!r} is not supported by the local adapter",
    )
