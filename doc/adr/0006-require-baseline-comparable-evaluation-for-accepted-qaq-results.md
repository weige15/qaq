# ADR 0006: Require Baseline-Comparable Evaluation for Accepted QAQ Results

## Status
Accepted

## Context

The QAQ paper reports comparisons against full FP16, static 8-bit, static 4-bit, QAQ on-demand off, and QAQ on-demand on across accuracy, perplexity, latency, and GPU memory. The local requirements state that QAQ results are not accepted if static baselines cannot run, if metrics omit latency or memory, or if different checkpoints, datasets, prompts, tokenizers, or metric implementations are used across modes.

The first milestone targets LLaMA-3.1-8B on a configurable subset of 8 RTX 3090 GPUs. Full paper-aligned reproduction remains Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B on HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande, WikiText-2, and PTB.

## Options Considered

- Baseline-comparable evaluation gate. Considered because QAQ is defined by its accuracy-memory-latency trade-off against static quantization. Pros: prevents unsupported claims, keeps metrics comparable, and matches paper-style reporting. Cons: requires more runs and may block acceptance when model, dataset, or hardware access is incomplete. Accepted.
- QAQ-only evaluation. Considered because it is cheaper to run. Pros: faster iteration. Cons: cannot prove accuracy retention, memory reduction, or latency overhead relative to baselines. Rejected for accepted results.
- Compare only against FP16. Considered because FP16 is the teacher/reference path. Pros: useful quality reference. Cons: misses the paper's claim that QAQ matches 8-bit and improves memory behavior relative to static quantization. Rejected.
- Accept smoke tests as final evidence. Considered because the repository starts without implementation. Pros: useful during development. Cons: fake-model or tiny smoke tests cannot support first-milestone or paper-aligned claims. Rejected for final acceptance.

## Decision

Accepted QAQ results must include the required comparison modes where hardware permits: `fp16`, `static_8bit`, `static_4bit`, `qaq_on_demand_off`, and `qaq_on_demand_on`. Fixed mixed precision is a diagnostic validation mode and does not replace paper-required modes.

All accepted comparisons must share the same model checkpoint, tokenizer, dataset split, prompt format, precision candidates, seed policy, selected metric implementation, and relevant runtime settings. Results must include score or perplexity, end-to-end latency, peak GPU memory, routing summary, loader summary for on-demand mode, logs, selected GPU IDs, and completion status.

## Consequences

This decision makes benchmark claims auditable and prevents "runs successfully" from being confused with "reproduces QAQ behavior." It also aligns the test plan and detailed design around schema validation, missing-baseline regression tests, and performance gates.

The tradeoff is higher evaluation cost. Some runs may be blocked by checkpoint licenses, dataset access, GPU availability, or runtime support. In those cases, reports must mark the run incomplete or constrained rather than claiming accepted reproduction. The currently assumed tolerances remain provisional until confirmed.

## Reversal Plan

Supersede this ADR only if the project goal changes away from QAQ reproduction or if official QAQ evaluation guidance defines a different acceptance matrix. Reversal requires updating `doc/requirements.md`, `doc/high-level-design.md`, `doc/detailed-design.md`, `doc/test-plan.md`, result schemas, benchmark scripts, and report language. Existing results that lack required baselines must remain labeled as diagnostic or incomplete.
