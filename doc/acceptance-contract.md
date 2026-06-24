# QAQ Research Acceptance Contract

This contract prevents smoke, fake, diagnostic, tiny fixture, sampled, or local CPU paths from being treated as completed QAQ research evidence. Diagnostic paths are allowed only as health checks. They must never satisfy `done`, `accepted`, `paper_result`, `benchmark_result`, or `real_qaq_result`.

The repository evidence ladder is: diagnostic fake path, tiny real-mechanism path, real-subset path, and accepted benchmark path. Result artifacts use the machine values below, but acceptance decisions must preserve that ladder: diagnostic fake and tiny-mechanism evidence are rejected as accepted benchmark evidence, and real-subset evidence is rejected when the run is a subset/debug run rather than the full comparable benchmark contract.

## Evidence Levels

### `diagnostic_health_check`

Scope: fake smoke configs, local fixtures, tiny synthetic data, router health checks, fake CPU runtime, sampled/truncated artifacts, and any run marked diagnostic.

May prove: wiring, schema shape, logging, routing trace shape, loader event shape, checkpoint reload contracts, and failure handling.

Must not prove: QAQ accuracy, latency, GPU memory, on-demand memory savings, paper-table comparisons, benchmark support, or implementation completion.

### `real_path_implemented`

Scope: real model adapter, real dataset adapter, real bit-plane artifact format, real runtime path, or GPU-selector-aware execution path exists, but a completed benchmark comparison is not yet present.

May prove: a real implementation mechanism exists and can be validated in isolation.

Must not prove: accepted benchmark result or paper-aligned QAQ claim unless the full comparison matrix and acceptance fields below pass.

### `accepted_experiment_result`

Scope: one result artifact that passes the artifact-level acceptance contract and can participate in a report-level accepted comparison. A report-level accepted comparison still requires all required modes under comparable settings.

Requires: real model snapshot, real benchmark dataset, non-fake split, full tensor-native runtime artifacts for quantized/QAQ modes, actual mixed-precision forward application for quantized/QAQ modes, GPU selector record for large-model experiments, result artifacts, metrics, and report aggregation.

## Required Result Artifact Fields

Every result artifact must include:

- `evidence_level`
- `diagnostic`
- `dataset_is_fake`
- `model_is_fake`
- `tokenizer_is_fake`
- `artifact_scope`
- `artifact_ref_mode`
- `mixed_precision_forward_applied`
- `benchmark_name`
- `benchmark_split`
- `gpu_selector_record`
- `accepted_as_qaq_result`
- `rejection_reasons`

## Artifact-Level Rejection Rules

`accepted_as_qaq_result` must be `false` when any of these are true:

- `diagnostic` is true.
- `dataset_is_fake` is true.
- `model_is_fake` is true.
- `tokenizer_is_fake` is true.
- The benchmark, split, artifact scope, or metadata indicates smoke, fixture, synthetic, tiny, sampled, truncated, router health-check, or diagnostic data.
- `completion_status` is not `completed`.
- `mode` is `fixed_mixed`.
- `mode` is `static_8bit`, `static_4bit`, `qaq_on_demand_off`, or `qaq_on_demand_on` and `mixed_precision_forward_applied` is not true.
- `mode` is `static_8bit`, `static_4bit`, `qaq_on_demand_off`, or `qaq_on_demand_on` and `artifact_ref_mode` is not `full_tensor_index`.
- `artifact_ref_mode` is `partial_tensor_index` or `legacy_bit_width_index`.
- A large-model experiment lacks a `gpu_selector_record` from `scripts/gpu_run.py`.
- A GPU-required or large-model run has a GPU selector record but lacks non-empty `selected_physical_gpu_ids`.

The schema must record every applicable rejection in `rejection_reasons`. Result validation rejects contradictory artifacts, such as a fake dataset marked accepted.

## Report-Level Acceptance Rules

A paper-like QAQ comparison is accepted only when all five modes are present:

- `fp16`
- `static_8bit`
- `static_4bit`
- `qaq_on_demand_off`
- `qaq_on_demand_on`

All five must share the same model, tokenizer, benchmark, split, prompt format, metric, precision candidates, seed, and block granularity. QAQ modes must include routing summaries. `qaq_on_demand_on` must include loader summary and loader activity. Every artifact in the comparison must have `accepted_as_qaq_result: true`.

If any condition is missing, `qaq.report` must return `invalid` or `diagnostic` with explicit rejection reasons instead of comparing silently.

## Evidence Rejection Rules

A result artifact must be rejected as accepted evidence if any of the following are true:

- `diagnostic == true`
- `fake == true`
- dataset is `fake_smoke`
- model id starts with `fake-` or `fake_`
- tokenizer id starts with `fake-` or `fake_`, or otherwise indicates smoke, fixture, synthetic, toy, or tiny provenance
- model source is mocked, synthetic, TinyHFModel, or fixture-only
- `mixed_precision_forward_applied != true` for quantized or QAQ modes
- `artifact_ref_mode != full_tensor_index` for accepted quantized runtime evidence
- selected physical GPU IDs are missing or empty for GPU-required runs
