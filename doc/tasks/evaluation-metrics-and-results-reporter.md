# Evaluation Metrics and Results Reporter

## Goal

Implement metric aggregation, result schema validation, comparison validation, and report generation so QAQ claims are accepted only when static baselines and adaptive modes are comparable.

## Inputs

- `doc/proposal.md`: Stage 7 requires the full comparison matrix and documented deviations from the paper.
- `doc/high-level-design.md`: Evaluation owns metric definitions and comparison rules while runtimes supply raw outputs and events.
- `doc/detailed-design.md`: Defines Result Artifact, comparison validation algorithm, acceptance states, and missing-summary failure behavior.
- `doc/test-plan.md`: Requires result schema tests, output validation, golden artifacts/report rows, regression tests, performance tests, and manual verification.

## Write Scope

Create or edit proposed paths: `qaq/results.py`, `qaq/metrics.py`, `qaq/report.py`, `qaq/evaluate.py`, `tests/unit/test_results_schema.py`, `tests/golden/`, `tests/regression/test_qaq_acceptance_guards.py`, and report configs.

## Read Scope

Inspect runtime output bundles, manifest fields, router traces, loader events, logging paths, benchmark adapter outputs, and accepted threshold assumptions.

## Dependencies

Experiment Configuration and Run Manifest, Static and Fixed Mixed-Precision Runtime, Adaptive Inference Runtime, Router Policy Module, Dynamic Loader and Memory Residency Manager, Logging and Progress Tracking.

## Tasks

- [x] Define result artifact schema with model, tokenizer, dataset, split, prompt format, mode, precision candidates, block granularity, seed, GPU IDs, hardware, score/perplexity, latency, memory, routing summary, loader summary, logs, and completion status.
- [x] Implement schema validation and incomplete/invalid/diagnostic/accepted comparison states.
- [x] Implement metric aggregation hooks for benchmark score, perplexity, latency, peak GPU memory, routing summaries, and loader summaries.
- [x] Implement comparison grouping by shared model, tokenizer, dataset, split, prompt format, metric, precision candidates, and seed policy.
- [x] Reject QAQ acceptance when static baselines are missing, settings differ across modes, routing summaries are missing, or on-demand loader summaries are missing.
- [x] Add golden result artifact and report-row tests using fake metrics.

## Tests and Quality Gates

- [x] Run `pytest -q tests/unit/test_results_schema.py tests/regression/test_qaq_acceptance_guards.py` when implemented.
- [x] Verify diagnostic modes cannot satisfy QAQ acceptance by accident.
- [x] Verify full reproduction claims require paper-aligned models/benchmarks or explicit deviation labels.

## Done When

- [x] Result artifacts validate all required fields and comparison groups reject invalid QAQ claims.
- [x] Fake paper-table rows are generated from stable fixtures.
- [x] Result schema and acceptance-guard tests pass.

Verified checkpoint:

- 2026-06-24T00:10:29+08:00: Implemented dependency-free result artifacts, metric hooks, comparison validation, report rows, a report CLI, evaluate result-artifact output, golden fixtures, and regression guards. Verified with `python -m pytest -q tests/unit/test_results_schema.py tests/regression/test_qaq_acceptance_guards.py`, `python -m pytest -q`, `python -m qaq.evaluate --config configs/smoke.json --skip-output-dir-check --print-result-json`, and `python -m qaq.report --results tests/golden/result_artifact_static.json --print-json`. Fake/CPU outputs are explicitly diagnostic and cannot satisfy accepted QAQ claims.
