# Test Strategy

## Scope

This strategy covers verification for the QAQ research prototype described in `QAQ.pdf`, `doc/requirements.md`, `doc/proposal.md`, and `doc/high-level-design.md`. Official QAQ code is private and unavailable, and `QAQ.pdf` is the only supplementary source material available to the project. The strategy focuses on correctness, reproducibility, benchmark validity, router behavior, bit-plane reconstruction, dynamic loading, and the paper-aligned accuracy, latency, and GPU-memory trade-off.

This strategy does not define production serving tests, UI tests, distributed service tests, or asynchronous prefetching tests because those are non-goals for the first rebuild. The repository currently has no implementation, package manifest, source directory, test directory, scripts, fixtures, or CI configuration, so all test locations and commands below are proposed or require confirmation unless explicitly labeled otherwise.

## Test Objectives

- Catch implementations that run inference but do not demonstrate QAQ's core claims: query-conditioned routing, block-level mixed precision, bit-plane reconstruction, and on-demand CPU-to-GPU loading.
- Prevent invalid comparisons where FP16, static 8-bit, static 4-bit, QAQ on-demand off, and QAQ on-demand on use different checkpoints, tokenizers, datasets, prompt formats, seeds, or metric code.
- Verify that bit-plane artifacts reconstruct valid lower- and higher-precision weights and that static-equivalent QAQ profiles match static quantized baselines within an agreed tolerance.
- Verify that the router emits valid per-block probability distributions or decisions and that routing is not a constant global precision choice.
- Ensure evaluation artifacts include benchmark score or perplexity, latency, GPU memory, routing summary, loader summary, selected GPU IDs, dependency/model metadata, durable logs, and completion status.
- Preserve reproducibility for the first LLaMA-3.1-8B milestone and later paper-aligned Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B comparisons.

## Unit Tests

What should be tested: isolated logic for configuration validation, precision-candidate validation, block ID mapping, bit-plane decomposition and reconstruction, router probability normalization, deterministic tie-breaking, result schema validation, logging event formatting, and loader request validation.

Why it matters: these are the failure points most likely to make later benchmark numbers meaningless while still allowing commands to complete.

Where tests should live: proposed `tests/unit/`.

Example test cases:

- `tests/unit/test_config_validation.py` proposed: reject invalid modes, missing model/tokenizer fields, duplicate or non-positive precision candidates, bit-widths above maximum `B`, invalid GPU IDs, unsafe output-directory reuse, and missing router checkpoint for QAQ evaluation.
- `tests/unit/test_bitplanes.py` proposed: decompose a small known integer tensor into bit-planes, reconstruct 4-bit and 8-bit views, verify shape preservation, verify top-bit-plane selection, and reject missing planes.
- `tests/unit/test_block_registry.py` proposed: assign stable block IDs for a small fake transformer with MHA and FFN blocks, reject unsupported module layouts, and preserve ordering across repeated discovery.
- `tests/unit/test_router_policy.py` proposed: verify probability sums equal 1 within tolerance, temperature changes distribution sharpness, probability ties resolve deterministically, and outputs exist for every configured block.
- `tests/unit/test_results_schema.py` proposed: require model, tokenizer, dataset, split, mode, precision candidates, block granularity, seed, selected GPU IDs, score or perplexity, latency, peak GPU memory, routing summary, logs, and completion status.

Expected result: all tests pass locally without requiring a real LLaMA-3.1-8B checkpoint or GPU. Small tensors, fake model blocks, and temporary directories are sufficient.

Required timing: required before merge for any implementation touching config parsing, quantization, block mapping, router policy, loader validation, logging, or result serialization. At least the config, bit-plane, and router policy tests should exist before implementation work is considered ready for larger integration work.

## Integration Tests

What should be tested: interactions between config loading, model or fake-model adapters, block registry, quantization artifacts, router checkpoints, adaptive runtime, dynamic loader, logs, and result writing.

Why it matters: QAQ is a pipeline; module-level correctness is not enough if artifacts, IDs, devices, or metrics do not line up across stages.

Where tests should live: proposed `tests/integration/`.

Example test cases:

- `tests/integration/test_quantized_artifact_roundtrip.py` proposed: create bit-plane artifacts for a tiny model block, save them, load them, reconstruct requested bit-widths, and verify metadata matches block IDs and maximum `B`.
- `tests/integration/test_static_equivalent_profiles.py` proposed: run all-8-bit and all-4-bit fixed profiles through the QAQ reconstruction path and compare against static 8-bit and static 4-bit paths within agreed numeric tolerance.
- `tests/integration/test_router_checkpoint_contract.py` proposed: save and load a tiny router checkpoint, confirm candidate precision metadata matches the active config, and fail clearly on mismatches.
- `tests/integration/test_on_demand_loader_simulation.py` proposed: simulate CPU-resident bit-planes and GPU materialization with small tensors, verify only requested planes are loaded, verify transfer events are logged, and verify missing planes fail clearly.
- `tests/integration/test_logging_and_incomplete_runs.py` proposed: interrupt or force a controlled failure and verify durable logs plus incomplete-run markers are written.

Expected result: pipeline artifacts are compatible across modules, invalid metadata fails before expensive execution, and every run writes reproducible logs and machine-readable metrics.

Required timing: required before merge for each pipeline stage and required before final submission for the full baseline-to-QAQ workflow.

## End-to-End Tests

What should be tested: full workflows from config to output artifact, including local smoke tests, benchmark subset tests, edge-case validation, output-format validation, reproducibility checks, and final paper-aligned comparison commands.

Why it matters: the project is benchmark-driven. The implementation is only useful if the complete path produces comparable FP16, static 8-bit, static 4-bit, QAQ on-demand off, and QAQ on-demand on results under one reproducible harness.

Where tests should live: proposed `tests/e2e/`; run configs should live in proposed `configs/`; outputs should go under an ignored run directory such as proposed `runs/`.

Example test cases:

- Local smoke test: run one tiny or fake-model prompt through `fp16`, `static_8bit`, `static_4bit`, `fixed_mixed`, `qaq_on_demand_off`, and `qaq_on_demand_on`; verify all modes complete and produce required metadata.
- Public/sample validation test: run a small held-out benchmark subset once benchmark tooling is chosen; verify score or perplexity is present for every required mode.
- Edge-case validation test: empty input, very long input according to configured policy, invalid precision candidate, missing router checkpoint, unavailable selected GPU, missing bit-plane artifact, and existing output directory without overwrite permission.
- Output format validation test: validate result artifacts against the selected schema and verify routing summaries and loader summaries are present in QAQ modes.
- Reproducibility test: run the same small config twice with a fixed seed and verify deterministic fields match and numeric metrics are within expected nondeterminism tolerance.
- Final submission or report test: run the LLaMA-3.1-8B first-milestone matrix and later the full paper-aligned matrix for Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B when resources and model access permit.

Expected result: smoke workflows complete quickly, produce complete result artifacts, and fail clearly when required dependencies or artifacts are absent. Full benchmark workflows produce a comparison table with all required modes and documented deviations.

Required timing: smoke E2E is required before merge for runtime changes. LLaMA-3.1-8B E2E is required before the first milestone is accepted. Full paper-aligned E2E is required before any full reproduction claim.

Commands:

```bash
# proposed: future smoke command after implementation exists
python -m qaq.evaluate --config configs/smoke.yaml --modes fp16 static_8bit static_4bit fixed_mixed qaq_on_demand_off qaq_on_demand_on

# proposed: future first-milestone LLaMA-3.1-8B command after config and CLI exist
python -m qaq.evaluate --config configs/llama31_8b_first_milestone.yaml --modes fp16 static_8bit static_4bit qaq_on_demand_off qaq_on_demand_on

# TODO: confirm command
TODO: confirm command for final paper-aligned reproduction across Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B
```

## Golden Tests

What should be tested: fixed examples whose outputs should remain stable across refactors.

Why it matters: QAQ has several subtle contracts, and stable golden cases protect against accidental changes to reconstruction, routing, result schemas, and comparison-table generation.

Where tests should live: proposed `tests/golden/` with proposed fixtures in `tests/fixtures/`.

Example test cases:

- Golden bit-plane tensor: a small known tensor with expected bit-plane representation and expected 4-bit and 8-bit reconstructions.
- Golden router decision: fixed hidden states, router weights, candidate bit-widths, and temperature producing a known probability vector and deterministic precision decision.
- Golden config validation: a minimal valid smoke config and known invalid configs with expected error categories.
- Golden result artifact: a complete fake QAQ result containing routing summary, loader summary, latency, memory, logs, hardware metadata, and completion status.
- Golden report row: one fixed comparison table row for FP16, static 8-bit, static 4-bit, QAQ on-demand off, and QAQ on-demand on using fake metrics.

Expected result: golden outputs match exactly for schema/text fields and within explicit numeric tolerance for floating-point values.

Required timing: required before merge once each artifact format is introduced. Required before final submission for report/table generation.

## Regression Tests

What should be tested: known failure modes, paper-specific edge cases, and mistakes that would invalidate QAQ claims.

Why it matters: regressions in this project are likely to be scientific validity failures, not only crashes.

Where tests should live: proposed `tests/regression/`.

Example test cases:

- All queries choose the same precision: run should be flagged as failing to demonstrate query-adaptive behavior unless explicitly labeled as a diagnostic baseline.
- All blocks choose the same precision: run should be flagged as failing to demonstrate block-wise adaptation unless explicitly labeled as a diagnostic baseline.
- Static baselines missing: QAQ results must not be reported as accepted if static 8-bit or static 4-bit did not run on the same benchmark settings.
- Different tokenizer or prompt format across modes: comparison should fail validation.
- Missing loader summary in `qaq_on_demand_on`: result should fail schema validation.
- On-demand transfer timing hidden inside generic latency only: result should fail observability validation when loader timing is required.
- Malformed result from interrupted run: result must be marked incomplete and excluded from accepted comparisons.

Expected result: invalid comparisons are rejected or clearly marked incomplete; diagnostic modes are labeled as such and cannot satisfy QAQ acceptance by accident.

Required timing: required before merge for evaluation/reporting logic and required before final submission for benchmark claims.

## Property-Based Tests

What should be tested: invariants over randomized tensors, configs, precision sets, and routing outputs.

Why it matters: fixed examples will not cover all bit-width combinations, tensor shapes, candidate precision sets, or routing ties.

Where tests should live: proposed `tests/property/`.

Example test cases:

- For valid candidate bit-widths, each bit-width is positive, no bit-width exceeds maximum `B`, and every requested bit-width maps to available planes.
- For randomized small integer tensors, bit-plane decomposition followed by full-width reconstruction preserves the quantized representation.
- Reconstructing with more selected bit-planes never reduces the number of represented precision levels.
- Router probabilities are finite, non-negative, and sum to 1 for randomized valid hidden states.
- Deterministic tie-breaking returns the same precision decision across repeated runs with the same seed.
- Result artifacts generated from randomized valid metadata always include required keys and never overwrite existing output directories unless overwrite is explicit.

Expected result: invariants hold across randomized examples; invalid generated cases fail with clear validation errors.

Required timing: recommended before merge for bit-plane, config, router, and result-schema modules. Required before final submission if randomized failures expose untested production paths.

## Performance Tests

What should be tested: accuracy or benchmark score, perplexity, end-to-end latency, peak GPU memory, router overhead, loader transfer overhead, and reproducibility of these metrics across modes.

Why it matters: QAQ is defined by an accuracy-memory-latency trade-off. A correct implementation must measure the trade-off, not only produce text or benchmark answers.

Where tests should live: proposed `tests/performance/` for automated threshold checks, proposed `benchmarks/` or `scripts/` for longer benchmark runners, and proposed `runs/` for outputs.

Example test cases:

- First-milestone benchmark: LLaMA-3.1-8B on at least one held-out accuracy task and one perplexity or language-modeling task if feasible.
- Paper-aligned benchmark: Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B on HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande, WikiText-2, and PTB when model and dataset access is available.
- Latency benchmark: measure end-to-end WikiText-2 latency for every required mode with consistent batch size, sequence length, warm-up, and logging policy.
- Memory benchmark: record peak GPU memory for every required mode and per-GPU memory when multiple GPUs are selected.
- Loader benchmark: compare `qaq_on_demand_off` and `qaq_on_demand_on`, verifying on-demand loader events exist and latency overhead is reported.
- Routing benchmark: verify aggregate routing decisions vary across queries or blocks on at least one accepted run.

Expected result: QAQ accuracy should be within the accepted tolerance against static 8-bit, currently assumed to be within 1 percentage point for classification or within 5 percent relative perplexity for language modeling. On-demand QAQ should target at least 5 percent lower peak GPU memory than comparable non-on-demand mode unless explicitly marked as constrained. On-demand latency overhead is acceptable only when measured and compared.

Required timing: lightweight performance smoke checks are required before merge for runtime changes. LLaMA-3.1-8B performance checks are required before first-milestone acceptance. Full paper-aligned performance checks are required before full reproduction claims.

Commands:

```bash
# proposed: future performance smoke command after implementation exists
python -m qaq.evaluate --config configs/perf_smoke.yaml --modes static_8bit qaq_on_demand_off qaq_on_demand_on

# proposed: future paper-table command after benchmark runner exists
python -m qaq.report --config configs/paper_table.yaml --output runs/paper_table/results.json

# TODO: confirm command
TODO: confirm command for GPU memory and latency benchmark once measurement tooling is selected
```

## Manual Verification

What should be checked: scientific validity, report sanity, output completeness, benchmark comparability, and any final submission or reproduction table.

Why it matters: some failures are visible only by reviewing artifacts and experiment context, especially if paper assumptions remain unresolved.

Where checks should live: proposed checklist in `doc/release-checklist.md` only if later requested; for now, this section is the manual checklist.

Example checks:

- Confirm the run uses the intended model checkpoint, tokenizer, dataset split, prompt format, seed, precision candidates, and selected GPU IDs.
- Confirm static 8-bit and static 4-bit baselines are present before accepting QAQ comparisons.
- Confirm QAQ on-demand off and on use the same router checkpoint and candidate precision set.
- Inspect routing summaries to ensure decisions vary by query or block.
- Inspect loader summaries to ensure `qaq_on_demand_on` actually loads selected CPU-resident bit-planes or precision variants.
- Review latency and memory tables for obvious measurement mistakes, such as missing warm-up policy, mixed batch sizes, or absent per-GPU memory metadata.
- Confirm report language documents deviations from the paper and does not claim full reproduction when only smoke or first-milestone results exist.

Expected result: every accepted result is traceable to a complete config, logs, metrics, artifacts, and stated limitations.

Required timing: required before first-milestone acceptance and before any final report or reproduction claim.

## Required Commands

No verification commands were found in the repository. There is no package manifest, source tree, test tree, scripts directory, Makefile, justfile, task file, or CI config. The commands below are therefore proposed or require confirmation.

```bash
# TODO: confirm command
TODO: confirm command for installing project dependencies

# proposed: use if the implementation is Python and pytest is adopted
pytest -q

# proposed: use if Python linting is adopted
ruff check .

# proposed: use if Python formatting checks are adopted
ruff format --check .

# proposed: use if Python type checking is adopted
mypy .

# proposed: future unit and integration test subsets after tests exist
pytest -q tests/unit tests/integration

# proposed: future E2E smoke command after CLI and smoke config exist
python -m qaq.evaluate --config configs/smoke.yaml --modes fp16 static_8bit static_4bit fixed_mixed qaq_on_demand_off qaq_on_demand_on

# proposed: future first-milestone verification command after CLI and config exist
python -m qaq.evaluate --config configs/llama31_8b_first_milestone.yaml --modes fp16 static_8bit static_4bit qaq_on_demand_off qaq_on_demand_on

# TODO: confirm command
TODO: confirm command for final paper-aligned reproduction report
```

Minimum command set once implementation exists:

```bash
# proposed
pytest -q

# proposed
python -m qaq.evaluate --config configs/smoke.yaml --modes fp16 static_8bit static_4bit fixed_mixed qaq_on_demand_off qaq_on_demand_on

# proposed
python -m qaq.evaluate --config configs/llama31_8b_first_milestone.yaml --modes fp16 static_8bit static_4bit qaq_on_demand_off qaq_on_demand_on
```

## CI / Automation Recommendation

Fast local checks should run on CPU or tiny fake models and include unit tests, schema validation, bit-plane toy-tensor tests, router probability tests, config validation, and result serialization tests. These should be the default pre-merge checks because they do not require model downloads or GPUs.

Full local checks should add integration tests, smoke E2E, small benchmark subsets, simulated dynamic-loader behavior, reproducibility checks with fixed seed, and log/incomplete-run validation.

GPU checks should run on a selected RTX 3090 or configured GPU subset and include memory measurement, CUDA loader behavior, static baselines, QAQ on-demand off, and QAQ on-demand on. These checks should be scheduled or manually triggered because they depend on checkpoint access, dataset access, GPU availability, and runtime duration.

CI should start with documentation linting only if tooling is added, then CPU unit/integration tests once implementation exists. GPU benchmark CI is not required until infrastructure is available; if added, it should publish run manifests, logs, metrics, and incomplete-run markers as artifacts.

## Acceptance Gate

Before implementation is considered complete for the first milestone, all of the following must pass or be explicitly marked as blocked by confirmed external constraints:

- Unit tests for config validation, bit-plane reconstruction, block registry, router policy, result schema, and loader request validation pass.
- Integration tests prove bit-plane artifact roundtrip, static-equivalent 4-bit and 8-bit profiles, router checkpoint compatibility, loader event logging, and incomplete-run handling.
- Smoke E2E runs complete for `fp16`, `static_8bit`, `static_4bit`, `fixed_mixed`, `qaq_on_demand_off`, and `qaq_on_demand_on`.
- LLaMA-3.1-8B first-milestone run completes for all required modes where hardware permits.
- Every accepted run writes durable training or inference logs, machine-readable metrics, run config, selected GPU IDs, latency, peak GPU memory, routing summary, loader summary for on-demand mode, and completion status.
- QAQ comparisons use the same checkpoint, tokenizer, dataset split, prompt format, seed policy, precision candidates, and metric implementation as the static baselines.
- QAQ accuracy is within the accepted tolerance against static 8-bit, currently assumed to be within 1 percentage point for classification or within 5 percent relative perplexity for language modeling.
- QAQ on-demand memory reduction and latency overhead are reported; the current memory target is at least 5 percent lower peak GPU memory than comparable non-on-demand mode unless a constrained-hardware exception is documented.
- Routing varies by query or block in at least one accepted QAQ run, or the run is rejected as not demonstrating QAQ behavior.
- No full paper reproduction claim is made until Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B are evaluated on the paper-aligned benchmark set or deviations are clearly labeled.

## Open Testing Questions

- Which implementation language, package manager, and test runner will be used?
- Which exact dependency install, lint, format, type-check, test, benchmark, and report commands should become `found` commands?
- Which external libraries are approved for model loading, quantization, evaluation, datasets, and GPU memory measurement?
- What canonical machine-readable result format should tests validate: JSON, JSONL, CSV, or a combination?
- What exact router-training loss, training dataset, calibration data, and held-out split should be used?
- What exact candidate precision set should be tested first: `{4, 8}` or a low/mid/high set such as `{4, 6, 8}`?
- Is MHA/FFN block granularity mandatory for the first milestone, or can whole-layer granularity pass as a temporary simplification?
- What numeric tolerance should static-equivalent QAQ use when comparing against static 4-bit and static 8-bit baselines?
- What exact accuracy, perplexity, memory, and latency thresholds supersede the current assumed tolerances?
- How should tests handle batching when different queries in one batch select different precision profiles?
- Are generation workloads in scope for first-milestone E2E tests, or should tests focus on benchmark scoring and perplexity?
