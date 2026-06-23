# Implementation Prompt

## Objective

Implement the QAQ research prototype end to end in this repository.

QAQ means Query-Adaptive Quantization as described in `QAQ.pdf`: a query-adaptive mixed-precision LLM inference system that decomposes weights into bit-planes, uses a lightweight trainable router to select block-level precision per query, and optionally loads selected bit-planes from CPU to GPU on demand.

The first milestone target is LLaMA-3.1-8B on a configurable subset of the available 8 NVIDIA GeForce RTX 3090 GPUs, each with 24 GiB VRAM. Full paper-aligned reproduction remains Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B on HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande, WikiText-2, and PTB.

Build a staged, testable research prototype. Do not build a production serving system. Do not claim paper reproduction until the required baselines, adaptive modes, metrics, logs, routing evidence, loader evidence, and benchmark coverage exist.

## Inputs to Read First

Read these before editing code:

- `QAQ.pdf` present. Primary source for QAQ architecture, equations, modes, metrics, model families, and limitations. It is a 5-page PDF.
- `README.md` present. Currently minimal: only `# qaq`.
- `doc/problem-brief.md` present. Optional input, but read it because it frames scope, success metrics, risks, and unknowns.
- `doc/repo-map.md` absent. There is no repo map yet.
- `doc/quality-gates.md` absent. Use `doc/test-plan.md`, `doc/detailed-design.md`, and this prompt for quality gates until that file exists.
- `doc/requirements.md` present. Read it even though it is not in the skill's minimum list; it contains confirmed user requirements, hardware facts, acceptance expectations, and unresolved details.
- `doc/proposal.md` present. Required.
- `doc/high-level-design.md` present. Required.
- `doc/test-plan.md` present. Required.
- `doc/detailed-design.md` present. Required.
- `doc/tasks/` present. Required. Important task files:
  - `doc/tasks/progress.md`
  - `doc/tasks/experiment-configuration-and-run-manifest.md`
  - `doc/tasks/model-and-benchmark-adapter.md`
  - `doc/tasks/block-registry-and-precision-plan.md`
  - `doc/tasks/quantization-and-bit-plane-store.md`
  - `doc/tasks/static-and-fixed-mixed-precision-runtime.md`
  - `doc/tasks/router-policy-module.md`
  - `doc/tasks/router-training-pipeline.md`
  - `doc/tasks/adaptive-inference-runtime.md`
  - `doc/tasks/dynamic-loader-and-memory-residency-manager.md`
  - `doc/tasks/evaluation-metrics-and-results-reporter.md`
  - `doc/tasks/logging-and-progress-tracking.md`
- `doc/adr/` present. Read all ADRs because they are accepted decisions:
  - `doc/adr/0001-build-a-staged-research-prototype.md`
  - `doc/adr/0002-use-bit-plane-artifacts-for-quantized-weights.md`
  - `doc/adr/0003-use-mha-and-ffn-blocks-as-primary-precision-granularity.md`
  - `doc/adr/0004-separate-adaptive-routing-from-on-demand-loading.md`
  - `doc/adr/0005-use-synchronous-on-demand-loading-for-the-first-rebuild.md`
  - `doc/adr/0006-require-baseline-comparable-evaluation-for-accepted-qaq-results.md`

After reading docs, inspect the current repository again with `rg --files -uu`, `git status --short`, and targeted file reads. The repo may have changed since this prompt was generated.

## Current Implementation

Current state observed while generating this prompt:

- Repository root: `/home/kuotzuwei15/qaq`.
- `README.md` contains only `# qaq`.
- `QAQ.pdf` is present and is the only paper source. Official QAQ code is private and unavailable.
- `doc/` contains planning documents, ADRs, and task breakdowns.
- `doc/tasks/progress.md` has all module checkboxes unchecked and all full-project gates unchecked.
- `doc/repo-map.md` is absent.
- `doc/quality-gates.md` is absent.
- No implementation package exists. There is no `qaq/` package or source tree.
- No `tests/`, `configs/`, `scripts/`, `benchmarks/`, or `runs/` directories were present.
- No package manifest or build configuration was found: no `pyproject.toml`, `requirements.txt`, `setup.py`, `package.json`, `Makefile`, `justfile`, or CI config.
- No test, lint, format, type-check, benchmark, or report command is currently configured.
- `git status --short` showed `?? doc/`, so the planning docs were untracked at the time of inspection. Do not revert or discard them.

The task docs propose a Python-shaped implementation, including paths such as `qaq/config.py`, `qaq/bitplanes.py`, `qaq/runtime/adaptive.py`, `qaq/router/policy.py`, `tests/unit/`, `tests/integration/`, and commands such as `pytest -q` and `python -m qaq.evaluate`. Unless the repository has changed or the user gives a different direction, use Python as the implementation language and create a minimal package/test scaffold before module work.

## Hard Constraints

- Rebuild QAQ from `QAQ.pdf` and local planning docs. Do not use official QAQ code; it is private and unavailable.
- Preserve the core QAQ components:
  - bit-plane weight representation,
  - trainable query-conditioned router,
  - block-wise mixed-precision inference,
  - optional on-demand CPU-to-GPU loading.
- Use a staged research prototype architecture. Validate static baselines, bit-plane reconstruction, fixed mixed precision, router behavior, adaptive inference, dynamic loading, and reporting separately.
- Bit-plane artifacts are the primary quantized weight representation. Independent full quantized model copies may be useful diagnostics but must not replace bit-plane artifacts for accepted QAQ results.
- Bit-plane artifacts must record model identity, block ID, tensor metadata, maximum bit-width `B`, available bit-plane indices, quantization parameters, reconstruction policy, version, integrity or validation status, and compatibility metadata.
- The paper gives `B = 8` as the example maximum bit-width. Lower effective precision uses selected most significant bit-planes.
- Candidate precision must include 4-bit and 8-bit behavior. A mid precision such as 6-bit is optional until approved or documented as an implementation assumption.
- MHA and FFN are the primary controlled block granularity. Whole-layer control is only a documented fallback if MHA/FFN-level control is infeasible, and such runs cannot be presented as fully paper-aligned without that limitation.
- The router must be lightweight relative to the base LLM. The paper describes it as an MLP over block hidden representation `h_j(x)`.
- Router output must be query-dependent and block-specific: probabilities or deterministic decisions over candidate bit-widths for every controlled block.
- Router training must use a full-precision teacher, a quantized student, frozen base LLM parameters, and a documented knowledge-distillation objective. The exact loss, training data, optimizer, and feature point are under-specified by the paper; choose and record a concrete method before any trained-router result is claimed.
- Static baselines are mandatory. Accepted comparisons must include `fp16`, `static_8bit`, `static_4bit`, `qaq_on_demand_off`, and `qaq_on_demand_on` where hardware permits. `fixed_mixed` is diagnostic and does not replace required modes.
- `qaq_on_demand_off` and `qaq_on_demand_on` must use the same routing semantics and candidate precision set. The intended difference is GPU-resident artifacts versus synchronous on-demand loading from CPU.
- The first on-demand loader must be synchronous. Do not claim asynchronous overlap, prefetching, or advanced scheduling in the first rebuild.
- Every accepted result must include benchmark score or perplexity, end-to-end latency, peak GPU memory, selected GPU IDs, routing summary for QAQ modes, loader summary for `qaq_on_demand_on`, durable logs, machine-readable metrics, and completion status.
- QAQ comparisons must use the same model checkpoint, tokenizer, dataset split, prompt format, precision candidates, seed policy, runtime settings where relevant, and metric implementation as static baselines.
- First-milestone acceptance targets LLaMA-3.1-8B. Full paper-aligned reproduction requires Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B across HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande, WikiText-2, and PTB.
- First-milestone execution does not need all 8 GPUs. GPU selection must be configurable and recorded.
- CPU-only, fake-model, and small-tensor tests are valid for correctness checks, schema checks, and smoke tests. They are not valid evidence for GPU memory or on-demand loading claims.
- The provisional accuracy target is within 1 percentage point of static 8-bit for classification or within 5 percent relative perplexity of static 8-bit for language modeling unless superseded.
- The provisional on-demand memory target is at least 5 percent lower peak GPU memory than the comparable non-on-demand or static mode unless a constrained-hardware exception is documented.
- Do not fabricate scores or treat unavailable datasets, models, GPUs, or credentials as successful runs. Mark runs incomplete, diagnostic, blocked, or constrained.
- Do not overwrite existing output directories unless an explicit overwrite option is set.
- Do not revert user or other-agent edits. Work with the current tree.

## Non-Goals

- Production serving, API service design, autoscaling, or deployment infrastructure.
- Training or fine-tuning the base LLM weights.
- UI, dashboard, or visualization product work.
- Exact reproduction of every paper number before missing details, model access, dataset access, and runtime choices are resolved.
- Supporting every model family in the first implementation milestone.
- Static-only quantization as the final target.
- Heuristic or fixed routing as an accepted QAQ result unless explicitly labeled diagnostic.
- Asynchronous prefetching, overlap, advanced memory scheduling, or disk-backed loading in the first rebuild.
- Full paper-reproduction claims from fake-model, tiny-model, smoke, or partial benchmark results.

## Execution Model

- Read all inputs listed above before making code edits.
- Re-inspect the repository before implementation. Prefer repository facts over this prompt if files have changed.
- Maintain `doc/tasks/progress.md`. Add timestamped or checkpoint-style progress entries after each module or small workstream starts, completes, is blocked, or is verified.
- Implement one module or small workstream at a time. Keep each workstream independently testable with fake models, small tensors, or fixtures before large GPU work.
- Use subagents only for independent workstreams with disjoint write scopes. Keep shared contracts and integration files under the main agent unless the write boundary is explicit.
- Avoid reverting edits made by the user or other agents.
- Keep implementation close to the task file write scopes. Do not introduce unrelated refactors.
- If no package scaffold exists, create a minimal Python package first, then add modules incrementally.
- Prefer small, explicit data contracts over hidden global state. Cross-module interactions should pass config, metadata, artifact IDs, events, or typed records.
- Run module-specific checks after each module. If a check cannot run because tooling or dependencies are missing, create or document the closest available check.
- Run full configured quality gates at the end.
- Provide command output summaries as evidence in the final response.
- Stop and ask only when truly blocked by missing requirements, conflicting docs, destructive choices, credentials, external services, unavailable model/dataset access, or quality gates that cannot be run.

## Module Workstreams

**Workstream 0: Project Scaffold and Shared Test Harness**

- Expected files: `pyproject.toml` or equivalent Python project metadata, `qaq/__init__.py`, shared package directories, `tests/`, `tests/fixtures/`, `configs/`, and `.gitignore` updates only if needed.
- Owns initial test runner setup and importability.
- Keep dependency choices minimal. If network or dependency installation is unavailable, make the scaffold usable with local/fake tests first.
- Verification: `python -m pytest -q` once pytest is configured, or the closest available import/test command.

**Workstream 1: Experiment Configuration and Run Manifest**

- Expected files: `qaq/config.py`, `qaq/manifest.py`, `qaq/errors.py`, `configs/`, `tests/unit/test_config_validation.py`, `tests/fixtures/configs/`.
- Owns run config schema, validation, output overwrite policy, GPU selection validation, mode-specific router checkpoint requirements, manifest status, and incomplete markers.
- Must validate required fields before model loading.
- Verification: `pytest -q tests/unit/test_config_validation.py`.

**Workstream 2: Logging and Progress Tracking**

- Expected files: `qaq/logging.py`, `qaq/progress.py`, `qaq/status.py`, `tests/unit/test_logging_events.py`, `tests/integration/test_logging_and_incomplete_runs.py`.
- Owns structured log events, console progress state, durable logs, flushing, completion/failure status, and incomplete-run markers.
- Must support training fields, inference/evaluation fields, loader events, routing summaries, warnings, and failures.
- Verification: `pytest -q tests/unit/test_logging_events.py tests/integration/test_logging_and_incomplete_runs.py`.

**Workstream 3: Block Registry and Precision Plan**

- Expected files: `qaq/blocks.py`, `qaq/precision_plan.py`, `tests/unit/test_block_registry.py`, `tests/fixtures/fake_transformer.py`.
- Owns stable MHA/FFN block IDs, block descriptors, mode-specific precision plans, precision validation, unsupported-layout failures, and no silent full-precision fallback in quantized modes.
- Verification: `pytest -q tests/unit/test_block_registry.py`.

**Workstream 4: Quantization and Bit-Plane Store**

- Expected files: `qaq/quantization.py`, `qaq/bitplanes.py`, `qaq/artifacts.py`, `tests/unit/test_bitplanes.py`, `tests/integration/test_quantized_artifact_roundtrip.py`, `tests/golden/`, `tests/fixtures/bitplanes/`.
- Owns toy/small-tensor quantization, bit-plane decomposition, top-bit-plane reconstruction, artifact metadata, save/load roundtrip, validation, and golden fixtures.
- Must document the chosen quantization scheme, scale/zero-point/grouping assumptions, and reconstruction policy.
- Verification: `pytest -q tests/unit/test_bitplanes.py tests/integration/test_quantized_artifact_roundtrip.py`.

**Workstream 5: Model and Benchmark Adapter**

- Expected files: `qaq/model_adapter.py`, `qaq/benchmark_adapter.py`, `qaq/data.py`, `tests/integration/test_model_adapter_smoke.py`, benchmark fixtures under `tests/fixtures/`.
- Owns model/tokenizer loading abstraction, fake/tiny model support, benchmark examples, prompt formatting, context-length policy, FP16/reference outputs, hidden features keyed by block ID, and metadata for comparability.
- Full LLaMA-3.1-8B access may be unavailable. Keep fake/tiny adapter tests passing without real checkpoints or GPUs.
- Verification: `pytest -q tests/integration/test_model_adapter_smoke.py`.

**Workstream 6: Static and Fixed Mixed-Precision Runtime**

- Expected files: `qaq/runtime/static.py`, `qaq/runtime/common.py`, `qaq/evaluate.py`, `tests/integration/test_static_equivalent_profiles.py`, `tests/e2e/test_smoke_modes.py`, smoke configs.
- Owns `fp16`, `static_8bit`, `static_4bit`, and `fixed_mixed` runtime paths, raw outputs, timing events, memory events, and baseline metadata.
- Must keep fixed mixed precision diagnostic only.
- Verification: `pytest -q tests/integration/test_static_equivalent_profiles.py tests/e2e/test_smoke_modes.py`.

**Workstream 7: Router Policy Module**

- Expected files: `qaq/router/policy.py`, `qaq/router/checkpoint.py`, `qaq/router/types.py`, `tests/unit/test_router_policy.py`, `tests/integration/test_router_checkpoint_contract.py`, `tests/golden/`.
- Owns router checkpoint metadata, lightweight MLP-compatible scoring interface, probability normalization, finite-value validation, deterministic decision policy, tie-breaking, and router traces.
- Must flag constant global precision behavior unless explicitly diagnostic.
- Verification: `pytest -q tests/unit/test_router_policy.py tests/integration/test_router_checkpoint_contract.py`.

**Workstream 8: Router Training Pipeline**

- Expected files: `qaq/router/train.py`, `qaq/router/losses.py`, `qaq/router/checkpoint.py`, `configs/router_train_smoke.yaml`, `tests/integration/test_router_checkpoint_contract.py`, `tests/integration/test_logging_and_incomplete_runs.py`.
- Owns router training config, preflight validation, frozen base LLM checks where detectable, teacher/student signal use, approved distillation objective, checkpointing, training metrics, and logs.
- If no real router loss/data is approved, implement explicit preflight failure for real training and a tiny/fake training path for tests. Do not silently launch long jobs with an undocumented heuristic.
- Verification: `pytest -q tests/integration/test_router_checkpoint_contract.py tests/integration/test_logging_and_incomplete_runs.py`.

**Workstream 9: Dynamic Loader and Memory Residency Manager**

- Expected files: `qaq/runtime/loader.py`, `qaq/loader.py`, `tests/unit/test_loader_validation.py`, `tests/integration/test_on_demand_loader_simulation.py`.
- Owns loader request/event types, residency map, synchronous CPU-to-device materialization for small tensors, cache-hit/load/release/failure events, transfer timing, bytes where available, and loader summaries.
- Must fail clearly when bit-planes are missing, bit-widths are invalid, CUDA is unavailable for real on-demand mode, or GPU memory is insufficient.
- Verification: `pytest -q tests/unit/test_loader_validation.py tests/integration/test_on_demand_loader_simulation.py`.

**Workstream 10: Adaptive Inference Runtime**

- Expected files: `qaq/runtime/adaptive.py`, `qaq/runtime/common.py`, `qaq/evaluate.py`, `tests/e2e/test_smoke_modes.py`, `tests/regression/test_qaq_acceptance_guards.py`, QAQ smoke configs.
- Owns `qaq_on_demand_off` and `qaq_on_demand_on`, router feature collection, precision-plan application, loader requests, adaptive traces, runtime status, latency, memory, and comparability metadata.
- The two QAQ modes must share routing semantics. Loader behavior is the only intended difference.
- Verification: `pytest -q tests/e2e/test_smoke_modes.py tests/regression/test_qaq_acceptance_guards.py`.

**Workstream 11: Evaluation Metrics and Results Reporter**

- Expected files: `qaq/results.py`, `qaq/metrics.py`, `qaq/report.py`, `qaq/evaluate.py`, `tests/unit/test_results_schema.py`, `tests/golden/`, `tests/regression/test_qaq_acceptance_guards.py`, report configs.
- Owns result artifact schema, metric aggregation hooks, comparison grouping, accepted/incomplete/invalid/diagnostic states, paper-table rows, and acceptance guards.
- Must reject QAQ acceptance when static baselines are missing, settings differ across modes, routing summaries are missing, or `qaq_on_demand_on` lacks loader summaries.
- Verification: `pytest -q tests/unit/test_results_schema.py tests/regression/test_qaq_acceptance_guards.py`.

**Workstream 12: Smoke, First-Milestone, and Report Configs**

- Expected files: `configs/smoke.yaml`, `configs/perf_smoke.yaml`, `configs/llama31_8b_first_milestone.yaml`, `configs/paper_table.yaml`, optional `benchmarks/` or `scripts/` if needed.
- Owns runnable config stubs, mode matrix commands, result output locations, and report commands.
- Must separate fake/tiny smoke configs from real LLaMA-3.1-8B configs.
- Verification: smoke E2E and report-generation commands once the CLI exists.

## Subagent Plan

Use subagents only after the main agent establishes package layout, shared data contracts, and initial tests. If no subagent tooling is available, implement the same plan sequentially.

Good subagent candidates with disjoint scopes:

- Config/logging subagent: `qaq/config.py`, `qaq/manifest.py`, `qaq/errors.py`, `qaq/logging.py`, `qaq/progress.py`, `qaq/status.py`, config/logging tests and fixtures.
- Block/quantization subagent: `qaq/blocks.py`, `qaq/precision_plan.py`, `qaq/quantization.py`, `qaq/bitplanes.py`, `qaq/artifacts.py`, fake transformer and bit-plane tests.
- Router subagent: `qaq/router/policy.py`, `qaq/router/types.py`, router policy tests, golden router decisions. Coordinate `qaq/router/checkpoint.py` with the main agent or router-training owner.
- Loader subagent: `qaq/runtime/loader.py`, `qaq/loader.py`, loader unit and simulation tests.
- Results subagent: `qaq/results.py`, `qaq/metrics.py`, `qaq/report.py`, result schema tests, report row golden fixtures, acceptance-guard tests.

Keep these shared or integration-heavy files main-agent owned unless a single owner is explicitly assigned:

- `pyproject.toml`
- `qaq/evaluate.py`
- `qaq/runtime/common.py`
- `qaq/router/checkpoint.py`
- `tests/e2e/test_smoke_modes.py`
- `tests/regression/test_qaq_acceptance_guards.py`
- `tests/golden/` when multiple modules need fixtures
- `configs/*.yaml`
- `doc/tasks/progress.md`

Subagents must not edit overlapping files concurrently. The main agent should merge subagent results by running the relevant module tests, inspecting shared schema compatibility, updating `doc/tasks/progress.md`, and only then proceeding to the next integration step.

## Implementation Order

1. Re-read the docs and inspect the current repo state.
   - Verification: summarize changed repo facts before coding.

2. Create the Python package scaffold and test harness if still absent.
   - Add only minimal project metadata, package directories, fixtures, and test runner setup needed for local work.
   - Verification: `python -m pytest -q` or closest available command.

3. Implement shared errors, config, manifest, status, and logging primitives.
   - These are foundational for every long-running module.
   - Verification: `pytest -q tests/unit/test_config_validation.py tests/unit/test_logging_events.py`.

4. Implement block registry and precision-plan validation with a fake transformer.
   - Use MHA/FFN as primary granularity.
   - Verification: `pytest -q tests/unit/test_block_registry.py`.

5. Implement bit-plane decomposition, reconstruction, metadata, and artifact roundtrip for small tensors.
   - Add golden fixtures before whole-model integration.
   - Verification: `pytest -q tests/unit/test_bitplanes.py tests/integration/test_quantized_artifact_roundtrip.py`.

6. Implement fake/tiny model and benchmark adapter.
   - Expose FP16/reference outputs and hidden features keyed by block ID.
   - Verification: `pytest -q tests/integration/test_model_adapter_smoke.py`.

7. Implement static and fixed mixed-precision runtime paths.
   - Get `fp16`, `static_8bit`, `static_4bit`, and `fixed_mixed` smoke paths working before router work is trusted.
   - Verification: `pytest -q tests/integration/test_static_equivalent_profiles.py tests/e2e/test_smoke_modes.py`.

8. Implement router policy and checkpoint compatibility.
   - Choose a deterministic decision policy and document it. If no better choice is approved, default to argmax with a stable tie-break and record the assumption.
   - Verification: `pytest -q tests/unit/test_router_policy.py tests/integration/test_router_checkpoint_contract.py`.

9. Implement synchronous loader simulation and validation.
   - Keep CPU-resident versus GPU/device-resident state explicit.
   - Verification: `pytest -q tests/unit/test_loader_validation.py tests/integration/test_on_demand_loader_simulation.py`.

10. Implement adaptive runtime for `qaq_on_demand_off` and `qaq_on_demand_on`.
    - Reuse the same router semantics for both modes.
    - Verification: `pytest -q tests/e2e/test_smoke_modes.py tests/regression/test_qaq_acceptance_guards.py`.

11. Implement result artifacts, metrics, comparison validation, and report generation.
    - Reject invalid QAQ claims even when individual runs completed.
    - Verification: `pytest -q tests/unit/test_results_schema.py tests/regression/test_qaq_acceptance_guards.py`.

12. Implement router training scaffolding and tiny/fake training tests.
    - Document or require a concrete KD loss before real training. Freeze base parameters where detectable.
    - Verification: `pytest -q tests/integration/test_router_checkpoint_contract.py tests/integration/test_logging_and_incomplete_runs.py`.

13. Add smoke configs and CLI commands for the full fake/tiny mode matrix.
    - Verification: `python -m qaq.evaluate --config configs/smoke.yaml --modes fp16 static_8bit static_4bit fixed_mixed qaq_on_demand_off qaq_on_demand_on`.

14. Add first-milestone LLaMA-3.1-8B config stubs and real-run preflight checks.
    - Do not require model downloads or credentials inside unit tests.
    - Verification when access exists: `python -m qaq.evaluate --config configs/llama31_8b_first_milestone.yaml --modes fp16 static_8bit static_4bit qaq_on_demand_off qaq_on_demand_on`.

15. Run the full configured quality gates and update `doc/tasks/progress.md` with final verification evidence.

## Testing and Quality Gates

No configured commands were found in the repository when this prompt was generated. Use these proposed commands once the corresponding tooling exists:

```bash
pytest -q
ruff check .
ruff format --check .
mypy .
pytest -q tests/unit tests/integration
python -m qaq.evaluate --config configs/smoke.yaml --modes fp16 static_8bit static_4bit fixed_mixed qaq_on_demand_off qaq_on_demand_on
python -m qaq.evaluate --config configs/llama31_8b_first_milestone.yaml --modes fp16 static_8bit static_4bit qaq_on_demand_off qaq_on_demand_on
python -m qaq.report --config configs/paper_table.yaml --output runs/paper_table/results.json
```

If `ruff`, `mypy`, or another static-analysis tool is not configured, do not pretend it passed. Either configure it deliberately or state that it is not configured and run the closest available checks.

Required fast checks before treating the implementation as coherent:

- Config validation tests.
- Logging event and incomplete-run tests.
- Block registry tests.
- Bit-plane unit, golden, and artifact roundtrip tests.
- Router policy and checkpoint contract tests.
- Loader validation and simulation tests.
- Result schema and acceptance-guard tests.

Required integration/E2E checks before first-milestone acceptance:

- Static-equivalent all-4-bit and all-8-bit profiles match static paths within the approved tolerance.
- Smoke run completes for `fp16`, `static_8bit`, `static_4bit`, `fixed_mixed`, `qaq_on_demand_off`, and `qaq_on_demand_on`.
- QAQ routing summaries show variation by query or block, or the run is rejected as non-adaptive unless diagnostic.
- `qaq_on_demand_on` result includes loader summaries and transfer timing.
- Durable training/inference/evaluation logs and machine-readable metrics are written.
- Interrupted or failed runs are marked incomplete.

Required performance/evaluation checks for accepted QAQ claims:

- Full FP16, static 8-bit, static 4-bit, QAQ on-demand off, and QAQ on-demand on run under comparable settings where hardware permits.
- Every accepted run includes score or perplexity, latency, peak GPU memory, selected GPU IDs, routing summary, loader summary for on-demand mode, logs, and completion status.
- LLaMA-3.1-8B first-milestone run completes for all required modes where hardware, model access, and datasets permit.
- Full paper-aligned claims require Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B on HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande, WikiText-2, and PTB, or explicit deviation labels.

## Progress Tracking

Maintain `doc/tasks/progress.md` throughout implementation.

Update it after every module or small workstream:

- mark started work,
- mark completed implementation,
- mark blocked work with the blocker,
- mark verified work with the command that passed,
- record skipped or unavailable checks with the reason,
- add timestamps or checkpoint entries so long-running work is auditable.

The progress file currently contains module checkboxes and full-project gates. You may extend it with a checkpoint log table or dated notes. Keep updates factual and tied to actual commands or blockers.

The main agent owns progress updates even when subagents implement modules.

## Commit or Checkpoint Strategy

Do not create commits unless the user asks for commits or repository docs later require them.

Use logical checkpoints in the working tree:

- scaffold and shared contracts,
- config/logging,
- block and bit-plane foundations,
- static runtime,
- router policy/training,
- loader/adaptive runtime,
- evaluation/reporting,
- smoke configs and final quality gates.

If commits are requested, make small commits grouped by workstream and include tests with the implementation they verify. Never include unrelated files in a checkpoint or commit.

## Acceptance Criteria

The implementation is accepted only when all applicable items are true:

- Required QAQ behavior is implemented: bit-plane artifacts, query-conditioned router, block-wise mixed precision, QAQ on-demand off, and QAQ on-demand on.
- All module task files under `doc/tasks/` are completed or explicitly marked blocked with a concrete reason.
- Static baselines are implemented and mandatory for accepted QAQ comparisons: `fp16`, `static_8bit`, and `static_4bit`.
- Fixed mixed precision exists as a diagnostic validation mode.
- Tests are added or updated for config validation, logging, block registry, bit-planes, artifact roundtrip, static-equivalent profiles, router policy, checkpoint contracts, loader simulation, adaptive smoke, result schema, and acceptance guards.
- Build/import checks pass when configured.
- Unit tests pass.
- Integration tests pass.
- E2E smoke mode matrix passes for fake/tiny configuration.
- Lint passes if configured.
- Format check passes if configured.
- Type/static analysis passes if configured.
- Evaluator or benchmark commands pass when configured and when required model/dataset/GPU access is available.
- LLaMA-3.1-8B first-milestone run completes for required modes where hardware permits, or external blockers are documented without claiming success.
- Accepted results include benchmark score or perplexity, latency, GPU memory, selected GPU IDs, routing summaries, loader summaries for on-demand mode, durable logs, run manifests, machine-readable metrics, and completion status.
- QAQ comparisons use common checkpoint, tokenizer, dataset split, prompt format, precision candidates, seed policy, metric implementation, and relevant runtime settings.
- QAQ accuracy and memory claims meet the currently assumed thresholds or explicitly document deviations and constrained-hardware exceptions.
- Docs are updated if implementation choices resolve open questions, especially router loss, quantization scheme, result format, dependency choices, and benchmark commands.
- `doc/tasks/progress.md` is updated with final verification evidence.
- No unrelated files are changed.

## Uncertainty Protocol

Make conservative, documented assumptions when doing so is safe and keeps work moving. Ask the user only when blocked by missing requirements, conflicting docs, destructive choices, credentials, external services, unavailable model or dataset access, or quality gates that cannot be run.

Safe default assumptions if the repo still has no implementation decisions:

- Use Python for the first implementation because all task files propose `qaq/*.py`, `pytest`, and `python -m qaq.evaluate`.
- Use pytest for tests.
- Use JSON or JSONL for manifests, logs, and result artifacts unless a better format is chosen and documented.
- Start fake/tiny correctness tests before real LLaMA-3.1-8B runs.
- Start candidate precision with `[4, 8]`; add 6-bit only after the core path works or a doc/user decision approves low/mid/high as `[4, 6, 8]`.
- Use per-query adaptive execution for initial correctness if batching with different precision plans is unresolved.
- Use deterministic router decisions for accepted runs. If no policy is approved, choose and document a stable argmax tie-break.
- Keep asynchronous prefetching out of scope.

Do not make hidden scientific choices. If you choose a quantization scheme, router feature point, distillation loss, result schema, benchmark framework, tolerance, or GPU memory measurement policy, document it in code comments where needed, config metadata, tests, and docs if it affects interpretation.

Ask before:

- deleting or overwriting user work,
- downloading large models or datasets when approval/credentials are needed,
- using paid external services,
- changing accepted ADR decisions,
- claiming full paper reproduction without the full comparison matrix,
- replacing bit-plane artifacts with independent quantized model copies as the core QAQ path.

## Final Response Requirements

The final response from the implementation session must be concise and include:

- implementation summary,
- changed files grouped by workstream,
- tests and quality gates run, with command output summaries,
- benchmark/evaluator runs and whether they were smoke, first-milestone, diagnostic, constrained, or accepted,
- known limitations and unresolved open questions,
- any follow-up required for model access, dataset access, GPU execution, or paper-aligned reproduction.

If any configured check could not run, state exactly why. Do not imply unrun checks passed.
