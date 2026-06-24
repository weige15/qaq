# Adaptive Inference Runtime

## Goal

Implement QAQ adaptive inference for on-demand off and on modes, applying router precision plans per query and emitting comparable outputs, traces, latency, memory, and loader request events.

## Inputs

- `doc/proposal.md`: Stages 5 and 6 require adaptive inference with on-demand loading disabled first, then enabled.
- `doc/high-level-design.md`: Adaptive Runtime applies router decisions and stays separate from Dynamic Loader so routing can be evaluated without transfer overhead.
- `doc/detailed-design.md`: Defines QAQ on-demand off/on flows, adaptive trace data, failure behavior, batching uncertainty, and comparability requirements.
- `doc/test-plan.md`: Requires E2E smoke, output validation, edge-case validation, reproducibility, routing benchmark, and performance tests.

## Write Scope

Create or edit proposed paths: `qaq/runtime/adaptive.py`, `qaq/runtime/common.py`, `qaq/evaluate.py`, `tests/e2e/test_smoke_modes.py`, `tests/regression/test_qaq_acceptance_guards.py`, and QAQ smoke configs.

## Read Scope

Inspect model adapter, block registry, bit-plane store, router policy, dynamic loader, evaluation reporter, and logging APIs.

## Dependencies

Model and Benchmark Adapter, Block Registry and Precision Plan, Router Policy Module, Quantization and Bit-Plane Store, Dynamic Loader and Memory Residency Manager, Evaluation Metrics and Results Reporter, Logging and Progress Tracking.

## Tasks

- [x] Implement `qaq_on_demand_off` flow that collects router features, obtains per-block precision plans, materializes selected precision from GPU-resident artifacts, and executes mixed-precision inference.
- [x] Implement `qaq_on_demand_on` flow that uses the same routing semantics and requests selected planes or precision artifacts from the Dynamic Loader.
- [x] Emit adaptive traces with query ID, precision plan, routing trace references, loader request references, latency, memory, and runtime status.
- [x] Validate checkpoint, artifact, block, and precision metadata before adaptive execution.
- [x] Flag constant precision behavior and reject accepted QAQ claims unless the run is explicitly diagnostic.
- [x] Add E2E smoke tests for QAQ on-demand off/on with fake or tiny models.

## Tests and Quality Gates

- [x] Run `pytest -q tests/e2e/test_smoke_modes.py tests/regression/test_qaq_acceptance_guards.py` when implemented.
- [x] Verify routing summaries are present for both QAQ modes.
- [x] Verify loader summaries are present for `qaq_on_demand_on`.

## Done When

- [x] QAQ on-demand off and on complete for a fake or tiny smoke run.
- [x] Adaptive outputs are comparable with static baselines through the shared result reporter.
- [x] Adaptive runtime smoke and regression tests pass.

Final completion evidence:

- 2026-06-23T23:54:56+08:00: Shared result reporter comparability was still pending the Evaluation Metrics and Results Reporter module. Current adaptive outputs expose comparable metadata, routing summaries, loader summaries, latency, memory, and per-query traces.
- 2026-06-24T00:10:29+08:00: The shared result reporter now exists. The adaptive checklist remains unchecked pending a dedicated adaptive pass that verifies the final comparability item against reporter artifacts.
- 2026-06-24T03:00:00+08:00: Added an E2E reporter-matrix check that runs fake CPU `fp16`, `static_8bit`, `static_4bit`, `qaq_on_demand_off`, and `qaq_on_demand_on`, writes result artifacts, groups them through `qaq.report.build_report`, verifies no required comparison modes are missing, and confirms QAQ routing plus on-demand loader evidence are present. Verified with `python -m pytest -q tests/e2e/test_smoke_modes.py tests/regression/test_qaq_acceptance_guards.py` and `python -m pytest -q`. This remains diagnostic fake/CPU evidence, not paper-scale QAQ evidence.

- 2026-06-24T14:14:39+08:00: Added a real selected-weight execution path for Hugging Face-shaped models when the artifact index provides full per-tensor bit-plane coverage. Adaptive routing still uses the reference hidden-state pass, then executes each query under its router-selected reconstructed weights and uses those outputs for metrics. Verified with `python -m pytest -q tests/integration/test_mixed_weight_runtime.py tests/integration/test_model_adapter_smoke.py tests/integration/test_tensor_bitplane_artifacts.py tests/unit/test_results_schema.py tests/regression/test_qaq_acceptance_guards.py` and full-suite `python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m pytest -q`. This is real mechanism coverage on a tiny HF-shaped torch module, not accepted paper-scale LLaMA evidence.
