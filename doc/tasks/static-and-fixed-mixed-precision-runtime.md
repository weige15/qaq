# Static and Fixed Mixed-Precision Runtime

## Goal

Implement baseline and diagnostic runtimes for FP16, static 8-bit, static 4-bit, and fixed mixed-precision execution so QAQ results have valid comparison anchors.

## Inputs

- `doc/proposal.md`: Stages 1-3 require baseline harness, bit-plane proof, and fixed mixed precision before learned routing.
- `doc/high-level-design.md`: Static runtime provides comparison anchors and validates static-equivalent QAQ paths within tolerance.
- `doc/detailed-design.md`: Defines runtime modes, baseline outputs, latency/memory events, fixed profile behavior, and baseline failure rules.
- `doc/test-plan.md`: Requires smoke E2E, static-equivalent profile tests, performance smoke checks, and regression tests for missing static baselines.

## Write Scope

Create or edit proposed paths: `qaq/runtime/static.py`, `qaq/runtime/common.py`, `qaq/evaluate.py`, `tests/integration/test_static_equivalent_profiles.py`, `tests/e2e/test_smoke_modes.py`, and smoke configs.

## Read Scope

Inspect config, model adapter, block registry, bit-plane store, evaluation reporter, and logging interfaces.

## Dependencies

Experiment Configuration and Run Manifest, Model and Benchmark Adapter, Block Registry and Precision Plan, Quantization and Bit-Plane Store, Evaluation Metrics and Results Reporter, Logging and Progress Tracking.

## Tasks

- [x] Implement FP16 reference execution path that emits raw outputs, timing events, memory events, and runtime status.
- [x] Implement static 8-bit and static 4-bit execution paths using bit-plane reconstruction or approved static quantized artifacts.
- [x] Implement `fixed_mixed` execution from a configured per-block profile without learned routing.
- [x] Record consistent latency and memory measurement events across modes with warm-up/cache policy metadata.
- [x] Fail clearly on missing static artifacts, missing block decisions, OOM, or non-comparable tokenizer/dataset/prompt settings.
- [x] Add smoke and static-equivalent integration tests for all-4-bit and all-8-bit profiles.

## Tests and Quality Gates

- [x] Run `pytest -q tests/integration/test_static_equivalent_profiles.py tests/e2e/test_smoke_modes.py` when implemented.
- [x] Verify one fake or tiny prompt completes in `fp16`, `static_8bit`, `static_4bit`, and `fixed_mixed`.
- [x] Verify missing static baselines prevent QAQ comparison acceptance.

## Done When

- [x] Baseline modes emit raw outputs, latency, memory, logs, and result-ready metadata.
- [x] All-4-bit and all-8-bit fixed profiles match corresponding static paths within approved tolerance.
- [x] Static runtime integration tests pass.
