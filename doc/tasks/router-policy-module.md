# Router Policy Module

## Goal

Implement lightweight router inference that converts per-block hidden features into valid, deterministic, traceable precision decisions for QAQ adaptive modes.

## Inputs

- `doc/proposal.md`: Router should consume block/query-dependent hidden representations and produce probabilities over candidate bit-widths.
- `doc/high-level-design.md`: Router Policy is distinct from Router Training and owns probability and decision semantics.
- `doc/detailed-design.md`: Defines Router Trace, checkpoint compatibility, score normalization, deterministic tie-breaking, and constant-precision rejection behavior.
- `doc/test-plan.md`: Requires router probability unit tests, checkpoint contract tests, golden router decisions, property tests, and routing variation checks.

## Write Scope

Create or edit proposed paths: `qaq/router/policy.py`, `qaq/router/checkpoint.py`, `qaq/router/types.py`, `tests/unit/test_router_policy.py`, `tests/integration/test_router_checkpoint_contract.py`, and `tests/golden/`.

## Read Scope

Inspect Model and Benchmark Adapter hidden-feature outputs, Block Registry IDs, Precision Plan fields, Logging event contracts, and Router Training checkpoint metadata.

## Dependencies

Experiment Configuration and Run Manifest, Model and Benchmark Adapter, Block Registry and Precision Plan, Logging and Progress Tracking. Coordinate checkpoint metadata with Router Training Pipeline.

## Tasks

- [x] Define router checkpoint metadata for model identity, block IDs, candidate bit-widths, feature source, temperature, and router parameters.
- [x] Implement router scoring over fake or tiny hidden features with a lightweight MLP-compatible interface.
- [x] Implement probability normalization over candidate bit-widths with finite-value checks and sum-to-one validation.
- [x] Implement deterministic decision conversion and tie-breaking for accepted evaluation runs.
- [x] Emit router traces with query ID, block ID, raw scores, probabilities, selected bit-width, temperature, and checkpoint ID.
- [x] Add tests for probability validity, temperature behavior, deterministic ties, checkpoint mismatch failure, and constant precision flagging.

## Tests and Quality Gates

- [x] Run `pytest -q tests/unit/test_router_policy.py tests/integration/test_router_checkpoint_contract.py` when implemented.
- [x] Verify outputs exist for every configured block.
- [x] Verify constant global precision is flagged unless explicitly diagnostic.

## Done When

- [x] Router policy produces valid per-block precision decisions from fixture hidden states.
- [x] Checkpoint metadata mismatches fail clearly.
- [x] Router policy and checkpoint contract tests pass.
