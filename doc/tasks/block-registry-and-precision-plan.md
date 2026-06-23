# Block Registry and Precision Plan

## Goal

Implement stable QAQ block discovery and precision-plan validation so MHA/FFN block IDs join model metadata, bit-plane artifacts, router traces, loader events, and results.

## Inputs

- `doc/proposal.md`: Block abstraction should align with MHA and FFN blocks from the paper figure and validate mixed-precision execution before routing.
- `doc/high-level-design.md`: Block Registry owns block IDs, granularity metadata, precision validation rules, and profile validation.
- `doc/detailed-design.md`: Defines Block Descriptor, Precision Plan, discovery contract, mode-specific precision plans, and unsupported-layout failures.
- `doc/test-plan.md`: Requires `tests/unit/test_block_registry.py`, static-equivalent integration tests, and regression tests for invalid global precision behavior.

## Write Scope

Create or edit proposed paths: `qaq/blocks.py`, `qaq/precision_plan.py`, `tests/unit/test_block_registry.py`, `tests/fixtures/fake_transformer.py`, and related integration fixtures.

## Read Scope

Inspect Model and Benchmark Adapter metadata, Quantization and Bit-Plane Store artifact metadata, Router Policy output contracts, and runtime mode definitions.

## Dependencies

Experiment Configuration and Run Manifest. Model and Benchmark Adapter for architecture metadata. Coordinate with Quantization and Bit-Plane Store and Router Policy Module.

## Tasks

- [x] Define `BlockDescriptor` fields for stable block ID, layer index, block type, source module path, tensor references, supported bit-widths, artifact references, and validation status.
- [x] Implement discovery for fake transformer MHA/FFN blocks, preserving model order and stable IDs across repeated runs.
- [x] Implement mode-specific precision-plan construction for `fp16`, `static_8bit`, `static_4bit`, `fixed_mixed`, and QAQ modes.
- [x] Validate every quantized block has a supported precision decision and artifact availability before runtime execution.
- [x] Reject unsupported layouts, missing decisions, invalid bit-widths, and silent full-precision fallback in quantized modes.
- [x] Add unit tests for stable IDs, unsupported layout failure, repeated discovery, and invalid precision decisions.

## Tests and Quality Gates

- [x] Run `pytest -q tests/unit/test_block_registry.py` when implemented.
- [x] Verify all-4-bit and all-8-bit precision plans are usable by static-equivalent integration tests.
- [x] Verify MHA/FFN remains the primary granularity unless an approved fallback is documented.

## Done When

- [x] Fake transformer discovery produces stable MHA/FFN block IDs.
- [x] Precision plans validate all required runtime modes.
- [x] Block registry unit tests pass.
