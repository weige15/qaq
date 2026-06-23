# Dynamic Loader and Memory Residency Manager

## Goal

Implement synchronous on-demand loading for selected bit-planes or precision artifacts, with explicit residency tracking, transfer timing, memory reporting, and clear failure behavior.

## Inputs

- `doc/proposal.md`: Stage 6 adds CPU-to-GPU selected bit-plane loading and reports memory reduction plus latency overhead.
- `doc/high-level-design.md`: Dynamic Loader owns GPU materialization and residency, exposing the expected lower-memory/higher-latency trade-off.
- `doc/detailed-design.md`: Defines Loader Event, residency map, synchronous loading algorithm, release policy openness, and loader failure behavior.
- `doc/test-plan.md`: Requires loader request validation tests, on-demand loader simulation, loader benchmark, memory benchmark, and missing-loader-summary regression tests.

## Write Scope

Create or edit proposed paths: `qaq/runtime/loader.py`, `qaq/loader.py`, `tests/unit/test_loader_validation.py`, `tests/integration/test_on_demand_loader_simulation.py`, and performance/loader fixtures.

## Read Scope

Inspect bit-plane artifact metadata, adaptive runtime loader request shape, logging APIs, evaluation reporter loader summary requirements, and selected tensor/CUDA runtime APIs.

## Dependencies

Quantization and Bit-Plane Store, Adaptive Inference Runtime, Logging and Progress Tracking. Requires approved tensor/CUDA runtime for real GPU transfer behavior.

## Tasks

- [x] Define loader request, loader event, residency map, and loader summary records.
- [x] Implement request validation for block ID, bit-width, CPU-resident artifact presence, and selected GPU/device.
- [x] Implement synchronous small-tensor CPU-to-device materialization with cache-hit, load, release, and failure events.
- [x] Record transfer timing, bytes where available, target device, residency state, and loader warnings.
- [x] Fail clearly on missing planes, invalid bit-widths, CUDA unavailable for real on-demand mode, and insufficient GPU memory.
- [x] Add unit and integration tests for loader validation, simulation, event logging, missing planes, and loader summary presence.

## Tests and Quality Gates

- [x] Run `pytest -q tests/unit/test_loader_validation.py tests/integration/test_on_demand_loader_simulation.py` when implemented.
- [x] Verify `qaq_on_demand_on` results include loader summaries.
- [x] Verify loader timing is reported separately from generic latency where practical.

## Done When

- [x] Simulated on-demand loading moves only requested planes and records loader events.
- [x] Invalid loader requests fail clearly.
- [x] Loader validation and simulation tests pass.
