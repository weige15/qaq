# Router Training Pipeline

## Goal

Implement router training orchestration that freezes base LLM parameters, uses teacher/student signals, records the approved distillation objective, and writes reusable router checkpoints with live console progress and durable logs.

## Inputs

- `doc/proposal.md`: Stage 4 trains a lightweight router using full-precision teacher and quantized student signals.
- `doc/high-level-design.md`: Router training depends on model adapter, bit-plane store, router policy, and logging; it must fail if no concrete training method is selected.
- `doc/detailed-design.md`: Defines router training config, loss record, checkpoint metadata, frozen-base requirement, and incomplete-run behavior.
- `doc/test-plan.md`: Requires router checkpoint contract tests, logging/incomplete-run tests, performance checks, and manual verification.

## Write Scope

Create or edit proposed paths: `qaq/router/train.py`, `qaq/router/losses.py`, `qaq/router/checkpoint.py`, `configs/router_train_smoke.yaml`, `tests/integration/test_router_checkpoint_contract.py`, and `tests/integration/test_logging_and_incomplete_runs.py`.

## Read Scope

Inspect config schema, model adapter teacher/student outputs, bit-plane artifacts, router policy APIs, and logging/progress APIs.

## Dependencies

Experiment Configuration and Run Manifest, Model and Benchmark Adapter, Quantization and Bit-Plane Store, Router Policy Module, Logging and Progress Tracking. Requires an approved router-training loss before accepted QAQ claims.

## Tasks

- [x] Add router training config fields for data source, teacher path, student quantized path, router hyperparameters, checkpoint interval, and logging settings.
- [x] Implement preflight validation that fails if no concrete distillation loss or training data is configured.
- [x] Implement training loop scaffolding that freezes base model parameters and updates only router parameters.
- [x] Implement file-backed non-diagnostic training data loading for the local real-data acceptance path.
- [x] Implement `router_cost_cross_entropy` with cost-derived targets from teacher/student signals, bit-plane reconstruction distortion, and a documented bit-cost assumption.
- [x] Write a target audit artifact that records sample counts, target records, candidate costs, target probabilities, objective name, and diagnostic status.
- [x] Keep the fake/smoke path under the explicit `python -m qaq.router.train --health-check` diagnostic command instead of using it as the acceptance gate.
- [x] Record loss, step/epoch, learning rate if available, elapsed time, checkpoint events, warnings, and failure status through logging.
- [x] Save router checkpoints with metadata required by Router Policy Module and Evaluation Reporter.
- [x] Add tiny/fake training tests for checkpoint save/load and incomplete-run marker behavior.
- [x] Add real-data acceptance coverage proving file-backed samples, target generation, objective loss, parameter updates, checkpoint reload, checkpoint-loaded evaluation, and validation metrics.

## Tests and Quality Gates

- [x] Run `pytest -q tests/integration/test_router_checkpoint_contract.py tests/integration/test_logging_and_incomplete_runs.py` when implemented.
- [x] Verify base model parameters are not trainable where detectable.
- [x] Verify a missing training method fails before long-running work begins.
- [x] Run `python -m qaq.router.train --config configs/router_train_real.yaml` as the non-diagnostic local acceptance command.
- [x] Run `python -m qaq.evaluate --config configs/router_eval_real.json --artifact-index configs/router_eval_real_artifacts.json --skip-output-dir-check --print-json` to verify checkpoint reload in evaluation.
- [x] Run `python -m qaq.router.train --health-check` only as a quick diagnostic health check.

## Done When

- [x] A file-backed non-diagnostic local training run writes checkpoint metadata, training metrics, target audit data, durable logs, and completion status.
- [x] The real objective emits validation metrics and records a nonzero router parameter update.
- [x] The saved checkpoint reloads and drives checkpoint-loaded evaluation.
- [x] Controlled failure writes incomplete markers.
- [x] Router training integration tests pass.
