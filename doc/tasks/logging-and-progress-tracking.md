# Logging and Progress Tracking

## Goal

Implement a live console monitor and durable structured logs for training, inference, evaluation, loader activity, warnings, failures, checkpoints, and incomplete-run status.

## Inputs

- `doc/proposal.md`: Validation requires reproducible runs with logs, latency, memory, and reported loader overhead.
- `doc/high-level-design.md`: Logging provides terminal progress and durable logs without distorting timing-sensitive latency metrics.
- `doc/detailed-design.md`: Defines console monitor settings, log events, progress counters, incomplete markers, and required training/inference fields.
- `doc/test-plan.md`: Requires logging event formatting, integration logging/incomplete-run tests, E2E output validation, and performance checks that logging does not distort latency.

## Write Scope

Create or edit proposed paths: `qaq/logging.py`, `qaq/progress.py`, `qaq/status.py`, `tests/unit/test_logging_events.py`, `tests/integration/test_logging_and_incomplete_runs.py`, and logging fixtures.

## Read Scope

Inspect config logging settings, manifest status fields, router training progress needs, runtime/evaluation event shapes, loader events, and result reporter log path requirements.

## Dependencies

Experiment Configuration and Run Manifest for run IDs and status. All long-running modules emit events to this module.

## Tasks

- [x] Define structured log event fields for progress, metrics, checkpoint, loader, routing, warning, error, completion, and incomplete status.
- [x] Implement live console monitor updates for training fields: step/epoch, loss, learning rate when available, elapsed time, checkpoint events, warnings, and failure status.
- [x] Implement live console monitor updates for inference/evaluation fields: mode, benchmark progress, processed examples, elapsed time, latency, memory, routing summary, loader summary, warnings, and failure status.
- [x] Implement durable log writing and flushing with paths recorded in the run manifest and result artifacts.
- [x] Implement failure and interruption handling that writes incomplete markers when possible.
- [x] Add tests for event formatting, console-monitor state, durable log fields, and incomplete-run markers.

## Tests and Quality Gates

- [x] Run `pytest -q tests/unit/test_logging_events.py tests/integration/test_logging_and_incomplete_runs.py` when implemented.
- [x] Verify console progress is not the only durable record.
- [x] Verify latency-sensitive regions can be measured separately from progress display overhead where practical.

## Done When

- [x] Training and inference expose live console progress and persist the same essential status to durable logs.
- [x] Controlled failures leave incomplete markers.
- [x] Logging unit and integration tests pass.
