# Code Review

## Scope Reviewed

- Compared the current working tree against `main`. The checkout is currently on `main`, so the effective PR scope is the modified `.gitignore` plus untracked `AGENTS.md`, `configs/`, `doc/`, `pyproject.toml`, `qaq/`, and `tests/`.
- Read repository intent from `AGENTS.md`, `doc/high-level-design.md`, `doc/detailed-design.md`, `doc/requirements.md`, `doc/test-plan.md`, and `doc/tasks/progress.md`.
- Spawned one parallel review agent for each requested point: security, code quality, bugs, race, test flakiness, and maintainability.
- Locally verified the full test suite and smoke commands:
  - `python -m pytest -q`
  - `python -m qaq.config configs/smoke.json --skip-output-dir-check --print-json`
  - `python -m qaq.evaluate --config configs/smoke.json --skip-output-dir-check --print-json`
  - `python -m qaq.router.train --health-check`

## Correctness

- Blocker: `qaq_on_demand_on` can silently run CPU simulation when the config requests CUDA. `qaq/runtime/adaptive.py` hard-codes `LoaderRequest(target_device="cpu")` in on-demand mode, bypassing the CUDA-unavailable guard in `qaq/runtime/loader.py`. This violates the project rule that unsupported CUDA behavior must fail clearly rather than fall back to fake behavior.
  - Evidence: `qaq/runtime/adaptive.py:190`, `qaq/runtime/loader.py:391`, `qaq/config.py:258`.
- Major: adaptive routing validates a checkpoint against its own precision candidates instead of the active run config. `route_hidden_states()` passes `checkpoint.metadata.candidate_bit_widths` into `validate_checkpoint_compatibility()`, so a config with different candidates can still run while metadata and precision plans disagree.
  - Evidence: `qaq/router/policy.py:36`, `qaq/router/checkpoint.py:177`, `qaq/runtime/adaptive.py:102`, `qaq/router/policy.py:84`.
- Major: loader cache hits bypass artifact compatibility checks. The cache is checked before reconstructing and validating the artifact against the request, and the cache key omits `model_id` and tensor identity.
  - Evidence: `qaq/runtime/loader.py:197`, `qaq/runtime/loader.py:443`.
- Major: non-diagnostic router training is written to the run manifest as diagnostic. `RouterTrainingConfig.to_run_config()` always sets `router_diagnostic: True`, while checkpoint metadata records `diagnostic_training: config.diagnostic`.
  - Evidence: `qaq/router/train.py:268`, `qaq/router/train.py:439`, `qaq/router/train.py:1177`.
- Major: the adaptive runtime returns `status="completed"` for the current CPU implementation while project progress still marks Adaptive Inference Runtime and Evaluation Metrics/Reporter incomplete. This can blur checkpoint-load smoke evidence with accepted adaptive QAQ behavior.
  - Evidence: `qaq/runtime/adaptive.py:25`, `qaq/runtime/adaptive.py:117`, `doc/tasks/progress.md:24`, `doc/tasks/progress.md:27`, `tests/integration/test_router_checkpoint_contract.py:278`.

## Edge Cases

- Major: malformed bit-plane artifact metadata can escape the artifact API with `QuantizationError` instead of the expected `BitPlaneError`, so router training does not consistently translate invalid artifacts into `student_artifact_invalid`.
  - Evidence: `qaq/bitplanes.py:84`, `qaq/artifacts.py:36`, `qaq/router/train.py:795`.
- Major: malformed router checkpoints can raise `AttributeError` instead of `RouterPolicyError` because parameter values are assumed to be mappings before `RouterBlockParameters.from_mapping()` calls `.get()`. Non-string parameter keys are also silently skipped.
  - Evidence: `qaq/router/checkpoint.py:53`, `qaq/router/checkpoint.py:56`, `qaq/router/types.py:97`.
- Minor: router training accepts non-finite positive-float hyperparameters for values such as `learning_rate`, `temperature`, and `target_temperature`.
  - Evidence: `qaq/router/train.py:71`, `qaq/router/train.py:1365`, `qaq/router/train.py:1387`.

## Error Handling

- Major: the checked-in adaptive eval config depends on an ignored generated checkpoint under `runs/`, so a clean checkout cannot run that eval command without first running router training.
  - Evidence: `configs/router_eval_real.json:20`, `.gitignore:11`, `doc/tasks/progress.md:135`.
- Major: the exception-style mismatches above weaken CLI failure stability for malformed artifacts and checkpoints.

## Concurrency

- Major: `overwrite=false` output-directory protection has a time-of-check/time-of-use gap. Config validation checks `output_dir.exists()` before work starts, but manifest creation later uses `mkdir(..., exist_ok=True)` and writes deterministic artifact names. Two concurrent runs can both pass validation and then share or clobber manifest, log, and checkpoint files.
  - Evidence: `qaq/config.py:219`, `qaq/manifest.py:86`, `qaq/manifest.py:137`, `qaq/router/train.py:439`, `qaq/logging.py:184`, `qaq/router/train.py:1191`.
- Major: checkpoint publication is non-atomic. Saving truncates and writes the final checkpoint path directly, while evaluation can load that same path directly.
  - Evidence: `qaq/router/checkpoint.py:203`, `qaq/router/checkpoint.py:210`, `qaq/runtime/adaptive.py:54`, `configs/router_eval_real.json:20`.
- Minor: repeated `overwrite=true` training runs append to the same JSONL log while overwriting the manifest and checkpoint, making logs dependent on prior runs.
  - Evidence: `configs/router_train_real.yaml:16`, `configs/router_train_real.yaml:34`, `qaq/logging.py:152`, `qaq/manifest.py:89`, `qaq/router/checkpoint.py:204`.

## Performance

- No concrete performance regression was found in the changed scaffold. The reviewed code is intentionally CPU/fake-path heavy and should not be treated as paper-scale QAQ performance evidence.

## Security

- Major: `logging.log_dir` and router `checkpoint_dir` can point outside `output_dir` and bypass the output reuse policy. Those paths are later used to create directories, append logs, and write checkpoints.
  - Evidence: `qaq/config.py:76`, `qaq/config.py:219`, `qaq/router/train.py:227`, `qaq/logging.py:149`, `qaq/logging.py:184`, `qaq/router/train.py:265`, `qaq/router/train.py:1191`, `qaq/router/checkpoint.py:202`.

## Readability

- Minor: `qaq/router/train.py` currently combines config schema, YAML parsing, validation, artifact indexing, training math, checkpointing, health artifact generation, and CLI handling in one large module. This raises the cost of safely replacing the router objective or real model path later.
  - Evidence: `qaq/router/train.py:112`, `qaq/router/train.py:322`, `qaq/router/train.py:430`, `qaq/router/train.py:777`, `qaq/router/train.py:1253`, `qaq/router/train.py:1430`.
- Minor: artifact-ref normalization and reconstruction-record creation are duplicated between static and adaptive runtimes.
  - Evidence: `qaq/runtime/static.py:176`, `qaq/runtime/static.py:202`, `qaq/runtime/adaptive.py:151`, `qaq/runtime/adaptive.py:182`.

## Test Coverage

- Major: several tests are current-working-directory dependent because they open repository fixtures through relative paths. One reviewer confirmed that running a targeted test from `/tmp` fails with `FileNotFoundError`.
  - Evidence: `tests/unit/test_config_validation.py:99`, `tests/unit/test_config_validation.py:114`, `tests/unit/test_config_validation.py:126`, `tests/integration/test_router_checkpoint_contract.py:80`, `tests/integration/test_router_checkpoint_contract.py:305`.
- Major: there is no regression coverage for CUDA on-demand mode failing clearly, adaptive checkpoint candidate mismatch, loader cache compatibility, output path containment, atomic checkpoint publication, or the router-training diagnostic manifest contradiction.

## Unnecessary Complexity

- Minor: the large router training module and duplicated artifact materialization logic are the main simplification candidates. These should not be broad refactors in the immediate repair pass; targeted fixes should come first.

## Review Summary

- Blocker: 1
  - CUDA `qaq_on_demand_on` silently using CPU simulation.
- Major: 14
  - Checkpoint candidate mismatch, loader cache compatibility bypass, diagnostic manifest contradiction, adaptive runtime accepted-looking status, artifact/checkpoint exception leaks, clean-checkout eval dependency, output-dir TOCTOU race, non-atomic checkpoint writes, output path escape, cwd-dependent tests, and missing regression coverage.
- Minor: 5
  - Non-finite hyperparameter acceptance, stale append-only logs on overwrite, router training module size, duplicated artifact materialization logic, and related simplification risk.
- Nit: 0

## Do Not Fix Yet

- Do not claim adaptive inference, evaluation reporting, CUDA loading, or paper-scale QAQ evidence complete as part of a review-only pass.
- Do not split `qaq/router/train.py` broadly before the correctness, safety, and test gaps above are fixed.
- Do not convert fake CPU simulation into real CUDA behavior without an explicit implementation task and targeted tests.
