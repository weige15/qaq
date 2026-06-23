# Repair Plan

## Repair Policy

Repairs must be minimal, targeted, and traceable to findings in `doc/review.md`. This review pass does not modify implementation code.

## Priority Order

1. Blockers
2. Major correctness issues
3. Error handling issues
4. Security issues
5. Test coverage gaps
6. Simple maintainability issues

## Fixes To Apply

* Review finding reference: Blocker, CUDA `qaq_on_demand_on` silently using CPU simulation.
  * Files likely affected: `qaq/runtime/adaptive.py`, `tests/integration/test_router_checkpoint_contract.py`, possibly `tests/e2e/test_smoke_modes.py`.
  * Minimal change required: derive the loader target device from `RunConfig.device` and `gpu_ids`, or explicitly reject CUDA on-demand runs before materialization until real CUDA loading exists.
  * Tests or commands to run after the fix: `python -m pytest -q tests/integration/test_router_checkpoint_contract.py tests/unit/test_loader_validation.py`, `python -m pytest -q`.
  * Risk of the fix: medium; it changes behavior for CUDA configs from misleading completion to explicit failure.
  * Rollback plan: revert the adaptive runtime target-device change and associated regression test.

* Review finding reference: Major, adaptive checkpoint candidate mismatch.
  * Files likely affected: `qaq/router/policy.py`, `qaq/runtime/adaptive.py`, `tests/unit/test_router_policy.py`, `tests/integration/test_router_checkpoint_contract.py`.
  * Minimal change required: pass active `RunConfig.precision_candidates` into router compatibility validation and fail on mismatches before building precision plans.
  * Tests or commands to run after the fix: `python -m pytest -q tests/unit/test_router_policy.py tests/integration/test_router_checkpoint_contract.py`, `python -m pytest -q`.
  * Risk of the fix: low; it tightens an intended compatibility check.
  * Rollback plan: revert the signature/argument change and the mismatch regression test.

* Review finding reference: Major, loader cache hits bypass artifact compatibility.
  * Files likely affected: `qaq/runtime/loader.py`, `tests/integration/test_on_demand_loader_simulation.py`.
  * Minimal change required: include request compatibility identity in the cache key or validate cached entries against `model_id`, `block_id`, `tensor_name`, and artifact metadata before returning cache hits.
  * Tests or commands to run after the fix: `python -m pytest -q tests/unit/test_loader_validation.py tests/integration/test_on_demand_loader_simulation.py`, `python -m pytest -q`.
  * Risk of the fix: medium; cache behavior and event counts may change.
  * Rollback plan: revert cache-key or cached-entry validation changes and test updates.

* Review finding reference: Major, router training manifest marks non-diagnostic training as diagnostic.
  * Files likely affected: `qaq/router/train.py`, `tests/integration/test_router_checkpoint_contract.py`, `doc/tasks/progress.md` if behavior documentation changes.
  * Minimal change required: preserve `config.diagnostic` when constructing the training run manifest, or add a separate training-mode field instead of forcing `router_diagnostic: True`.
  * Tests or commands to run after the fix: `python -m pytest -q tests/integration/test_router_checkpoint_contract.py`, `python -m qaq.router.train --config configs/router_train_real.yaml`, `python -m pytest -q`.
  * Risk of the fix: medium; existing tests may assert the old manifest shape indirectly.
  * Rollback plan: revert the manifest mapping change and test updates.

* Review finding reference: Major, adaptive runtime completed status can be mistaken for accepted adaptive QAQ.
  * Files likely affected: `qaq/runtime/adaptive.py`, `qaq/evaluate.py`, `tests/integration/test_router_checkpoint_contract.py`, `doc/tasks/progress.md`.
  * Minimal change required: label the current adaptive path as CPU-simulated/diagnostic in metadata and logs, or fail for non-diagnostic accepted runs until the Adaptive Inference Runtime task is complete.
  * Tests or commands to run after the fix: `python -m pytest -q tests/integration/test_router_checkpoint_contract.py tests/e2e/test_smoke_modes.py`, `python -m pytest -q`.
  * Risk of the fix: medium; downstream smoke expectations may need updated assertions.
  * Rollback plan: revert metadata/status guard changes and assertions.

* Review finding reference: Major, artifact/checkpoint exception-boundary leaks.
  * Files likely affected: `qaq/bitplanes.py`, `qaq/artifacts.py`, `qaq/router/checkpoint.py`, `qaq/router/types.py`, `qaq/router/train.py`, related unit/integration tests.
  * Minimal change required: catch and wrap `QuantizationError` as `BitPlaneError` at the artifact boundary; validate checkpoint parameter keys and values before constructing `RouterBlockParameters`.
  * Tests or commands to run after the fix: `python -m pytest -q tests/unit/test_bitplanes.py tests/unit/test_router_policy.py tests/integration/test_router_checkpoint_contract.py`, `python -m pytest -q`.
  * Risk of the fix: low; it should only stabilize invalid-input failures.
  * Rollback plan: revert wrapping/validation changes and new negative tests.

* Review finding reference: Major, output path escape for log and checkpoint directories.
  * Files likely affected: `qaq/config.py`, `qaq/router/train.py`, `tests/unit/test_config_validation.py`, `tests/integration/test_router_checkpoint_contract.py`.
  * Minimal change required: require `logging.log_dir` and router `checkpoint_dir` to resolve under `output_dir`, or apply equivalent overwrite/reuse validation to each caller-provided output root.
  * Tests or commands to run after the fix: `python -m pytest -q tests/unit/test_config_validation.py tests/integration/test_router_checkpoint_contract.py`, `python -m pytest -q`.
  * Risk of the fix: medium; configs using external log/checkpoint paths may start failing.
  * Rollback plan: revert path containment/reuse validation and tests.

* Review finding reference: Major, output-directory TOCTOU and non-atomic checkpoint writes.
  * Files likely affected: `qaq/config.py`, `qaq/manifest.py`, `qaq/router/checkpoint.py`, tests for manifest/checkpoint behavior.
  * Minimal change required: atomically claim new output directories when `overwrite=false`; write checkpoints and manifests to temporary files and replace atomically.
  * Tests or commands to run after the fix: targeted manifest/checkpoint tests plus `python -m pytest -q`.
  * Risk of the fix: medium; filesystem semantics differ by platform, though this repo targets local Unix-like development.
  * Rollback plan: revert atomic create/replace changes and tests.

* Review finding reference: Major, clean-checkout eval config depends on ignored checkpoint.
  * Files likely affected: `configs/router_eval_real.json`, docs, possibly tests.
  * Minimal change required: either document the train-then-eval dependency directly beside the config and avoid listing eval as standalone, or provide a fixture checkpoint committed under `tests/fixtures/` for self-contained eval smoke coverage.
  * Tests or commands to run after the fix: `python -m qaq.evaluate --config configs/router_eval_real.json --artifact-index configs/router_eval_real_artifacts.json --skip-output-dir-check --print-json` from a clean state if made standalone; otherwise run the documented train-then-eval sequence.
  * Risk of the fix: low to medium depending on whether a fixture checkpoint is introduced.
  * Rollback plan: revert config/docs/test changes.

* Review finding reference: Major, cwd-dependent tests.
  * Files likely affected: `tests/unit/test_config_validation.py`, `tests/integration/test_router_checkpoint_contract.py`, shared fixture helpers if added.
  * Minimal change required: resolve fixture/config paths relative to repository root or test file location rather than process cwd.
  * Tests or commands to run after the fix: `python -m pytest -q`, plus a targeted pytest invocation from `/tmp`.
  * Risk of the fix: low.
  * Rollback plan: revert path helper/test edits.

* Review finding reference: Minor, non-finite router-training hyperparameters.
  * Files likely affected: `qaq/router/train.py`, router training config tests.
  * Minimal change required: require `math.isfinite()` in the positive-float coercion helper.
  * Tests or commands to run after the fix: `python -m pytest -q tests/integration/test_router_checkpoint_contract.py`, `python -m pytest -q`.
  * Risk of the fix: low.
  * Rollback plan: revert helper and negative test.

* Review finding reference: Minor, stale append-only logs on overwrite.
  * Files likely affected: `qaq/logging.py`, `qaq/manifest.py`, router training tests.
  * Minimal change required: when `overwrite=true` starts a fresh run, truncate or rotate the run log, or generate a unique run id/log path.
  * Tests or commands to run after the fix: targeted logging tests plus `python -m pytest -q tests/integration/test_logging_and_incomplete_runs.py`.
  * Risk of the fix: medium; append-only semantics may be intentional for some logs.
  * Rollback plan: revert log open mode/path changes and tests.

## Fixes Explicitly Not Included

- Broadly splitting `qaq/router/train.py` into multiple modules.
- Implementing real CUDA on-demand loading.
- Completing the Adaptive Inference Runtime or Evaluation Metrics and Results Reporter tasks.
- Adding lint, format, or type-check tooling.
- Refactoring duplicated static/adaptive artifact materialization before the behavioral findings are fixed.

## Verification Commands

* `python -m pytest -q tests/unit`
* `python -m pytest -q tests/integration`
* `python -m pytest -q tests/e2e`
* `python -m pytest -q`
* `python -m qaq.config configs/smoke.json --skip-output-dir-check --print-json`
* `python -m qaq.evaluate --config configs/smoke.json --skip-output-dir-check --print-json`
* `python -m qaq.router.train --health-check`
* `python -m qaq.router.train --config configs/router_train_real.yaml`
* `python -m qaq.evaluate --config configs/router_eval_real.json --artifact-index configs/router_eval_real_artifacts.json --skip-output-dir-check --print-json`
