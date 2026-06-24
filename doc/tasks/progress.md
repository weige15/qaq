# Task Progress

## Scaffold Status

- [x] Project Scaffold and Shared Test Harness
  - 2026-06-23T18:05:41+08:00: Added minimal Python package scaffold, pytest configuration, config/test fixture directories, Python ignore rules, and import smoke test. Verified with `python -m pytest -q`.

## Module Status

- [x] Experiment Configuration and Run Manifest (`doc/tasks/experiment-configuration-and-run-manifest.md`)
  - 2026-06-23T18:13:18+08:00: Implemented and verified config validation, manifest lifecycle updates, JSON/TOML config loading, `use_model_tokenizer` fallback, and `python -m qaq.config` validation.
- [x] Model and Benchmark Adapter (`doc/tasks/model-and-benchmark-adapter.md`)
  - Diagnostic fake/local adapter paths remain regression coverage only.
  - Optional Hugging Face/LLaMA metadata, tokenizer, reference-forward, and weight-load support exists with explicit fake/tiny/real-subset evidence labels.
  - Real local `meta-llama/Llama-3.1-8B` adapter/tokenizer verification and real HellaSwag tokenization evidence are recorded; full benchmark acceptance remains a separate reporter/evaluation milestone.
  - Fake/tiny tests are diagnostic or tiny-mechanism evidence only and cannot close benchmark, quantized runtime, or paper-scale claims.
  - 2026-06-25T04:05:18+08:00: Added adapter-level provenance metadata to reference outputs: adapter kind, model source, fake model/tokenizer flags, fake dataset flag, fixture-only flag, real-benchmark flag, diagnostic flag, selected GPU IDs, dataset sources, and context-length policy. Tests now prove fake-smoke outputs remain diagnostic, local-root HellaSwag rows are recognized as real benchmark data while still diagnostic under a fake model, and injected TinyHF/mocked Hugging Face objects cannot satisfy real-adapter evidence. Verified with `python -m py_compile qaq/model_adapter.py tests/integration/test_model_adapter_smoke.py`, `python -m pytest -q tests/integration/test_model_adapter_smoke.py`, `python -m pytest -q tests/integration/test_static_equivalent_profiles.py`, and `python -m pytest -q tests/unit/test_config_validation.py tests/integration/test_model_adapter_smoke.py tests/integration/test_static_equivalent_profiles.py`. Real local LLaMA snapshot verification and lab-server GPU loading remain incomplete.
  - 2026-06-25T05:30:00+08:00: Repaired the acceptance guidance and result contract so fake/tiny/smoke evidence cannot close this module: result artifacts now track `tokenizer_is_fake`, fake tokenizer IDs reject acceptance, malformed GPU selector records reject large-model acceptance, and fake/tiny/model-adapter smoke tests are classified as diagnostic or tiny-mechanism evidence only. The module remains partial until actual cached `meta-llama/Llama-3.1-8B` adapter verification and real-subset benchmark/tokenization verification are recorded.
  - 2026-06-25T05:31:00+08:00: Completed the first-milestone Model and Benchmark Adapter verification. `python -m qaq.model_adapter --config configs/benchmarks/llama_first_milestone/hellaswag/fp16.json --limit 8 --output runs/model_adapter/llama31_hellaswag_real_subset_adapter.json --print-json` resolved the actual cached `meta-llama/Llama-3.1-8B` snapshot `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`, used the actual local tokenizer, tokenized 8 real HellaSwag validation rows from `/nfs/home/s314511048/qaq_benchmarks/hellaswag/validation.jsonl`, and recorded `evidence_level: real_subset_path`, `diagnostic: false`, `dataset_is_fake: false`, `model_is_fake: false`, and `tokenizer_is_fake: false`. `python scripts/gpu_run.py --count 1 --min-free-mb 18000 --status-file runs/gpu-selector/model-adapter-load-weights.json -- python -m qaq.model_adapter --config configs/benchmarks/llama_first_milestone/hellaswag/fp16.json --limit 1 --load-weights --output runs/model_adapter/llama31_hellaswag_weight_load.json --print-json` selected physical RTX 3090 GPU 0, loaded the local LLaMA weights, embedded the GPU selector record, and recorded `weights_loaded: true`. Verified with `python -m py_compile qaq/model_adapter.py tests/integration/test_model_adapter_smoke.py`, `python -m pytest -q tests/integration/test_model_adapter_smoke.py` (11 passed), `python -m pytest -q tests/unit/test_results_schema.py tests/regression/test_qaq_acceptance_guards.py` (35 passed, 1 skipped), `python -m pytest -q tests/unit/test_config_validation.py` (30 passed), `python -m pytest -q tests/integration/test_static_equivalent_profiles.py` (9 passed), and `python -m pytest -q` (165 passed, 1 skipped). Additional real forward evidence: after an initial `unsafe_output_reuse` preflight failure without `--skip-output-dir-check`, `python scripts/gpu_run.py --count 1 --min-free-mb 18000 --status-file runs/gpu-selector/hellaswag-fp16-subset.json -- python -m qaq.evaluate --config configs/benchmarks/llama_first_milestone/hellaswag/fp16.json --skip-output-dir-check --max-examples 8 --eval-batch-size 1 --hf-device-map auto --result-output runs/llama_first_milestone/hellaswag/fp16_subset/result_artifact.json --print-result-json` selected physical RTX 3090 GPU 0, loaded the real local LLaMA weights, ran 8 real HellaSwag FP16 examples, and wrote a result artifact with `completion_status: completed`, `diagnostic: false`, `dataset_is_fake: false`, `model_is_fake: false`, `tokenizer_is_fake: false`, `processed_examples: 8`, `total_examples: 10042`, `peak_gpu_memory_gb: 15.032358169555664`, `latency_seconds: 18.9547351768706`, and rejection reason `benchmark_subset_not_full_acceptance`. This is real-subset adapter/FP16 evidence, not accepted benchmark or paper-scale QAQ evidence.
- [x] Block Registry and Precision Plan (`doc/tasks/block-registry-and-precision-plan.md`)
  - 2026-06-23T18:20:27+08:00: Implemented fake-transformer MHA/FFN block discovery, stable block descriptors, static/fixed/QAQ precision-plan validation, artifact availability checks, and block registry tests. Verified with `python -m pytest -q tests/unit/test_block_registry.py` and `python -m pytest -q`.
- [x] Quantization and Bit-Plane Store (`doc/tasks/quantization-and-bit-plane-store.md`)
  - 2026-06-23T18:33:53+08:00: Implemented dependency-free per-tensor affine unsigned quantization, uint identity fixture support, MSB bit-plane decomposition/reconstruction, bit-plane artifact metadata/checksum validation, JSON save/load roundtrip, golden 4-bit/8-bit reconstruction fixtures, and model/block/tensor compatibility checks. Verified with `python -m pytest -q tests/unit/test_bitplanes.py tests/integration/test_quantized_artifact_roundtrip.py` and `python -m pytest -q`.
  - 2026-06-24T04:22:00+08:00: Added tensor-native safetensors-backed bit-plane artifacts with packed `torch.uint8` quantized values, QAQ metadata in safetensors metadata, reconstruction/distortion helpers, and selected-plane packing for CPU/CUDA loader materialization. Verified with `python -m pytest -q tests/integration/test_tensor_bitplane_artifacts.py tests/integration/test_llama_bitplane_generation.py tests/integration/test_router_checkpoint_contract.py`, `python -m qaq.llama_bitplanes --model meta-llama/Llama-3.1-8B --artifact-format safetensors --output-dir runs/llama31_8b_native_bitplanes_probe --block-limit 1 --tensor-limit-per-block 2 --max-elements-per-tensor 16 --overwrite --print-json`, and the full single-tensor command that wrote a non-truncated 16,777,216-element Llama q-projection artifact.
  - 2026-06-24T19:01:01+08:00: Repaired and verified the LLaMA bit-plane runtime-index contract for the per-tensor weight-override runtime path. `runtime_artifact_index.json` now remains tensor-name keyed, full-runtime requests can fail with `incomplete_tensor_artifact_index`, and manifests distinguish `partial_tensor_index`, sampled/truncated diagnostic probes, full tensor-native runtime artifacts, and accepted full quantized inference artifacts. Verified with `python -m pytest -q tests/integration/test_llama_bitplane_generation.py tests/integration/test_mixed_weight_runtime.py tests/unit/test_results_schema.py` and `python -m pytest -q tests/unit tests/integration`. This is artifact/runtime-contract evidence on tiny local fixtures, not accepted LLaMA benchmark evidence.
- [x] Static and Fixed Mixed-Precision Runtime (`doc/tasks/static-and-fixed-mixed-precision-runtime.md`)
  - 2026-06-23T18:41:44+08:00: Implemented fake CPU static runtime outputs for `fp16`, `static_8bit`, `static_4bit`, and diagnostic `fixed_mixed`, with precision plans, bit-plane reconstruction records, latency/memory events, result-ready metadata, log-event records, static-baseline acceptance guard, and a minimal `python -m qaq.evaluate` entry point. Verified with `python -m pytest -q tests/integration/test_static_equivalent_profiles.py tests/e2e/test_smoke_modes.py`, `python -m pytest -q`, and `python -m qaq.evaluate --config configs/smoke.json --skip-output-dir-check --print-json`.
  - 2026-06-24T14:14:39+08:00: Added real Hugging Face weight-override execution for static quantized modes when the artifact index provides full per-tensor bit-plane coverage for every controlled block. Static outputs now record whether mixed-precision weights were actually applied and keep reconstructed-only legacy artifact indexes out of accepted-style evidence. Verified with focused mixed-weight/runtime tests and the full GPU-wrapped suite.
- [x] Router Policy Module (`doc/tasks/router-policy-module.md`)
  - 2026-06-23T19:10:17+08:00: Implemented dependency-free router checkpoint metadata, JSON checkpoint save/load, per-block linear scoring over fake hidden features, finite softmax normalization, deterministic argmax tie-breaking with lowest-bit fallback, router traces, per-query precision plans, and constant-global-precision summary flagging. Verified with `python -m pytest -q tests/unit/test_router_policy.py tests/integration/test_router_checkpoint_contract.py` and `python -m pytest -q`.
- [x] Router Training Pipeline (`doc/tasks/router-training-pipeline.md`)
  - 2026-06-23T23:08:25+08:00: Replaced diagnostic-only router training with file-backed real-data loading, compatible bit-plane artifact validation, `router_cost_cross_entropy`, cost-derived targets, router-softmax loss, router-parameter gradient updates, validation metrics, checkpoint metadata, checkpoint-loaded evaluation, YAML/JSON/TOML config loading, separate diagnostic health checks, and controlled-failure incomplete markers. Verified with `python -m qaq.router.train --config configs/router_train_real.yaml`, `python -m qaq.evaluate --config configs/router_eval_real.json --artifact-index configs/router_eval_real_artifacts.json --skip-output-dir-check --print-json`, `python -m pytest -q tests/integration/test_router_checkpoint_contract.py tests/integration/test_logging_and_incomplete_runs.py`, and `python -m pytest -q`.
  - 2026-06-24T00:26:12+08:00: Re-verified the stricter real-data acceptance gate after implementation permission. `python -m qaq.router.train --config configs/router_train_real.yaml` completed with 3 file-backed training samples, 12 target records, objective `router_cost_cross_entropy`, checkpoint `runs/router_train_real/checkpoints/router_step_0003.json`, target audit `runs/router_train_real/router_targets.json`, validation loss `0.693113872343573`, and nonzero parameter update metadata. `python -m qaq.evaluate --config configs/router_eval_real.json --artifact-index configs/router_eval_real_artifacts.json --skip-output-dir-check --print-json` reloaded that checkpoint and emitted 8 non-diagnostic routing decisions over 2 validation samples. `python -m qaq.router.train --health-check` remains diagnostic only. Targeted integration tests and the full suite passed.
  - 2026-06-24T01:16:01+08:00: Added optional Hugging Face Llama adapter support for the base `meta-llama/Llama-3.1-8B` target. The adapter now resolves cached Hugging Face IDs to local snapshots, reads Llama metadata without network access, exposes 64 MHA/FFN controlled blocks with Llama tensor names, and can lazily load the real model for reference forwards when matching bit-plane artifacts are available. Verified the cached base model metadata probe with 64 blocks and hidden size 4096. Re-ran `python -m qaq.router.train --config configs/router_train_real.yaml`, which still completed with 3 samples, 12 target records, objective `router_cost_cross_entropy`, checkpoint `runs/router_train_real/checkpoints/router_step_0003.json`, and validation loss `0.693113872343573`. Full Llama router training remains blocked on Llama-compatible bit-plane artifacts for every controlled block and a GPU-backed teacher/student run.
  - 2026-06-24T03:03:14+08:00: Added `python -m qaq.prepare_bitplanes` for local Hugging Face safetensors. The command discovers Llama MHA/FFN blocks from real model metadata, reads sampled real BF16 weight values from the cached base `meta-llama/Llama-3.1-8B` safetensors shards, writes trainer-compatible QAQ bit-plane artifacts, and emits an absolute-path `artifact_index.json`. Verified `python -m qaq.prepare_bitplanes --model meta-llama/Llama-3.1-8B --output-dir runs/llama31_8b_bitplanes_sampled --sample-values 16 --overwrite --print-json` with 64 artifacts and then validated the generated artifact index through `validate_router_training_preflight` for `router_cost_cross_entropy`. Also hardened `python -m qaq.llama_bitplanes` truncated generation to slice local safetensors instead of loading whole tensors before truncation, and verified `python -m qaq.llama_bitplanes --model meta-llama/Llama-3.1-8B --output-dir runs/llama31_8b_bitplanes_probe --block-limit 1 --tensor-limit-per-block 2 --max-elements-per-tensor 16 --overwrite --print-json` with 2 BF16 real-weight artifacts. Full Llama router training remains blocked on the GPU-backed teacher/student forward run, and these sampled artifacts are not full quantized inference artifacts.
  - 2026-06-24T03:35:32+08:00: Added CUDA capacity preflight for router training before Hugging Face model weights are loaded. The preflight estimates teacher/student model-weight bytes from local safetensors metadata, probes visible CUDA memory, and fails with `insufficient_cuda_memory` when the current separate teacher+student adapter path cannot fit before activations. Added `configs/router_train_llama31_8b_sampled.yaml` as the reproducible one-sample Llama sampled-artifact training command. Verified the config under escalated GPU access with `python -m qaq.router.train --config configs/router_train_llama31_8b_sampled.yaml`; it correctly failed before model loading because the visible RTX 4050 has 4.96 GiB free of 6.00 GiB total while the current path requires at least 30.42 GiB free. This is not a successful Llama training run; full Llama training remains blocked on sufficient GPU memory or a real shared/sequential teacher-student execution path.
  - 2026-06-24T04:22:00+08:00: Extended router training artifact loading to accept tensor-native `.qaq.safetensors` student artifacts from directories, indexes, or direct artifact paths. The real objective still computes `router_cost_cross_entropy` from real samples, hidden features, bit-cost penalties, and artifact reconstruction distortion; native artifacts only replace the storage/distortion read path. Verified a one-step integration run using native student artifacts with real file-backed samples, target production, nonzero loss, parameter update, validation loss, and reloadable checkpoint metadata. Re-ran `python -m qaq.router.train --config configs/router_train_real.yaml`, which completed with 3 training samples, 12 target records, objective `router_cost_cross_entropy`, checkpoint `runs/router_train_real/checkpoints/router_step_0003.json`, and validation loss `0.693113872343573`.
  - 2026-06-24T03:56:30+08:00: Replaced the same-model teacher/student duplicate reference-loading path with a shared frozen reference adapter and shared reference-forward output for the current distortion-based `router_cost_cross_entropy` objective. Checkpoint metadata and `router_targets.json` now record `shared_teacher_student_reference`. Verified with `python -m pytest -q tests/integration/test_router_checkpoint_contract.py tests/integration/test_logging_and_incomplete_runs.py` (15 passed), `python -m qaq.router.train --config configs/router_train_real.yaml`, `python -m qaq.evaluate --config configs/router_eval_real.json --artifact-index configs/router_eval_real_artifacts.json --skip-output-dir-check --print-json`, `python -m qaq.router.train --health-check`, and `python -m pytest -q` (`119 passed, 2 skipped`). Re-ran `python -m qaq.router.train --config configs/router_train_llama31_8b_sampled.yaml` with escalated CUDA access; it now fails on the local RTX 4050 with the shared-reference estimate of 15.46 GiB required before activations, not the old 30.42 GiB two-copy estimate. Full Llama training still requires the lab RTX 3090 or equivalent memory.
  - 2026-06-25T05:54:43+08:00: Completed the real first-milestone LLaMA/HellaSwag router checkpoint. Added `reference_batch_size` to `qaq.router.train` so the 128-train/32-validation subset uses bounded single-example LLaMA reference forwards instead of one activation-heavy batch. `python scripts/gpu_run.py --count 2 --min-free-mb 22000 --status-file runs/gpu-selector/hellaswag-router-train-full.json -- python -m qaq.router.train --config configs/router_train_llama31_8b_full_hellaswag.yaml` ran on `basic-2`, selected physical RTX 3090 GPUs 0 and 1, used real HellaSwag file-backed rows and `runs/llama31_8b_full_tensor_bitplanes/runtime_artifact_index.json`, and wrote `runs/llama_first_milestone/router/checkpoints/router_final.json`. The checkpoint is non-diagnostic, covers 64 controlled blocks, candidates `[4, 8]`, feature source `layer_output_pooled_shared_mha_ffn`, 128 training samples, 32 validation samples, 8192 target records, validation loss `0.6931017754088582`, and parameter update L2 `0.0009781714124595159`. A stale `INCOMPLETE` marker from the previous OOM was removed after fixing manifest completion to clean stale run-directory markers. This checkpoint unblocks first-milestone QAQ evaluation, but the full five-mode matrix is not complete.
- [x] Adaptive Inference Runtime (`doc/tasks/adaptive-inference-runtime.md`)
  - 2026-06-23T23:54:56+08:00: Implemented and verified adaptive runtime evidence gaps: explicit preflight checks for router precision metadata, selected artifact/block/tensor/granularity metadata validation, per-query adaptive traces, on-demand-off/on shared routing semantics, on-demand loader summaries, and adaptive acceptance guards for non-diagnostic constant precision and missing loader summaries. Verified with `python -m pytest -q tests/e2e/test_smoke_modes.py tests/regression/test_qaq_acceptance_guards.py`, `python -m pytest -q`, and `python -m qaq.evaluate --config configs/router_eval_real.json --artifact-index configs/router_eval_real_artifacts.json --skip-output-dir-check --print-json`. Module remains unchecked pending a dedicated adaptive verification pass against the shared result reporter added on 2026-06-24.
  - 2026-06-24T03:00:00+08:00: Completed the remaining shared-result-reporter comparability item for fake/CPU adaptive outputs. Added an E2E reporter-matrix test that builds result artifacts for `fp16`, `static_8bit`, `static_4bit`, `qaq_on_demand_off`, and `qaq_on_demand_on`, groups them through `qaq.report.build_report`, verifies all required modes are present with no missing modes, and checks QAQ routing plus on-demand loader report rows. Verified with `python -m pytest -q tests/e2e/test_smoke_modes.py tests/regression/test_qaq_acceptance_guards.py` and `python -m pytest -q` (111 passed, 1 CUDA skip). Evidence remains diagnostic fake/CPU comparability, not paper-scale QAQ evidence.
  - 2026-06-24T14:14:39+08:00: Added real selected-weight execution for adaptive Hugging Face runs when artifact refs use the full tensor-native index shape (`block_id -> tensor_name -> .qaq.safetensors/json`). The runtime still routes from reference hidden states, then executes each query under its own reconstructed router-selected weight overrides and uses those outputs for metrics. Legacy single-artifact indexes remain supported but are marked `mixed_precision_forward_applied: false`. Verified with a torch/HF-shaped integration test that changes actual model predictions through reconstructed bit-plane weights plus the full GPU-wrapped test suite. This is a real mechanism step, not paper-scale LLaMA benchmark evidence.
- [x] Dynamic Loader and Memory Residency Manager (`doc/tasks/dynamic-loader-and-memory-residency-manager.md`)
  - 2026-06-23T21:38:48+08:00: Implemented synchronous CPU-simulated on-demand loader requests, events, residency records, summaries, selected-plane materialization, cache-hit/release/failure tracking, transfer timing/byte accounting, invalid request validation, explicit CUDA-unavailable failure, missing-plane failure mapping, and simulated memory-capacity checks. Verified with `python -m pytest -q tests/unit/test_loader_validation.py tests/integration/test_on_demand_loader_simulation.py` and `python -m pytest -q`.
  - 2026-06-24T03:06:31+08:00: Implemented real CUDA materialization for selected bit-plane tensors when torch can access CUDA. The loader now validates `cuda:<id>` devices, transfers selected planes as `torch.uint8` tensors, records CUDA target devices and resident bytes, keeps CPU simulation for CPU runs, and still fails clearly when CUDA is unavailable. Verified with targeted normal pytest, escalated CUDA pytest, and the full suite.
  - 2026-06-24T04:22:00+08:00: Added tensor-native loader support for `.qaq.safetensors` artifacts. Native materialization reconstructs the requested effective bit width, packs selected MSB planes into a byte tensor, transfers that packed tensor to CPU or CUDA, and records resident bytes. Verified with normal pytest and escalated CUDA pytest for both JSON-plane and native packed-plane loader paths.
- [x] Evaluation Metrics and Results Reporter (`doc/tasks/evaluation-metrics-and-results-reporter.md`)
  - 2026-06-24T00:10:29+08:00: Implemented result artifact schema, metric aggregation hooks, comparison grouping, accepted/diagnostic/incomplete/invalid comparison states, missing-baseline/settings-mismatch/routing-summary/loader-summary acceptance guards, paper-reproduction scope guard, fake paper-table rows, report CLI, and optional `qaq.evaluate` result-artifact output. Verified with `python -m pytest -q tests/unit/test_results_schema.py tests/regression/test_qaq_acceptance_guards.py`, `python -m pytest -q tests/unit/test_results_schema.py tests/regression/test_qaq_acceptance_guards.py tests/integration/test_router_checkpoint_contract.py`, `python -m pytest -q`, `python -m qaq.evaluate --config configs/smoke.json --skip-output-dir-check --print-result-json`, and `python -m qaq.report --results tests/golden/result_artifact_static.json --print-json`. Fake/CPU runs remain diagnostic and are not accepted QAQ evidence. This provides the shared reporter needed by the Adaptive Inference Runtime, but the adaptive task checklist was not changed in this reporter-only pass.
  - 2026-06-24T14:14:39+08:00: Added comparison validation that rejects non-diagnostic `static_8bit`, `static_4bit`, `qaq_on_demand_off`, and `qaq_on_demand_on` results unless runtime metadata explicitly records `mixed_precision_forward_applied: true`. This prevents reconstructed-only artifact logging from being accepted as QAQ mixed-weight inference evidence.
  - 2026-06-24T23:20:00+08:00: Added the strict research acceptance contract to result artifacts and reports. Result artifacts now expose evidence level, fake/diagnostic flags, artifact scope/ref mode, mixed-forward evidence, benchmark fields, GPU selector record, accepted flag, and rejection reasons. Reports reject missing modes, mixed fake/real groups, partial/legacy artifacts, fake/smoke/router-health-check paths, and quantized/QAQ artifacts without actual mixed-precision forward evidence. Added structural first-milestone benchmark config stubs for real benchmark names plus a benchmark integration plan documenting that no benchmark support is accepted yet. Verified with `python -m pytest -q tests/unit/test_results_schema.py`, `python -m pytest -q tests/unit/test_config_validation.py`, `python -m pytest -q tests/regression/test_qaq_acceptance_guards.py`, and `python -m pytest -q tests/e2e/test_smoke_modes.py`.
- [x] Logging and Progress Tracking (`doc/tasks/logging-and-progress-tracking.md`)
  - 2026-06-23T18:17:22+08:00: Implemented structured JSONL events, manifest-registered durable logs, console progress state, timing measurement helper, completion/failure helpers, and incomplete-run tests. Verified with `python -m pytest -q tests/unit/test_logging_events.py tests/integration/test_logging_and_incomplete_runs.py` and `python -m pytest -q`.

## Full-Project Gates

- [ ] Build passes
- [x] Unit tests pass
- [ ] Lint passes
- [ ] Format check passes
- [ ] Type/static analysis passes if configured
- [x] Evaluator or benchmark passes if configured

## Progress Log

### 2026-06-23T18:13:18+08:00

- Completed project scaffold and Experiment Configuration and Run Manifest workstream.
- Added package/test scaffold support already present in the worktree: `pyproject.toml`, `qaq/`, `qaq/runtime/`, `qaq/router/`, `configs/`, and `tests/`.
- Added `qaq/config.py`, `qaq/manifest.py`, and `qaq/errors.py` with dependency-free config validation, categorized errors, JSON/TOML config loading, manifest serialization, status updates, incomplete markers, and a config-validation CLI.
- Added or verified checked-in config stubs: `configs/smoke.json`, `configs/llama31_8b_first_milestone.json`, `configs/smoke.toml`, and `configs/llama31_8b_first_milestone.toml`.
- Added config fixtures and `tests/unit/test_config_validation.py`.
- Evidence: `python -m pytest -q tests/unit/test_config_validation.py` passed with 23 tests.
- Evidence: `python -m pytest -q` passed with 25 tests.
- Evidence: `python -m qaq.config configs/smoke.json --skip-output-dir-check --print-json` passed.
- Evidence: `python -m qaq.config configs/llama31_8b_first_milestone.json --skip-output-dir-check --print-json` passed.
- Evidence: `python -m qaq.config configs/smoke.toml --skip-output-dir-check --print-json` passed.
- Evidence: `python -m qaq.config configs/llama31_8b_first_milestone.toml --skip-output-dir-check --print-json` passed.
- Evidence: `python -m qaq.config tests/fixtures/configs/invalid_mode.json --skip-output-dir-check` failed with exit code 2 and `invalid_mode`, as expected.
- Remaining gates: no lint, format, type/static analysis, evaluator, or benchmark commands are configured yet.

### 2026-06-23T18:17:22+08:00

- Completed Logging and Progress Tracking workstream.
- Added `qaq/status.py`, `qaq/logging.py`, and `qaq/progress.py` with structured event types, JSONL event writing, manifest log-path registration, console progress snapshots, separated timing measurement, and run completion/failure helpers.
- Added `tests/unit/test_logging_events.py` and `tests/integration/test_logging_and_incomplete_runs.py`.
- Evidence: `python -m pytest -q tests/unit/test_logging_events.py tests/integration/test_logging_and_incomplete_runs.py` passed with 8 tests.
- Evidence: `python -m pytest -q` passed with 33 tests.
- Remaining gates: no lint, format, type/static analysis, evaluator, or benchmark commands are configured yet.

### 2026-06-23T18:20:27+08:00

- Completed Block Registry and Precision Plan workstream.
- Added `qaq/blocks.py` and `qaq/precision_plan.py` with stable `layer_000.mha` / `layer_000.ffn` fake-transformer discovery, block descriptors, static 4/8-bit plans, fixed mixed profiles, QAQ router decision validation, and optional artifact-reference checks.
- Added `tests/fixtures/fake_transformer.py`, explicit local test package markers, and `tests/unit/test_block_registry.py`.
- Evidence: initial `python -m pytest -q tests/unit/test_block_registry.py` exposed an import collision with an installed `tests` package; fixed by adding local `tests/__init__.py` and `tests/fixtures/__init__.py`.
- Evidence: `python -m pytest -q tests/unit/test_block_registry.py` passed with 9 tests after the import fix.
- Evidence: `python -m pytest -q` passed with 42 tests.
- Remaining gates: no lint, format, type/static analysis, evaluator, or benchmark commands are configured yet.

### 2026-06-23T18:25:05+08:00

- Completed Model and Benchmark Adapter workstream for the approved fake/tiny scope.
- Added `qaq/data.py`, `qaq/benchmark_adapter.py`, and `qaq/model_adapter.py` with built-in and fixture-backed benchmark examples, prompt formatting, tokenization metadata, deterministic fake reference execution, block-keyed hidden features, fake architecture metadata, base-parameter freezing, local fake metadata loading, and clear errors for missing or unsupported model/dataset assets.
- Added `tests/integration/test_model_adapter_smoke.py` and `tests/fixtures/benchmarks/fake_smoke.jsonl`.
- Evidence: `python -m pytest -q tests/integration/test_model_adapter_smoke.py` passed with 4 tests.
- Evidence: `python -m pytest -q` passed with 46 tests.
- Remaining gates: no lint, format, type/static analysis, evaluator, or benchmark commands are configured yet. Real LLaMA-3.1-8B/Hugging Face model loading is deferred until an external dependency is approved.

### 2026-06-23T18:33:53+08:00

- Completed Quantization and Bit-Plane Store workstream for the small-tensor/fake-artifact scope.
- Added `qaq/quantization.py`, `qaq/bitplanes.py`, and `qaq/artifacts.py` with explicit per-tensor affine unsigned quantization metadata, uint identity quantized fixtures, MSB-truncation reconstruction, artifact metadata and checksum validation, compatibility checks, and deterministic JSON artifact serialization.
- Added `tests/unit/test_bitplanes.py`, `tests/integration/test_quantized_artifact_roundtrip.py`, `tests/golden/bitplanes_u8.json`, and `tests/fixtures/bitplanes/README.md`.
- Evidence: pre-change `python -m pytest -q` passed with 46 tests.
- Evidence: `python -m pytest -q tests/unit/test_bitplanes.py tests/integration/test_quantized_artifact_roundtrip.py` passed with 10 tests.
- Evidence: `python -m pytest -q` passed with 56 tests.
- Remaining gates: no lint, format, type/static analysis, evaluator, or benchmark commands are configured yet. Quantization is a documented small-tensor prototype path and not yet evidence of optimized LLaMA-3.1-8B quantized execution.

### 2026-06-23T18:41:44+08:00

- Completed Static and Fixed Mixed-Precision Runtime workstream for fake/tiny CPU execution.
- Added `qaq/runtime/common.py`, `qaq/runtime/static.py`, and `qaq/evaluate.py` with runtime output bundles, latency and memory events, `fp16` reference execution, static 4/8-bit artifact reconstruction, fixed mixed profiles, static-baseline acceptance guard, JSON artifact-index loading, and a minimal static-runtime CLI.
- Added `tests/integration/test_static_equivalent_profiles.py`, `tests/e2e/__init__.py`, and `tests/e2e/test_smoke_modes.py`.
- Evidence: `python -m pytest -q tests/integration/test_static_equivalent_profiles.py tests/e2e/test_smoke_modes.py` passed with 8 tests.
- Evidence: `python -m pytest -q` passed with 64 tests.
- Evidence: `python -m qaq.evaluate --config configs/smoke.json --skip-output-dir-check --print-json` passed for the fake FP16 smoke path.
- Remaining gates: no lint, format, or type/static analysis commands are configured yet. Quantized runtime paths currently validate bit-plane reconstruction and metadata in the fake CPU path; they are not optimized LLaMA-3.1-8B quantized execution or GPU memory evidence.

### 2026-06-23T19:10:17+08:00

- Completed Router Policy Module workstream for fake/tiny hidden-state routing.
- Added `qaq/router/types.py`, `qaq/router/checkpoint.py`, and `qaq/router/policy.py` with router checkpoint metadata, per-block linear router parameters, JSON checkpoint serialization, compatibility validation against model/block/candidate/feature metadata, softmax probability normalization, deterministic decision conversion, precision-plan generation, router traces, and routing summaries.
- Added `tests/unit/test_router_policy.py`, `tests/integration/test_router_checkpoint_contract.py`, and `tests/golden/router_decision.json`.
- Evidence: pre-change `python -m pytest -q` passed with 64 tests.
- Evidence: first targeted router run exposed a fixture tie where one fake hidden vector produced equal scores; adjusted the test fixture bias to make the intended high/low precision behavior explicit.
- Evidence: `python -m pytest -q tests/unit/test_router_policy.py tests/integration/test_router_checkpoint_contract.py` passed with 10 tests.
- Evidence: `python -m pytest -q` passed with 74 tests.
- Remaining gates: no lint, format, or type/static analysis commands are configured yet. Router policy is validated on fake hidden features and checkpoint metadata only; no trained router or paper-scale routing evidence is claimed.

### 2026-06-23T21:38:48+08:00

- Completed Dynamic Loader and Memory Residency Manager workstream for CPU small-tensor simulation.
- Added `qaq/runtime/loader.py` and `qaq/loader.py` with loader request validation, selected MSB-plane materialization from CPU-resident bit-plane artifacts, cache-hit and release handling, failure events, residency records, loader summaries, simulated transfer byte/timing accounting, and explicit failure for real CUDA requests until a tensor/CUDA runtime is approved.
- Added `tests/unit/test_loader_validation.py` and `tests/integration/test_on_demand_loader_simulation.py`.
- Evidence: pre-change `python -m pytest -q` passed with 74 tests.
- Evidence: `python -m pytest -q tests/unit/test_loader_validation.py tests/integration/test_on_demand_loader_simulation.py` passed with 8 tests.
- Evidence: `python -m pytest -q` passed with 82 tests.
- Remaining gates: no lint, format, or type/static analysis commands are configured yet. Loader behavior is a CPU simulation using small tensor artifacts and is not GPU memory or CPU-to-GPU transfer evidence for LLaMA-3.1-8B.

### 2026-06-23T23:08:25+08:00

- Replaced the diagnostic-only Router Training Pipeline with a minimal real router-training implementation.
- Updated `qaq/router/losses.py` and `qaq/router/train.py` with the documented `router_cost_cross_entropy` objective, file-backed non-diagnostic data validation, compatible bit-plane artifact loading, cost-derived real targets, router softmax cross-entropy, gradient updates to router weights/biases only, validation metrics, checkpoint metadata, CLI support, and a separate quick health-check command.
- Added `configs/router_train_real.yaml`, file-backed router training examples, local model/tokenizer metadata fixtures, checked-in bit-plane artifacts, `doc/router-training.md`, and `doc/residual-risk.md`.
- Kept diagnostic health checks separate through `python -m qaq.router.train --health-check`; this path is not an acceptance gate.
- Evidence: `python -m pytest -q tests/integration/test_router_checkpoint_contract.py` passed with 7 tests.
- Evidence: `python -m pytest -q tests/integration/test_logging_and_incomplete_runs.py` passed with 4 tests.
- Evidence: `python -m qaq.router.train --config configs/router_train_real.yaml` completed with 3 training samples, 12 target records, objective `router_cost_cross_entropy`, checkpoint `runs/router_train_real/checkpoints/router_step_0003.json`, and validation loss `0.693113872343573`.
- Evidence: `python -m qaq.evaluate --config configs/router_eval_real.json --artifact-index configs/router_eval_real_artifacts.json --skip-output-dir-check --print-json` loaded checkpoint `runs/router_train_real/checkpoints/router_step_0003.json`, emitted 8 routing decisions for 2 validation samples, and materialized 8 selected precision artifacts.
- Evidence: `python -m qaq.router.train --health-check` completed as a diagnostic health check.
- Evidence: `python -m pytest -q` passed with 86 tests.
- Remaining gates: no lint, format, or type/static analysis commands are configured yet. The official paper loss, training corpus, and paper-scale model runs remain unavailable, so this is local minimal real-training evidence rather than paper-scale QAQ reproduction evidence.

### 2026-06-23T23:54:56+08:00

- Advanced the Adaptive Inference Runtime workstream without marking it complete.
- Updated `qaq/runtime/adaptive.py` with router checkpoint precision preflight, selected artifact metadata validation before materialization, per-query adaptive traces, and an adaptive acceptance guard for routing summaries, non-diagnostic constant precision, adaptive traces, and on-demand loader summaries.
- Updated `tests/e2e/test_smoke_modes.py` to run fake QAQ on-demand off and on with the same router checkpoint and artifact index, verify shared routing summaries and precision plans, verify `qaq_on_demand_on` loader summaries, and exercise the evaluate CLI for on-demand mode.
- Added `tests/regression/test_qaq_acceptance_guards.py` and `tests/regression/__init__.py` for constant-precision, missing-loader-summary, and checkpoint precision-mismatch guards.
- Evidence: `python -m pytest -q tests/e2e/test_smoke_modes.py tests/regression/test_qaq_acceptance_guards.py` passed with 8 tests.
- Evidence: `python -m pytest -q` passed with 92 tests.
- Evidence: `python -m qaq.evaluate --config configs/router_eval_real.json --artifact-index configs/router_eval_real_artifacts.json --skip-output-dir-check --print-json` completed for `qaq_on_demand_off`, loaded `runs/router_train_real/checkpoints/router_step_0003.json`, emitted 2 adaptive traces and 8 routing decisions, and flagged non-constant routing.
- Blocker: adaptive module final completion still depends on the Evaluation Metrics and Results Reporter workstream because there is no shared result artifact/reporting layer yet for static-baseline comparability.

### 2026-06-24T00:10:29+08:00

- Completed the Evaluation Metrics and Results Reporter workstream for the current dependency-free prototype scope.
- Added `qaq/metrics.py`, `qaq/results.py`, and `qaq/report.py` with result schema validation, target-loss-backed fake/local quality metrics, latency and memory aggregation, comparison grouping, accepted/diagnostic/incomplete/invalid states, QAQ acceptance guards, paper-reproduction scope validation, stable report rows, JSON save/load helpers, and a report CLI.
- Updated `qaq/evaluate.py` with optional machine-readable result artifact output while preserving existing runtime JSON output behavior.
- Added `tests/unit/test_results_schema.py`, `tests/golden/result_artifact_static.json`, and `tests/golden/report_rows.json`; extended `tests/regression/test_qaq_acceptance_guards.py` with reporter-level guards.
- Evidence: `python -m pytest -q tests/unit/test_results_schema.py tests/regression/test_qaq_acceptance_guards.py` passed with 19 tests.
- Evidence: after fixing compatibility with the existing `router_acceptance` metric name, `python -m pytest -q tests/unit/test_results_schema.py tests/regression/test_qaq_acceptance_guards.py tests/integration/test_router_checkpoint_contract.py` passed with 26 tests.
- Evidence: `python -m pytest -q` passed with 107 tests.
- Evidence: `python -m qaq.evaluate --config configs/smoke.json --skip-output-dir-check --print-result-json` passed and emitted a diagnostic fake/CPU result artifact.
- Evidence: `python -m qaq.report --results tests/golden/result_artifact_static.json --print-json` passed and reported the single-row comparison as invalid because required modes were missing.
- Remaining gates: no lint, format, or type/static analysis commands are configured. Reporter output validates fake/local metrics and acceptance guards; it is not paper-scale QAQ reproduction evidence.

### 2026-06-24T03:00:00+08:00

- Completed the Adaptive Inference Runtime workstream's remaining shared-result-reporter comparability item.
- Added `tests/e2e/test_smoke_modes.py` coverage that runs the fake CPU static and adaptive modes, writes per-mode result artifacts, groups them through `qaq.report.build_report`, verifies all five required comparison modes are present with no missing modes, and checks that adaptive rows expose non-constant routing plus on-demand loader activity.
- Evidence: `python -m pytest -q tests/e2e/test_smoke_modes.py tests/regression/test_qaq_acceptance_guards.py` passed with 13 tests.
- Evidence: `python -m pytest -q` passed with 111 tests and 1 expected CUDA skip.
- Remaining gates: no lint, format, or type/static analysis commands are configured. This verifies reporter comparability for fake/CPU diagnostic artifacts only; it is not LLaMA-3.1-8B, GPU memory, or paper-scale QAQ evidence.

### 2026-06-24T03:06:31+08:00

- Advanced the GPU on-demand loading and LLaMA bit-plane artifact path without claiming paper-scale completion.
- Updated `qaq/runtime/loader.py` so CUDA requests validate torch CUDA availability and materialize selected bit-plane values as `torch.uint8` tensors on the requested device.
- Updated `qaq/runtime/adaptive.py` so CUDA on-demand runs request `cuda:<gpu_id>` from the loader and report CUDA loader/runtime plus torch peak-memory metadata.
- Added `qaq/llama_bitplanes.py` and `doc/llama-bitplanes.md` for local Hugging Face LLaMA safetensor artifact generation, including real-weight probe artifacts and guarded full JSON generation.
- Added CUDA/materialization and LLaMA artifact tests.
- Evidence: normal targeted checks passed with `python -m pytest -q tests/unit/test_loader_validation.py tests/integration/test_on_demand_loader_simulation.py tests/integration/test_llama_bitplane_generation.py tests/regression/test_qaq_acceptance_guards.py` (`17 passed, 1 skipped`; skip was CUDA hidden in sandbox).
- Evidence: escalated CUDA check passed with `python -m pytest -q tests/integration/test_on_demand_loader_simulation.py::test_on_demand_loader_materializes_requested_planes_on_cuda_when_available tests/regression/test_qaq_acceptance_guards.py::test_cuda_on_demand_runtime_never_silently_uses_cpu_loader` (`2 passed`).
- Evidence: LLaMA probe command `python -m qaq.llama_bitplanes --model meta-llama/Llama-3.1-8B --output-dir runs/llama31_8b_bitplanes_probe --block-limit 1 --tensor-limit-per-block 2 --max-elements-per-tensor 16 --overwrite --print-json` resolved the local LLaMA-3.1-8B snapshot and generated 2 real-weight truncated artifacts from `layer_000.mha`.
- Evidence: router real-data command `python -m qaq.router.train --config configs/router_train_real.yaml` still completed with 3 training samples, 12 target records, checkpoint `runs/router_train_real/checkpoints/router_step_0003.json`, and validation loss `0.693113872343573`.
- Evidence: checkpoint-loaded evaluation `python -m qaq.evaluate --config configs/router_eval_real.json --artifact-index configs/router_eval_real_artifacts.json --skip-output-dir-check --print-json` still completed with 8 non-diagnostic routing decisions over 2 validation samples.
- Evidence: full suite `python -m pytest -q` passed with `114 passed, 1 skipped`.
- Remaining limitation: the actual visible escalated GPU is one 6 GiB RTX 4050, not the 8 RTX 3090 setup in `doc/requirements.md`; full LLaMA-3.1-8B QAQ execution and GPU-memory claims remain constrained until the intended hardware is available and the complete all-block native artifact/training run has been executed.

### 2026-06-24T03:56:30+08:00

- Continued the Router Training Pipeline workstream by sharing a single frozen reference adapter when `teacher_model` and `student_model` are the same exact model reference.
- Updated `qaq/router/train.py` so same-model training reuses the teacher reference forward as the student reference output for the current distortion-based objective, records shared-reference metadata, and counts one LLaMA model-weight copy in CUDA capacity preflight. Distinct teacher/student refs still use separate adapters.
- Updated `tests/integration/test_router_checkpoint_contract.py` with shared-reference checkpoint/audit assertions and LLaMA CUDA preflight coverage.
- Updated `doc/router-training.md` and `doc/residual-risk.md` to replace the old duplicate-adapter LLaMA memory blocker with the verified shared-reference preflight result.
- Evidence: `python -m pytest -q tests/integration/test_router_checkpoint_contract.py tests/integration/test_logging_and_incomplete_runs.py` passed with 15 tests.
- Evidence: `python -m qaq.router.train --config configs/router_train_real.yaml` completed with 3 training samples, 12 target records, objective `router_cost_cross_entropy`, checkpoint `runs/router_train_real/checkpoints/router_step_0003.json`, and validation loss `0.693113872343573`.
- Evidence: `python -m qaq.evaluate --config configs/router_eval_real.json --artifact-index configs/router_eval_real_artifacts.json --skip-output-dir-check --print-json` reloaded that checkpoint and emitted 8 non-diagnostic routing decisions over 2 validation samples.
- Evidence: `python -m qaq.router.train --health-check` completed as a diagnostic-only health check with 2 training samples, 8 target records, and validation loss `0.693144350913909`.
- Evidence: normal sandbox `python -m qaq.router.train --config configs/router_train_llama31_8b_sampled.yaml` failed with `cuda_unavailable`, then escalated GPU access reached the intended preflight and failed with `insufficient_cuda_memory`: the shared path needs 15.46 GiB free before activations, while cuda:0 reports 4.96 GiB free of 6.00 GiB total.
- Evidence: full suite `python -m pytest -q` passed with `119 passed, 2 skipped`.
- Remaining limitation: this is not a successful LLaMA router-training run; it only verifies the improved preflight and local real-data training path. The visible GPU is still too small for base LLaMA 3.1 8B reference execution.

### 2026-06-24T10:33:57+08:00

- Added the remote GPU command guard for heavy router training and related ML commands.
- Added `scripts/gpu_run.py`, which queries `nvidia-smi`, selects eligible physical RTX 3090 GPU IDs from the lab server range, records the selected IDs and PyTorch logical mapping, sets `CUDA_VISIBLE_DEVICES` only for the child command, and fails before launching when no GPU satisfies `--count`, `--min-free-mb`, and the default `RTX 3090` name filter.
- Added unit coverage with fake `nvidia-smi` output proving the wrapper selects the freest physical GPUs, rejects local/non-lab GPU names by default, exports the expected child-process environment, writes a status file, and does not run the child command when no suitable GPU is free.
- Updated `AGENTS.md`, `doc/router-training.md`, and `doc/residual-risk.md` so heavy training, inference, evaluation, benchmark, and large-model-loading examples use `python scripts/gpu_run.py --count <N> --min-free-mb <MB> -- <command>` instead of assuming physical GPU 0 or direct local execution.
- Evidence: `python -m py_compile scripts/gpu_run.py` passed.
- Evidence: `python -m pytest -q tests/unit/test_gpu_run.py` passed with 4 tests.
- Evidence: `python -m pytest -q tests/unit` passed with 70 tests.
- Full training, inference, evaluation, benchmark, and large-model-loading commands were not run in this local cycle because they now require the lab-server GPU selector and suitable free RTX 3090 devices.

### 2026-06-24T13:04:56+08:00

- Ran the next QAQ experiment workflow on `basic-2`, which the GPU selector confirmed as the lab RTX 3090 server with physical GPU IDs 0-7 eligible. All GPU-wrapped commands selected physical GPU 6 and mapped child-process `cuda:0` to physical 6.
- Added `doc/experiments/2026-06-24-qaq-experiment-report.md` with hostname, commit, Python/package environment, GPU selector records, exact commands, output directories, artifacts, metrics, diagnostic-only scope, non-diagnostic small-scale evidence, blockers, claims, and next experiment.
- Evidence: `python -m pytest -q` passed with `125 passed in 9.56s`.
- Evidence: diagnostic smoke config validation and GPU-wrapped smoke evaluation completed, but remain diagnostic only: `python -m qaq.config configs/smoke.json --skip-output-dir-check --print-json` and `python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.evaluate --config configs/smoke.json --skip-output-dir-check --print-json`.
- Evidence: non-diagnostic local-fixture router training completed through the GPU selector with 3 file-backed training samples, 12 target records, checkpoint `runs/router_train_real/checkpoints/router_step_0003.json`, target audit `runs/router_train_real/router_targets.json`, validation loss `0.693113872343573`, and non-diagnostic checkpoint/target metadata.
- Evidence: local-fixture checkpoint-loaded evaluation completed through the GPU selector, reloaded `runs/router_train_real/checkpoints/router_step_0003.json`, emitted 8 non-diagnostic routing decisions over 2 validation samples, and flagged non-constant routing.
- Evidence: sampled LLaMA artifact preparation completed through the GPU selector with 64 artifacts under `runs/llama31_8b_bitplanes_sampled`, `artifact_scope` recorded as `sampled_weight_values`, and sample artifact metadata showing real BF16 LLaMA safetensor source shards and tensor names. These artifacts are explicitly not full quantized inference artifacts.
- Evidence: one-step sampled LLaMA router training completed through the GPU selector with checkpoint `runs/router_train_llama31_8b_sampled/checkpoints/router_step_0001.json`, target audit `runs/router_train_llama31_8b_sampled/router_targets.json`, 1 training sample, 64 target records, validation loss `0.6929037649795032`, parameter update L2 `0.0020548994143244448`, non-diagnostic metadata, and completed manifest/logs.
- Blocker: checkpoint-loaded LLaMA evaluation using the produced checkpoint and sampled artifact index reached model loading but failed result aggregation with `unsupported_metric: unsupported metric 'router_acceptance' has no registered aggregator and no target-derived losses`. The blocked command is `python scripts/gpu_run.py --count 1 --min-free-mb 18000 -- python -m qaq.evaluate --config runs/router_train_llama31_8b_sampled/eval_qaq_on_demand_off.json --artifact-index runs/llama31_8b_bitplanes_sampled/artifact_index.json --result-output runs/router_train_llama31_8b_sampled/eval_qaq_on_demand_off/result_artifact.json`.
- Remaining limitation: no accepted paper-scale QAQ evidence was produced. The successful LLaMA run is non-diagnostic small-scale sampled-artifact router-training evidence only; full QAQ inference, real benchmark metrics, comparable static baselines, on-demand memory savings, and full paper reproduction remain blocked.

### 2026-06-24T13:49:10+08:00

- Continued the Model and Benchmark Adapter workstream to address the current sampled LLaMA evaluation blocker while keeping the pass scoped to `qaq/model_adapter.py` and `tests/integration/test_model_adapter_smoke.py`.
- Added Hugging Face target-token negative log-likelihood computation for reference outputs when benchmark examples include targets. The adapter now returns finite target-derived losses and records `loss_source: hf_target_token_nll` plus `target_loss_count` metadata.
- Added a tiny in-memory torch/Hugging Face-style adapter test that exercises target-loss reporting without loading real LLaMA weights or using a GPU.
- Evidence: `python -m pytest -q tests/integration/test_model_adapter_smoke.py` passed with `6 passed`.
- Evidence: the previously blocked command now completes and writes `runs/router_train_llama31_8b_sampled/eval_qaq_on_demand_off/result_artifact.json`: `python scripts/gpu_run.py --count 1 --min-free-mb 18000 -- python -m qaq.evaluate --config runs/router_train_llama31_8b_sampled/eval_qaq_on_demand_off.json --artifact-index runs/llama31_8b_bitplanes_sampled/artifact_index.json --result-output runs/router_train_llama31_8b_sampled/eval_qaq_on_demand_off/result_artifact.json`.
- Evidence: `python -m pytest -q tests/integration/test_router_checkpoint_contract.py` passed with `10 passed`.
- Evidence: full suite passed under the GPU selector with `python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m pytest -q` (`126 passed in 9.32s`).
- Remaining limitation: the LLaMA result artifact is still non-diagnostic small-scale fixture evidence over sampled 16-value bit-plane artifacts. It is not accepted QAQ benchmark evidence and does not replace the required comparable FP16/static/QAQ mode matrix on real benchmark data.


### 2026-06-24T14:14:39+08:00

- Continued the Adaptive Inference Runtime / static-runtime integration path to implement the next real QAQ mechanism gap: applying selected bit-plane artifacts to actual Hugging Face model weights during forward execution.
- Added `qaq/runtime/weight_overrides.py` with full tensor-index coverage detection, selected artifact reconstruction to torch tensors, per-query batch slicing, and output recombination.
- Updated the Hugging Face adapter to support temporary parameter-data overrides with shape/name validation and restoration after forward; fake adapters now fail clearly if weight overrides are requested.
- Updated static and adaptive runtimes to accept both legacy bit-width artifact indexes and full tensor-native indexes. With full tensor indexes, static quantized modes execute one selected-weight forward and adaptive modes execute each query under its router-selected reconstructed weights.
- Updated result comparison validation so non-diagnostic quantized/QAQ modes cannot be accepted unless `mixed_precision_forward_applied: true` is present in runtime metadata.
- Added `tests/integration/test_mixed_weight_runtime.py`, which builds real tensor-native bit-plane artifacts for a tiny HF-shaped torch module, runs adaptive routing with a full tensor index, verifies predictions change only through reconstructed selected weights, and verifies original model parameters are restored.
- Evidence: `python -m pytest -q tests/integration/test_mixed_weight_runtime.py tests/integration/test_model_adapter_smoke.py tests/integration/test_tensor_bitplane_artifacts.py tests/unit/test_results_schema.py tests/regression/test_qaq_acceptance_guards.py` passed with `30 passed`.
- Evidence: full suite passed through the lab GPU selector with physical GPU 6 selected: `python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m pytest -q` (`128 passed in 7.75s`).
- Remaining limitation: this verifies the real selected-weight mechanism on a tiny HF-shaped torch module and full tensor artifact index. It does not generate the complete LLaMA-3.1-8B all-block artifact set or run the required FP16/static/QAQ benchmark matrix, so no accepted paper-scale QAQ claim is made.

### 2026-06-24T19:01:01+08:00

- Continued the Quantization and Bit-Plane Store / runtime-contract follow-up for LLaMA artifact indexes.
- Updated `qaq/llama_bitplanes.py` so full-runtime artifact generation can require complete non-truncated tensor coverage and fail with `incomplete_tensor_artifact_index` instead of silently treating an incomplete request as accepted evidence.
- Updated the LLaMA generation manifest fields to make sampled/truncated probes, partial tensor indexes, full tensor-native runtime artifacts, and accepted full quantized inference artifacts machine-readable.
- Updated `tests/integration/test_llama_bitplane_generation.py` to assert runtime indexes use `block_id -> tensor_name -> artifact_path`, never legacy `"4"`/`"8"` keys for the tensor-native runtime path.
- Added tiny full-layer safetensors coverage for one LLaMA-shaped layer with MHA `q_proj`, `k_proj`, `v_proj`, `o_proj` and FFN `gate_proj`, `up_proj`, `down_proj`, proving `artifact_ref_mode(...) == "full_tensor_index"` for complete tensor-native refs.
- Evidence: `python -m pytest -q tests/integration/test_llama_bitplane_generation.py tests/integration/test_mixed_weight_runtime.py tests/unit/test_results_schema.py` passed with 17 tests.
- Evidence: `python -m pytest -q tests/unit tests/integration` passed with 114 tests.
- No large LLaMA weights, router training, full inference, evaluation, benchmark, or GPU memory measurement was run locally. Full LLaMA artifact generation remains a lab-server task through `scripts/gpu_run.py`.
### 2026-06-24T23:20:00+08:00

- Continued the Evaluation Metrics and Results Reporter plus first-milestone config workstream in IMPLEMENT mode.
- Added `doc/acceptance-contract.md` with three evidence levels: `diagnostic_health_check`, `real_path_implemented`, and `accepted_experiment_result`.
- Updated `qaq/results.py` so every result artifact carries acceptance fields and hard rejection reasons; inconsistent artifacts such as fake datasets marked accepted fail schema validation.
- Updated report validation so a comparison cannot be accepted unless every artifact passes the contract and the five required modes are present under comparable settings.
- Added `doc/benchmark-integration-plan.md` and structural configs under `configs/benchmarks/llama_first_milestone/` for HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, and WikiText-2 across `fp16`, `static_8bit`, `static_4bit`, `qaq_on_demand_off`, and `qaq_on_demand_on`. PTB remains a documented TODO because the current benchmark path has no real PTB adapter.
- Added/updated unit and regression tests proving fake smoke, router health-check metadata, partial tensor indexes, mixed fake/real report inputs, missing modes, and quantized results without mixed-forward evidence cannot be accepted.
- Evidence: `python -m pytest -q tests/unit/test_results_schema.py` passed with 16 tests.
- Evidence: `python -m pytest -q tests/unit/test_config_validation.py` passed with 24 tests and validates the 30 benchmark config stubs structurally.
- Evidence: `python -m pytest -q tests/regression/test_qaq_acceptance_guards.py` passed with 11 tests and 1 local CUDA-policy skip.
- Evidence: `python -m pytest -q tests/e2e/test_smoke_modes.py` passed with 5 tests.
- Evidence: `python -m pytest -q tests/unit tests/regression tests/e2e` passed with 92 tests and 1 expected CUDA-policy skip.
- No real benchmark, inference, large-model loading, training, or GPU memory measurement was run locally. No accepted QAQ result is claimed.


### 2026-06-24T23:45:00+08:00

- Patched the benchmark data loader so supported first-milestone benchmark names such as `hellaswag` resolve from local JSON/JSONL real-data files under `QAQ_BENCHMARK_DATA_ROOT` or from an already-cached Hugging Face `datasets` copy using local files only.
- Updated `qaq/benchmark_adapter.py` so `prompt_format: lm_eval:<task>` is accepted by the repo-native formatter.
- Kept missing real data as a hard failure with `benchmark_dataset_unavailable`; no fake fallback is introduced for real benchmark names.
- Evidence: `python -m py_compile qaq/data.py qaq/benchmark_adapter.py` passed.
- Evidence: `python -m pytest -q tests/integration/test_model_adapter_smoke.py` passed with 8 tests.
- Remaining limitation: this fixes benchmark-name data resolution only. It is not lm-evaluation-harness integration and not accepted benchmark evidence until a real data file/cache, all five modes, GPU selector records, metrics, and `qaq.report` acceptance are present.

### 2026-06-25T00:41:38+08:00

- Continued the Evaluation Metrics / Static Runtime / Model and Benchmark Adapter workstreams to repair the real HellaSwag FP16 OOM path without adding torchrun/DDP.
- Added evaluator config fields and `qaq.evaluate` CLI overrides for `max_examples`, `eval_batch_size`, `hf_device_map`, and `hf_max_memory_per_gpu`; CLI overrides are copied into result runtime metadata.
- Updated static and adaptive evaluation to stream examples through `eval_batch_size` micro-batches, aggregate compact predictions/losses, avoid full hidden-state and full-logit retention by default outside router feature extraction, and record processed/total examples, subset/debug status, micro-batch count, peak CUDA memory, and model device map metadata.
- Updated the Hugging Face adapter to support compact forwards, `torch.inference_mode()`, peak CUDA memory metadata, and single-process Transformers `device_map="auto"` sharding without a post-load `model.to(...)` call.
- Updated result acceptance so `max_examples`/subset runs are marked `subset_debug_run` and cannot be accepted as full benchmark evidence.
- Updated `doc/benchmark-integration-plan.md` to document that ordinary torchrun/DDP is intentionally out of scope because it replicates the model unless a tensor-parallel path is implemented; future Transformers `tp_plan="auto"` support remains out of scope.
- Evidence: `python -m py_compile qaq/config.py qaq/evaluate.py qaq/model_adapter.py qaq/runtime/adaptive.py qaq/runtime/static.py qaq/runtime/weight_overrides.py qaq/results.py` passed.
- Evidence: focused tests passed with `python -m pytest -q tests/unit/test_config_validation.py tests/integration/test_model_adapter_smoke.py tests/integration/test_static_equivalent_profiles.py` (`48 passed`).
- Evidence: adaptive streaming smoke coverage passed with `python -m pytest -q tests/integration/test_mixed_weight_runtime.py tests/e2e/test_smoke_modes.py` (`6 passed`).
- Evidence: requested local verification passed with `python -m pytest -q tests/unit tests/integration` (`131 passed`).
- Remaining limitation: no real HellaSwag benchmark, large-model load, GPU memory benchmark, or full FP16/static/QAQ comparison was run locally. Real evidence still requires lab RTX 3090 execution through `scripts/gpu_run.py`.

### 2026-06-25T04:05:18+08:00

- Continued the Model and Benchmark Adapter workstream only.
- Updated `qaq/model_adapter.py` so both fake and Hugging Face reference outputs carry explicit adapter provenance: adapter kind, model source, fake model/tokenizer flags, fake dataset flag, fixture-only flag, real-benchmark flag, diagnostic flag, selected GPU IDs, dataset sources, and context-length policy.
- Updated `tests/integration/test_model_adapter_smoke.py` to assert fake-smoke outputs remain diagnostic-only, HellaSwag rows loaded from a local benchmark root are recognized as real benchmark data, and injected TinyHF/mocked Hugging Face objects are labeled diagnostic rather than accepted real-adapter evidence.
- Updated `doc/tasks/model-and-benchmark-adapter.md` done conditions only for the metadata/provenance items actually verified. The module remains incomplete.
- Evidence: `python -m py_compile qaq/model_adapter.py tests/integration/test_model_adapter_smoke.py` passed.
- Evidence: `python -m pytest -q tests/integration/test_model_adapter_smoke.py` passed with 9 tests.
- Evidence: `python -m pytest -q tests/integration/test_static_equivalent_profiles.py` passed with 9 tests.
- Evidence: `python -m pytest -q tests/unit/test_config_validation.py tests/integration/test_model_adapter_smoke.py tests/integration/test_static_equivalent_profiles.py` passed with 48 tests.
- Remaining limitation: no real local `meta-llama/Llama-3.1-8B` snapshot verification, large checkpoint load, real GPU execution, or accepted benchmark artifact was run in this pass. Those remain lab-server tasks through `scripts/gpu_run.py`.

### 2026-06-25T04:14:33+08:00

- Continued the Model and Benchmark Adapter workstream only after the module still described itself around the fake smoke path.
- Updated `qaq/model_adapter.py` docstring and added `verify_model_adapter_config` plus `python -m qaq.model_adapter --config ... --print-json`. The verifier loads the configured adapter and tokenizer, loads local benchmark rows, builds a tokenized batch, emits architecture/provenance metadata, and keeps large weight loading opt-in with `--load-weights`.
- Added a lab-server-compatible large checkpoint verification command shape through `--load-weights`; this command must be launched through `scripts/gpu_run.py` for LLaMA-sized checkpoints.
- Updated `tests/integration/test_model_adapter_smoke.py` with a non-fake local LLaMA-shaped Hugging Face config directory, a non-fake tokenizer wrapper, local HellaSwag rows under `QAQ_BENCHMARK_DATA_ROOT`, and the new CLI. The test verifies `accepted_as_real_adapter_verification: true` without loading weights.
- Evidence: `python -m py_compile qaq/model_adapter.py tests/integration/test_model_adapter_smoke.py` passed.
- Evidence: `python -m pytest -q tests/integration/test_model_adapter_smoke.py` passed with 10 tests.
- Evidence: `python -m pytest -q tests/unit/test_config_validation.py tests/integration/test_model_adapter_smoke.py tests/integration/test_static_equivalent_profiles.py` passed with 49 tests.
- Remaining limitation: no actual cached `meta-llama/Llama-3.1-8B` snapshot verification, no lab RTX 3090 `--load-weights` execution, no full benchmark result artifact, and no accepted QAQ evidence was produced locally.

### 2026-06-25T05:54:43+08:00

- Produced the compatible real first-milestone router checkpoint for the LLaMA-3.1-8B HellaSwag subset matrix without using the sampled checkpoint as evidence.
- Added router-training `reference_batch_size` support and set `configs/router_train_llama31_8b_full_hellaswag.yaml` to `reference_batch_size: 1`, preserving the 128 training / 32 validation real HellaSwag subset while avoiding the prior all-examples activation OOM.
- Fixed `RunManifest.mark_completed()` so completed overwrite reruns remove stale `INCOMPLETE` files even when the fresh manifest started with `incomplete_marker: null`. The prior `runs/llama_first_milestone/router/INCOMPLETE` file contained the old CUDA OOM and was removed after the new manifest completed.
- Evidence: `python scripts/gpu_run.py --count 2 --min-free-mb 22000 --status-file runs/gpu-selector/hellaswag-router-train-full.json -- python -m qaq.router.train --config configs/router_train_llama31_8b_full_hellaswag.yaml` ran on `basic-2`, selected physical RTX 3090 GPUs `[0, 1]`, completed, and wrote `runs/llama_first_milestone/router/checkpoints/router_final.json`.
- Evidence: `runs/llama_first_milestone/router/manifest.json` reports `status: completed`, `failure: null`, and artifact paths for `router_final_checkpoint`, `router_targets`, and `router_train_log`.
- Evidence: `runs/llama_first_milestone/router/router_targets.json` reports `diagnostic_training: false`, `training_sample_count: 128`, `validation_sample_count: 32`, `reference_batch_size: 1`, `target_record_count: 8192`, and `validation_target_record_count: 2048`.
- Evidence: checkpoint compatibility validation loaded `runs/llama_first_milestone/router/checkpoints/router_final.json` against both HellaSwag QAQ first-milestone configs and verified `diagnostic: false`, model `meta-llama/Llama-3.1-8B`, 64 blocks, candidates `[4, 8]`, hidden size 4096, and feature source `layer_output_pooled_shared_mha_ffn`.
- Evidence: `python -m py_compile qaq/router/train.py tests/integration/test_router_checkpoint_contract.py` passed.
- Evidence: `python -m pytest -q tests/unit/test_config_validation.py::test_manifest_completion_removes_incomplete_marker tests/unit/test_config_validation.py::test_manifest_completion_removes_stale_incomplete_marker` passed with 2 tests.
- Evidence: `python -m pytest -q tests/integration/test_router_checkpoint_contract.py` passed with 15 tests.
- Evidence: `python -m pytest -q tests/regression/test_qaq_acceptance_guards.py` passed with 17 tests and 1 expected CUDA-wrapper skip.
- Remaining limitation: this completes the router checkpoint prerequisite only. The full first-milestone HellaSwag matrix is still incomplete until `fp16`, `static_8bit`, `static_4bit`, `qaq_on_demand_off`, and `qaq_on_demand_on` result artifacts exist with compatible metadata and are aggregated through `qaq.report`.

