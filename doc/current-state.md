# Current State

## Product Goal

QAQ is a Python 3.12 research prototype for query-adaptive mixed-precision LLM inference. The intended system decomposes LLM weights into bit-plane artifacts, trains a query-conditioned router for MHA/FFN block precision choices, compares FP16/static 8-bit/static 4-bit/QAQ on-demand-off/QAQ on-demand-on modes, and reports score or perplexity, latency, GPU memory, routing, loader activity, logs, and reproducible metadata. Source-of-truth docs are `QAQ.pdf`, `doc/requirements.md`, `doc/high-level-design.md`, `doc/detailed-design.md`, and `doc/acceptance-contract.md`.

## Current Milestone

The repo has moved from scaffold to real-path prototype mechanisms, but it has not produced an accepted QAQ benchmark result. Current milestone work should close the smallest real adapter/benchmark verification gap before attempting the full LLaMA-3.1-8B five-mode matrix: run a bounded real HellaSwag FP16 evaluation on the lab RTX 3090 server using the real local Hugging Face model/tokenizer path and real benchmark data.

## Implemented

- Config parsing, validation, CLI overrides, run manifests, output-dir safety, and evaluator fields such as `max_examples`, `eval_batch_size`, `hf_device_map`, and `hf_max_memory_per_gpu` exist in `qaq/config.py` and `qaq/evaluate.py`, with coverage in `tests/unit/test_config_validation.py`.
- Structured JSONL logging, progress/status records, completion/failure markers, and incomplete-run handling exist in `qaq/logging.py`, `qaq/progress.py`, `qaq/status.py`, and `qaq/manifest.py`, with tests in `tests/unit/test_logging_events.py` and `tests/integration/test_logging_and_incomplete_runs.py`.
- MHA/FFN block discovery and precision-plan validation exist in `qaq/blocks.py` and `qaq/precision_plan.py`, with `tests/unit/test_block_registry.py`.
- JSON bit-plane artifacts and tensor-native `.qaq.safetensors` artifacts exist in `qaq/bitplanes.py`, `qaq/quantization.py`, `qaq/artifacts.py`, `qaq/tensor_bitplanes.py`, and `qaq/llama_bitplanes.py`; tests cover artifact roundtrips, loader materialization, sampled/full runtime-index shape, and incomplete full-runtime rejection.
- Static and adaptive runtime entry points exist in `qaq/runtime/static.py`, `qaq/runtime/adaptive.py`, `qaq/runtime/weight_overrides.py`, and `qaq/evaluate.py`. Recent progress records streaming micro-batches, compact outputs, peak CUDA memory metadata, and tiny Hugging Face-shaped selected-weight execution evidence in `tests/integration/test_static_equivalent_profiles.py` and `tests/integration/test_mixed_weight_runtime.py`.
- Router policy, checkpoint save/load, compatibility validation, routing summaries, and constant-precision guards exist in `qaq/router/policy.py`, `qaq/router/checkpoint.py`, and `qaq/router/types.py`, with unit, integration, and regression tests.
- Router training now has a minimal real objective, `router_cost_cross_entropy`, in `qaq/router/train.py` and `qaq/router/losses.py`; it loads file-backed samples, rejects prohibited non-diagnostic inputs, writes `router_targets.json`, updates router parameters, saves reloadable checkpoints, and records validation metrics. Evidence is logged in `doc/router-training.md` and `doc/tasks/progress.md`.
- Optional local Hugging Face/LLaMA adapter support exists in `qaq/model_adapter.py`: local snapshot/config resolution, LLaMA MHA/FFN metadata, reference forwards, hidden states, target-token NLL losses, temporary parameter overrides, compact forwards, and single-process `device_map="auto"` loading. Tests in `tests/integration/test_model_adapter_smoke.py` cover this with fixtures and mocked tiny HF-shaped modules.
- Benchmark data loading supports smoke/file inputs plus named first-milestone benchmarks such as `hellaswag` from `QAQ_BENCHMARK_DATA_ROOT` or cached Hugging Face `datasets` local files only; missing real data fails with `benchmark_dataset_unavailable` instead of fake fallback. See `qaq/data.py`, `qaq/benchmark_adapter.py`, `configs/README.md`, and `tests/integration/test_model_adapter_smoke.py`.
- Result artifacts, metric aggregation, report grouping, comparison validation, strict acceptance fields, and rejection reasons exist in `qaq/results.py`, `qaq/metrics.py`, and `qaq/report.py`, with schema and acceptance-guard tests in `tests/unit/test_results_schema.py` and `tests/regression/test_qaq_acceptance_guards.py`.
- `scripts/gpu_run.py` selects eligible physical RTX 3090 GPUs, records physical-to-logical CUDA mapping, writes status records, and refuses unsuitable devices; `tests/unit/test_gpu_run.py` verifies selection and failure behavior.
- Structural first-milestone configs exist for HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, and WikiText-2 across all five required modes under `configs/benchmarks/llama_first_milestone/`; config validation covers these stubs.

## Partially Implemented

- Model and Benchmark Adapter is explicitly not done in `doc/tasks/progress.md` and `doc/tasks/model-and-benchmark-adapter.md`: optional LLaMA/HF support exists, but a real local LLaMA adapter verification and real-subset benchmark/tokenization verification have not been recorded.
- LLaMA tensor-native bit-plane generation can produce accepted-shaped full runtime indexes and reject incomplete requests, but the full LLaMA-3.1-8B all-block artifact set has not been generated and validated on the lab server.
- Static/adaptive selected-weight execution is real mechanism evidence on tiny HF-shaped modules, not accepted LLaMA-3.1-8B quantized inference evidence.
- CUDA on-demand materialization exists for JSON and tensor-native artifacts, but accepted memory-savings and latency claims still need full QAQ runtime execution and comparable GPU measurements.
- Benchmark support is data-loading and structural-config support only. `doc/benchmark-integration-plan.md` says this is not accepted benchmark support until real result artifacts pass the full acceptance contract.
- `configs/router_train_llama31_8b_full_hellaswag.yaml` defines a first-milestone router-training shape, but it depends on exported real HellaSwag rows, full tensor-native LLaMA artifacts, and lab RTX 3090 execution.
- Optional runtime dependencies such as `torch`, `transformers`, `safetensors`, and possibly `datasets` are environment-provided; `pyproject.toml` declares no production dependencies.

## Diagnostic Only / Fake Only

- `configs/smoke.json`, `configs/smoke.toml`, `configs/router_train_smoke.yaml`, built-in `fake_smoke`, fake model/tokenizer identifiers, fake CPU runtimes, fixture-only data, and TinyHF-style mocked modules are diagnostic or mechanism-test paths.
- `fixed_mixed` is a validation mode, not a required paper comparison mode.
- `python -m qaq.router.train --health-check` is diagnostic only and may use generated fake artifacts.
- Fake CPU five-mode matrices prove schema, CLI, routing summary, loader summary, and reporter guards only. They are not LLaMA, real benchmark, latency, GPU memory, or paper-scale QAQ evidence.
- Sampled/truncated LLaMA artifacts and sampled one-step LLaMA router training are useful integration evidence but are explicitly not full quantized inference artifacts or accepted benchmark evidence.

## Not Yet Implemented

- Accepted full LLaMA-3.1-8B quantized execution using complete all-block tensor-native bit-plane artifacts.
- Accepted five-mode first-milestone result matrix for `fp16`, `static_8bit`, `static_4bit`, `qaq_on_demand_off`, and `qaq_on_demand_on` on the same model, tokenizer, benchmark split, prompt format, precision candidates, and metric.
- Accepted latency and peak GPU memory comparisons for static baselines, QAQ on-demand off, and QAQ on-demand on.
- Real held-out benchmark result artifacts that pass `doc/acceptance-contract.md`.
- PTB benchmark support; `configs/benchmarks/llama_first_milestone/TODO-ptb.md` says a PTB adapter or lm-eval integration is still required.
- Qwen3-4B and Qwen3-8B paths for full paper reproduction.
- CI, lint, format, and type-check gates. `pyproject.toml` configures pytest only.

## Blocked

- Heavy model loading, router training, full inference, full benchmark evaluation, and GPU memory measurement are blocked locally by `AGENTS.md`; they must run on the lab RTX 3090 server through `python scripts/gpu_run.py --count <N> --min-free-mb <MB> -- <command>`.
- Accepted QAQ claims are blocked until full tensor-native artifacts, real benchmark data, real model/tokenizer execution, all five comparable modes, mixed-forward evidence, GPU selector records, metrics, routing/loader summaries, and `qaq.report` acceptance exist.
- Full LLaMA router training is blocked until accepted full tensor-native artifacts and sufficient lab GPU memory are available; the documented local RTX 4050 preflight is insufficient.
- Real benchmark runs are blocked if `QAQ_BENCHMARK_DATA_ROOT` or an already-cached Hugging Face dataset copy is unavailable.

## Current Quality Gates

- Main configured local gate: `python -m pytest -q`.
- Useful focused local gates from `AGENTS.md`: `python -m pytest -q tests/unit`, `python -m pytest -q tests/integration`, `python -m pytest -q tests/e2e`, and `python -m qaq.config configs/smoke.json --skip-output-dir-check --print-json`.
- GPU-wrapped diagnostic gates from `AGENTS.md`: `python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.evaluate --config configs/smoke.json --skip-output-dir-check --print-json` and `python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.router.train --health-check`.
- Recent recorded gates in `doc/tasks/progress.md`: `python -m py_compile qaq/config.py qaq/evaluate.py qaq/model_adapter.py qaq/runtime/adaptive.py qaq/runtime/static.py qaq/runtime/weight_overrides.py qaq/results.py`; `python -m pytest -q tests/unit/test_config_validation.py tests/integration/test_model_adapter_smoke.py tests/integration/test_static_equivalent_profiles.py` passed with 48 tests; `python -m pytest -q tests/integration/test_mixed_weight_runtime.py tests/e2e/test_smoke_modes.py` passed with 6 tests; `python -m pytest -q tests/unit tests/integration` passed with 131 tests.
- Not configured: lint, format check, type/static analysis, CI, accepted benchmark runner, or full paper reproduction command.

## Last Verified Commands

- `pwd`: confirmed the working directory is `/nfs/home/s314511048/qaq`.
- `rg --files`: inventoried repository files, including `qaq/`, `configs/`, `tests/`, `scripts/`, and docs.
- `git status --short`: showed pre-existing uncommitted changes in `AGENTS.md`, `doc/acceptance-contract.md`, `doc/current-state.md`, `doc/tasks/model-and-benchmark-adapter.md`, and `doc/tasks/progress.md`.
- `git log --oneline -n 10`: reviewed recent commits through `c2d2f39 implemented true qunatized execution`.
- `git diff --stat` and targeted `git diff`: reviewed current documentation changes before replacing this dashboard.
- `cat`/`rg` reads over `AGENTS.md`, `README.md`, `pyproject.toml`, `doc/current-state.md`, `doc/tasks/progress.md`, `doc/residual-risk.md`, `doc/router-training.md`, `doc/requirements.md`, `doc/high-level-design.md`, `doc/detailed-design.md`, `doc/test-plan.md`, `doc/acceptance-contract.md`, `configs/`, and `tests/`: used to classify implemented, diagnostic, blocked, and next-step evidence.
- Not run in this pass: pytest, model loading, router training, inference, benchmark evaluation, and GPU memory measurement. This pass was a documentation/state audit only.

## Next 1-3 Actions

- Run the smallest real adapter/benchmark verification on the lab RTX 3090 server with real local LLaMA files and real HellaSwag data, for example: `QAQ_BENCHMARK_DATA_ROOT=/path/to/qaq-benchmarks python scripts/gpu_run.py --count 1 --min-free-mb 18000 --status-file runs/gpu-selector/hellaswag-fp16-subset.json -- python -m qaq.evaluate --config configs/benchmarks/llama_first_milestone/hellaswag/fp16.json --max-examples 8 --eval-batch-size 1 --hf-device-map auto --result-output runs/llama_first_milestone/hellaswag/fp16_subset/result_artifact.json --print-result-json`.
- If that succeeds, update `doc/tasks/model-and-benchmark-adapter.md`, `doc/tasks/progress.md`, and this dashboard with the exact hostname, selected physical GPU IDs, dataset path/cache provenance, output artifact path, metric/loss fields, and evidence level.
- Recommended next task: complete the bounded real HellaSwag FP16 adapter verification above; it is the smallest real task that does not require full LLaMA bit-plane generation or a five-mode QAQ matrix.

## Evidence Required Before Claiming Done

- For the recommended next task, the lab-server command must complete through `scripts/gpu_run.py` and record hostname, selected physical RTX 3090 GPU IDs, child `CUDA_VISIBLE_DEVICES`, command, git commit, Python environment, dataset root or cached dataset provenance, and output path.
- The result artifact must show a real model/tokenizer path for `meta-llama/Llama-3.1-8B`, a real `hellaswag` split, `dataset_is_fake: false`, `model_is_fake: false`, `diagnostic: false`, processed example counts, target-derived losses or benchmark metric fields, runtime metadata for `max_examples`, `eval_batch_size`, and `hf_device_map`, and no TinyHF/fake-smoke provenance.
- Because the recommended run uses `--max-examples`, it should be classified as real-subset or subset-debug adapter evidence, not accepted full benchmark evidence.
- Focused local checks should still pass after any code/doc update: `python -m pytest -q tests/unit/test_config_validation.py tests/integration/test_model_adapter_smoke.py tests/integration/test_static_equivalent_profiles.py`.
- Do not call the adapter task complete from fixture-only tests, mocked TinyHF modules, synthetic tensors, sampled artifacts, or smoke configs.

## Risks / Residual Risk

- The router objective is still an implementation assumption: `router_cost_cross_entropy` estimates quantized-student behavior from reconstruction distortion and a bit-cost term rather than an official QAQ loss.
- Full selected-weight execution has tiny HF-shaped evidence only; it has not been verified as accepted LLaMA-3.1-8B execution.
- Structural benchmark configs and real-data loading support are not accepted benchmark support by themselves.
- Subset real runs with `max_examples` are useful next evidence but cannot satisfy full benchmark acceptance.
- Full LLaMA artifact generation, QAQ five-mode evaluation, memory savings, and latency claims remain lab-server tasks.
- README is still only a one-line title, and setup documentation/dependency declaration is incomplete for optional real-model paths.
