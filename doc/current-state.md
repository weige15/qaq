# Current State

## Product Goal

QAQ is a Python 3.12 research prototype for query-adaptive mixed-precision LLM inference. The intended system decomposes LLM weights into bit-plane artifacts, trains a query-conditioned router for MHA/FFN block precision choices, compares against FP16/static 8-bit/static 4-bit baselines, and optionally loads only selected bit-planes from CPU to GPU on demand. `QAQ.pdf`, `doc/requirements.md`, `doc/high-level-design.md`, and `doc/detailed-design.md` remain the source of truth.

## Current Milestone

The scaffold has moved past basic plumbing: config, logging, block discovery, bit-plane artifacts, router policy/training, static/adaptive runtimes, result reporting, a GPU selector, sampled LLaMA artifact preparation, and tiny Hugging Face-shaped weight override execution all exist. The LLaMA bit-plane runtime-index contract is now verified for tiny local fixtures: accepted runtime indexes are tensor-name based, and sampled/partial probes are explicitly non-accepted. The next milestone is generating and validating the full LLaMA-3.1-8B all-block tensor-native artifact set on the lab server before any first-milestone LLaMA matrix or benchmark run is treated as meaningful.

## Implemented


- `doc/acceptance-contract.md` now defines the strict research acceptance contract. Result artifacts expose evidence level, fake/diagnostic flags, artifact scope/ref mode, mixed-forward evidence, benchmark fields, GPU selector record, accepted flag, and rejection reasons. `qaq.report` rejects missing five-mode matrices, mixed fake/real inputs, partial or legacy artifacts, router health-check metadata, fake/smoke/fixture data, and quantized/QAQ rows without real mixed-forward evidence.
- `configs/benchmarks/llama_first_milestone/` now contains structural LLaMA-3.1-8B configs for HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, and WikiText-2 across all five required modes. PTB remains documented as a TODO until a real adapter exists. These configs are not benchmark evidence.
- `doc/benchmark-integration-plan.md` documents the optional lm-evaluation-harness integration path and GPU-selector command templates. Benchmark support is still not accepted until at least one real benchmark produces a QAQ result artifact through `qaq.evaluate` and passes the acceptance contract.
- Config parsing, validation, run manifests, status updates, and output-dir safety are implemented in `qaq/config.py`, `qaq/manifest.py`, and `qaq/status.py`, with coverage in `tests/unit/test_config_validation.py`.
- Structured JSONL logging, progress tracking, completion/failure markers, and incomplete-run handling are implemented in `qaq/logging.py` and `qaq/progress.py`, with tests in `tests/unit/test_logging_events.py` and `tests/integration/test_logging_and_incomplete_runs.py`.
- MHA/FFN block discovery and precision-plan validation are implemented in `qaq/blocks.py` and `qaq/precision_plan.py`, with tests in `tests/unit/test_block_registry.py`.
- JSON bit-plane artifacts and tensor-native `.qaq.safetensors` artifacts are implemented in `qaq/bitplanes.py`, `qaq/quantization.py`, `qaq/artifacts.py`, and `qaq/tensor_bitplanes.py`, with unit, integration, and golden coverage under `tests/unit/`, `tests/integration/`, and `tests/golden/`.
- Static and adaptive runtime entry points exist in `qaq/runtime/static.py`, `qaq/runtime/adaptive.py`, and `qaq/evaluate.py`. They support fake/local smoke paths and, when a full per-tensor artifact index is present, Hugging Face-shaped weight overrides through `qaq/runtime/weight_overrides.py`.
- Router policy, router checkpoint validation, routing summaries, and non-constant routing guards are implemented in `qaq/router/policy.py`, `qaq/router/checkpoint.py`, and `qaq/router/types.py`, with tests in `tests/unit/test_router_policy.py`, `tests/integration/test_router_checkpoint_contract.py`, and `tests/regression/test_qaq_acceptance_guards.py`.
- Router training has a minimal real objective, `router_cost_cross_entropy`, implemented in `qaq/router/train.py` and `qaq/router/losses.py`. It loads file-backed samples, rejects smoke/fixture/sampled/truncated inputs for non-diagnostic runs, validates accepted full tensor-native artifact manifests when present, writes `router_targets.json`, updates router parameters, saves reloadable step checkpoints plus `router_final.json`, and records validation metrics.
- Optional local Hugging Face/LLaMA adapter support exists in `qaq/model_adapter.py`: local snapshot/config resolution, tokenizer/model loading with local files, LLaMA MHA/FFN metadata, hidden states, reference forwards, target-token NLL losses, and temporary parameter overrides.
- Result artifacts, metric aggregation, comparison grouping, report rows, and acceptance guards are implemented in `qaq/results.py`, `qaq/metrics.py`, and `qaq/report.py`, with tests in `tests/unit/test_results_schema.py` and `tests/regression/test_qaq_acceptance_guards.py`.
- `scripts/gpu_run.py` is implemented and tested in `tests/unit/test_gpu_run.py`; it selects eligible physical RTX 3090 GPUs, records physical-to-logical CUDA mapping, and refuses unsuitable devices.
- `doc/experiments/2026-06-24-qaq-experiment-report.md` records a lab-server RTX 3090 workflow that passed prior correctness gates, generated sampled LLaMA artifacts, trained a one-step sampled LLaMA router, and later produced a sampled checkpoint-loaded evaluation artifact.

## Partially Implemented

- LLaMA bit-plane generation in `qaq/llama_bitplanes.py` now writes tensor-name runtime indexes, labels `partial_tensor_index` and sampled/truncated probes as non-accepted, and can fail incomplete full-runtime requests with `incomplete_tensor_artifact_index`. The focused tests now cover sampled partial probes and full tiny tensor-native indexes.
- Tiny Hugging Face-shaped adaptive weight override execution is covered by `tests/integration/test_mixed_weight_runtime.py`; this proves the mechanism on a small CPU model, not on LLaMA-3.1-8B.
- CUDA on-demand materialization exists in `qaq/runtime/loader.py`, including tensor-native artifacts, but accepted memory-savings claims still need full runtime execution and comparable GPU measurements.
- Sampled LLaMA artifact preparation and sampled LLaMA one-step router training exist, but sampled 16-value artifacts are explicitly not full quantized inference artifacts and are rejected for non-diagnostic router-training preflight.
- `configs/llama31_8b_first_milestone.json` and `.toml` are stubs, not a complete five-mode first-milestone matrix.
- `README.md` is only a one-line title, and `pyproject.toml` declares no dependencies even though optional LLaMA paths require environment-provided packages such as `torch`, `transformers`, and `safetensors`.

## Diagnostic Only / Fake Only

- `configs/smoke.json`, `configs/smoke.toml`, `configs/router_train_smoke.yaml`, built-in `fake_smoke`, fake model/tokenizer identifiers, fake CPU runtimes, and fixture-backed fake adapters are diagnostic/local health paths.
- `fixed_mixed` is a diagnostic validation mode and does not replace required QAQ modes.
- `python -m qaq.router.train --health-check` is diagnostic only and may generate temporary fake artifacts.
- Fake CPU static/adaptive mode matrices prove schema, routing, loader, and reporter guards only. They are not LLaMA, GPU memory, latency, or paper-scale QAQ evidence.
- The checked-in local router fixture path is useful non-diagnostic implementation evidence, but it is still a tiny file-backed fixture path and not accepted benchmark evidence.

## Not Yet Implemented

- Accepted full quantized LLaMA-3.1-8B execution using full all-tensor bit-plane artifacts.
- A complete first-milestone result matrix for `fp16`, `static_8bit`, `static_4bit`, `qaq_on_demand_off`, and `qaq_on_demand_on` on the same model, tokenizer, dataset split, prompt format, precision candidates, and metric.
- Real held-out benchmark adapters/configs for the full paper set: HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande, WikiText-2, and PTB.
- Accepted latency and peak GPU memory comparisons for static baselines, QAQ on-demand off, and QAQ on-demand on.
- Qwen3-4B and Qwen3-8B paths for full paper reproduction.
- CI, lint, format, and type-check gates. `pyproject.toml` configures pytest only.

## Blocked

- Heavy model loading, training, inference, evaluation, benchmarks, and GPU memory measurements are blocked locally by repository policy. They must run on the lab RTX 3090 server through `python scripts/gpu_run.py --count <N> --min-free-mb <MB> -- <command>`.
- Accepted QAQ benchmark claims remain blocked until full artifacts, verified quantized runtime use, real held-out benchmark data, all required static/QAQ modes, comparable result artifacts, and GPU-selector records exist.

## Current Quality Gates

- Configured local gate: `python -m pytest -q`.
- Useful focused gates are documented in `AGENTS.md`: `tests/unit`, `tests/integration`, `tests/e2e`, config validation, GPU-wrapped smoke evaluation, and GPU-wrapped router health check.
- Current focused gate passed: `python -m pytest -q tests/integration/test_llama_bitplane_generation.py tests/integration/test_mixed_weight_runtime.py tests/unit/test_results_schema.py` passed with 17 tests.
- Current broader lightweight gate passed: `python -m pytest -q tests/unit tests/integration` passed with 114 tests.
- No configured lint, format, type/static analysis, or CI gate was found in `pyproject.toml`.

## Last Verified Commands

- `git status --short`: showed `M doc/current-state.md` before this pass; final status shows scoped updates in `doc/current-state.md`, `doc/tasks/progress.md`, `qaq/llama_bitplanes.py`, and `tests/integration/test_llama_bitplane_generation.py`.
- `sed`/`rg` reads over `AGENTS.md`, the implementation-loop skill, `doc/current-state.md`, `doc/requirements.md`, `doc/high-level-design.md`, `doc/detailed-design.md`, `doc/test-plan.md`, `doc/tasks/progress.md`, relevant task docs, `doc/residual-risk.md`, source entry points, and representative tests.
- `git diff --stat` and targeted `git diff` reads reviewed the final source, test, and documentation changes.
- `python -m pytest -q tests/integration/test_llama_bitplane_generation.py tests/integration/test_mixed_weight_runtime.py tests/unit/test_results_schema.py`: passed with 17 tests.
- `python -m pytest -q tests/unit tests/integration`: passed with 114 tests.

## Next 1-3 Actions

- On the lab RTX 3090 server, generate the full LLaMA-3.1-8B tensor-native artifact set with `python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.llama_bitplanes --model meta-llama/Llama-3.1-8B --artifact-format safetensors --max-elements-per-tensor 0 --require-full-runtime-coverage --output-dir runs/llama31_8b_full_tensor_bitplanes --overwrite --print-json`.
- Export real HellaSwag router rows with `python -m qaq.data export-router-jsonl --dataset hellaswag --output runs/llama_first_milestone/router/hellaswag_router_train.jsonl --overwrite`, then run `python scripts/gpu_run.py --count 2 --min-free-mb 22000 --status-file runs/gpu-selector/hellaswag-router-train-full.json -- python -m qaq.router.train --config configs/router_train_llama31_8b_full_hellaswag.yaml` on the lab RTX 3090 server.
- Validate the generated `runtime_artifact_index.json` through the static/adaptive runtime paths before running any benchmark matrix.
- Build the first-milestone LLaMA mode matrix only after full artifacts and runtime metadata show `artifact_ref_mode: full_tensor_index`, `mixed_precision_forward_applied: true` for quantized modes, and the QAQ configs can load `runs/llama_first_milestone/router/checkpoints/router_final.json`.

## Evidence Required Before Claiming Done

- `tests/integration/test_llama_bitplane_generation.py` passes and asserts the current intended runtime-index shape.
- Generated runtime metadata distinguishes `full_tensor_index`, `partial_tensor_index`, sampled/truncated diagnostic probes, and non-accepted partial coverage correctly.
- Incomplete full-runtime requests fail with `incomplete_tensor_artifact_index`.
- Truncated or sampled artifacts continue to record that they are not accepted full quantized inference artifacts.
- The focused command `python -m pytest -q tests/integration/test_llama_bitplane_generation.py tests/integration/test_mixed_weight_runtime.py tests/unit/test_results_schema.py` passes.
- No large LLaMA weights were loaded locally; any real LLaMA generation/evaluation run must be launched on the lab RTX 3090 server through `scripts/gpu_run.py` and record selected physical GPU IDs.

## Risks / Residual Risk

- The router objective is an implementation assumption: `router_cost_cross_entropy` estimates quantized-student behavior from reconstruction distortion and a bit-cost term, not from an official QAQ loss.
- The full-tensor weight override path has tiny-model evidence only; it has not been verified as accepted LLaMA-3.1-8B execution.
- Sampled LLaMA artifacts and one-step sampled LLaMA router training are useful integration evidence but cannot support accepted benchmark claims.
- Optional model-loading dependencies are environment-provided rather than declared in `pyproject.toml`.
- Documentation has some stale historical notes: the experiment report still contains an output-directory sentence saying no LLaMA result artifact was produced, while later sections say the follow-up evaluation now writes one.
- Full paper reproduction remains out of scope until LLaMA first-milestone evidence is accepted and Qwen plus full benchmark coverage are added.
