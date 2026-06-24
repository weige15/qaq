# Current State

## Product Goal

QAQ is a Python 3.12 research prototype for query-adaptive mixed-precision LLM inference. The intended system decomposes LLM weights into bit-plane artifacts, trains a query-conditioned router for MHA/FFN block precision choices, compares against FP16/static 8-bit/static 4-bit baselines, and optionally loads only selected bit-planes from CPU to GPU on demand. `QAQ.pdf`, `doc/requirements.md`, `doc/high-level-design.md`, and `doc/detailed-design.md` define the research target.

## Current Milestone

The current milestone is no longer basic scaffolding. The repo has a working prototype, a small real LLaMA sampled-artifact workflow, and uncommitted per-tensor Hugging Face weight override work with tiny-model test evidence. The next milestone is to make the LLaMA artifact/index side real: full tensor artifact indexes, not sampled or first-tensor-per-block indexes, before any static or QAQ LLaMA result is treated as real quantized execution evidence.

## Implemented

- Config and manifest handling are implemented in `qaq/config.py` and `qaq/manifest.py`, with validation coverage in `tests/unit/test_config_validation.py` and documented status in `doc/tasks/progress.md`.
- Structured logging, progress tracking, completion/failure markers, and incomplete-run handling are implemented in `qaq/logging.py`, `qaq/progress.py`, and `qaq/status.py`, with unit/integration coverage in `tests/unit/test_logging_events.py` and `tests/integration/test_logging_and_incomplete_runs.py`.
- MHA/FFN block discovery and precision-plan validation are implemented in `qaq/blocks.py` and `qaq/precision_plan.py`, with coverage in `tests/unit/test_block_registry.py`.
- JSON bit-plane artifacts and tensor-native `.qaq.safetensors` artifacts are implemented in `qaq/bitplanes.py`, `qaq/quantization.py`, `qaq/artifacts.py`, and `qaq/tensor_bitplanes.py`, with round-trip and tensor-native tests under `tests/unit/test_bitplanes.py` and `tests/integration/`.
- Static/fixed runtime and adaptive QAQ runtime entry points exist in `qaq/runtime/static.py`, `qaq/runtime/adaptive.py`, and `qaq/evaluate.py`, with fake/small E2E coverage in `tests/e2e/test_smoke_modes.py`.
- Router policy, checkpoint contracts, and non-constant routing guards are implemented in `qaq/router/policy.py`, `qaq/router/checkpoint.py`, and `qaq/router/types.py`, with coverage in `tests/unit/test_router_policy.py`, `tests/integration/test_router_checkpoint_contract.py`, and `tests/regression/test_qaq_acceptance_guards.py`.
- Router training has a minimal real objective, `router_cost_cross_entropy`, in `qaq/router/train.py` and `qaq/router/losses.py`. It loads file-backed samples, validates student artifacts, freezes detectable base parameters, writes `router_targets.json`, updates router parameters, saves reloadable checkpoints, and records validation metrics. `configs/router_train_real.yaml` and `tests/integration/test_router_checkpoint_contract.py` provide the checked-in local-fixture acceptance path.
- Optional local Hugging Face LLaMA support exists in `qaq/model_adapter.py`: local snapshot resolution, tokenizer/model loading with `local_files_only`, LLaMA MHA/FFN metadata, reference forwards, hidden states, and target-token NLL losses when targets are present. The current change is uncommitted and covered only for target-loss behavior by `tests/integration/test_model_adapter_smoke.py`.
- Sampled real-weight LLaMA artifact preparation exists in `qaq/prepare_bitplanes.py`; LLaMA JSON/tensor-native artifact generation exists in `qaq/llama_bitplanes.py`. `doc/experiments/2026-06-24-qaq-experiment-report.md` records a 64-block sampled LLaMA artifact run.
- Result artifacts, metric aggregation hooks, report rows, comparison grouping, and acceptance guards are implemented in `qaq/results.py`, `qaq/metrics.py`, and `qaq/report.py`, with tests in `tests/unit/test_results_schema.py` and regression guards.
- `scripts/gpu_run.py` is implemented and unit-tested in `tests/unit/test_gpu_run.py`; it selects eligible physical RTX 3090 GPUs, records physical-to-logical CUDA mapping, and refuses unsuitable devices.

## Partially Implemented

- Uncommitted runtime work now adds `qaq/runtime/weight_overrides.py` and modifies `qaq/model_adapter.py`, `qaq/runtime/static.py`, `qaq/runtime/adaptive.py`, and `qaq/precision_plan.py` so Hugging Face adapters can temporarily apply reconstructed per-tensor bit-plane weights when every tensor in a controlled block has an artifact ref. A tiny CPU integration test now passes, but this is still not LLaMA or benchmark evidence.
- Static/adaptive runtimes fall back to reference-forward behavior unless `artifact_ref_mode` is `full_tensor_index` and the adapter reports `supports_weight_overrides`. `tests/integration/test_mixed_weight_runtime.py` proves the adaptive tiny-model path can change predictions under full tensor overrides, but sampled LLaMA artifact indexes still do not satisfy the full-tensor condition.
- CUDA on-demand materialization exists for selected JSON and tensor-native bit-plane tensors when `torch` can access CUDA, but accepted memory-savings claims still need a verified runtime that applies those tensors to model execution and compares against static baselines.
- LLaMA sampled router training completed on the lab RTX 3090 server, and the uncommitted target-token NLL adapter change allows the sampled checkpoint evaluation to write a result artifact. This is small real integration evidence, not accepted benchmark evidence.
- LLaMA bit-plane generation supports tensor-native artifacts and has a verified full single-tensor probe, but `qaq.llama_bitplanes` writes a runtime index using the first tensor per block. That is not yet a complete all-tensors-per-controlled-block inference artifact path.
- `qaq/results.py` and `tests/unit/test_results_schema.py` now have uncommitted reporting-side changes that reject non-diagnostic quantized comparison rows unless `mixed_precision_forward_applied` is true. That guard is useful, but it does not by itself prove the runtime path works.
- `configs/llama31_8b_first_milestone.json` and `.toml` are stubs. They do not define a complete comparable five-mode first-milestone matrix.
- `README.md` is only a one-line title, and `pyproject.toml` declares no dependencies even though optional LLaMA paths require installed packages such as `torch`, `transformers`, and `safetensors` in the experiment environment.

## Diagnostic Only / Fake Only

- `configs/smoke.json`, `configs/smoke.toml`, `configs/router_train_smoke.yaml`, built-in `fake_smoke`, fake model/tokenizer identifiers, and fixture-backed fake adapters are diagnostic/local health paths only.
- `python -m qaq.router.train --health-check` is explicitly diagnostic and may generate temporary fake artifacts.
- `fixed_mixed` is a diagnostic validation mode and does not replace required QAQ modes.
- Fake CPU static/adaptive mode matrices prove schema, routing, loader, and reporter guards only. They are not GPU memory, LLaMA, or QAQ performance evidence.
- `tests/fixtures/benchmarks/router_training_real.jsonl` is file-backed and non-diagnostic for local router plumbing, but it is still a tiny checked-in fixture, not a paper benchmark.
- Sampled LLaMA artifacts from `qaq.prepare_bitplanes` record `artifact_scope: sampled_weight_values`, `full_tensor_values_stored: false`, and `accepted_as_full_quantized_inference_artifact: false`.
- The sampled LLaMA checkpoint-loaded evaluation result is small-scale fixture evidence over sampled artifacts. It is not an accepted QAQ benchmark result.

## Not Yet Implemented

- Verified and accepted LLaMA full quantized transformer execution evidence. The uncommitted weight-override path has tiny-model evidence, but still needs full LLaMA tensor artifacts, lab-server run records, and comparable result artifacts.
- Full all-tensor, all-controlled-block LLaMA bit-plane artifact coverage suitable for accepted static/QAQ inference.
- A complete first-milestone LLaMA result matrix across `fp16`, `static_8bit`, `static_4bit`, `qaq_on_demand_off`, and `qaq_on_demand_on` on the same model, tokenizer, dataset split, prompt format, precision candidates, and metric.
- Real benchmark adapters or run configs for the full paper benchmark set: HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande, WikiText-2, and PTB.
- Accepted latency and peak GPU memory comparisons for static baselines, QAQ on-demand off, and QAQ on-demand on.
- Qwen3-4B and Qwen3-8B paths for full paper reproduction.
- CI, lint, format, and type-check gates. `pyproject.toml` configures pytest only.

## Blocked

- Local ML execution is blocked by policy: local RTX 4050 use is limited to editing, small CPU tests, lint/format/syntax checks, and tiny smoke checks. Heavy model loading, training, inference, evaluation, and benchmarks must run on the lab RTX 3090 server through `scripts/gpu_run.py`.
- Accepted QAQ benchmark claims are blocked until full artifacts, verified quantized runtime use, real held-out benchmark data, all required static/QAQ modes, and comparable result artifacts exist.
- Paper-scale reproduction is blocked until the LLaMA first-milestone path is real and comparable, then Qwen3-4B/Qwen3-8B and the full benchmark suite are added or deviations are explicitly labeled.
- There is no hard external blocker for the next verification task. The next work is project work: test and verify the full-tensor weight-override runtime path, then build the first-milestone LLaMA matrix on top of it.

## Current Quality Gates

- Documented full suite result: `python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m pytest -q` passed with `126 passed in 9.32s` in `doc/tasks/progress.md`.
- Documented targeted result: `python -m pytest -q tests/integration/test_model_adapter_smoke.py` passed with `6 passed` after the Hugging Face target-token NLL change.
- Documented GPU-wrapped sampled LLaMA evaluation now writes `runs/router_train_llama31_8b_sampled/eval_qaq_on_demand_off/result_artifact.json`, but that result remains non-accepted fixture evidence.
- Current worktree is dirty: `doc/tasks/progress.md`, `qaq/model_adapter.py`, `qaq/precision_plan.py`, `qaq/results.py`, `qaq/runtime/adaptive.py`, `qaq/runtime/static.py`, `tests/integration/test_model_adapter_smoke.py`, and `tests/unit/test_results_schema.py` are modified; `doc/current-state.md`, `doc/experiments/`, `qaq/runtime/weight_overrides.py`, and `tests/integration/test_mixed_weight_runtime.py` are untracked.
- Ran one lightweight CPU test command in this audit pass: `python -m pytest -q tests/integration/test_mixed_weight_runtime.py tests/unit/test_results_schema.py`, which passed with `14 passed in 2.77s`. No router training, LLaMA loading, inference, benchmark runs, lint, format, or type checks were run.
- No configured lint, format, type/static analysis, or CI gate was found in `pyproject.toml`.
- Focused tiny-model runtime evidence now exists for applying `weight_overrides`, and result-schema evidence now exists for rejecting quantized comparison rows that lack mixed-weight-forward metadata. This remains small CPU evidence only.

## Last Verified Commands

- `pwd`: confirmed repository root `/nfs/home/s314511048/qaq` after rerunning outside the failed sandbox wrapper.
- `rg --files`: inventoried source, docs, configs, tests, scripts, and fixtures.
- `git status --short`: showed modified docs/runtime/model-adapter/test files plus untracked `doc/current-state.md`, `doc/experiments/`, and `qaq/runtime/weight_overrides.py`.
- `git log --oneline -n 10` and `git rev-parse HEAD`: current commit is `d5bd3ce0eb205c58a4cc93001ff63420ab6672f7`.
- `git diff --stat` and targeted `git diff` reads: confirmed the uncommitted adapter change adds Hugging Face target-token NLL losses, the runtime diffs add full-tensor weight-override plumbing, and result comparison now guards non-diagnostic quantized rows without mixed-weight-forward metadata.
- `hostname`: confirmed this audit ran on `basic-2`.
- `cat`/`sed`/`rg` reads over `AGENTS.md`, `README.md`, `pyproject.toml`, `configs/`, `tests/`, `doc/tasks/progress.md`, `doc/residual-risk.md`, `doc/router-training.md`, design docs, latest experiment report, and representative runtime/router/result source files.
- `python -m pytest -q tests/integration/test_mixed_weight_runtime.py tests/unit/test_results_schema.py`: passed with `14 passed in 2.77s`, proving the tiny-model adaptive override path and result-schema mixed-forward guard.

## Next 1-3 Actions

- Edit `qaq/llama_bitplanes.py` and/or `qaq/prepare_bitplanes.py` so a LLaMA artifact run can emit a runtime-usable full tensor artifact index for every tensor in each controlled MHA/FFN block, instead of the current sampled or first-tensor-per-block runtime index.
- Add/update tests around `tests/integration/test_llama_bitplane_generation.py` and `tests/integration/test_mixed_weight_runtime.py`, then run `python -m pytest -q tests/integration/test_llama_bitplane_generation.py tests/integration/test_mixed_weight_runtime.py tests/unit/test_results_schema.py`.
- Recommended next task: make LLaMA full-tensor artifact/index generation compatible with the verified weight-override runtime path.

## Evidence Required Before Claiming Done

- For the recommended next task: generated LLaMA artifact metadata must show every controlled block has artifact refs for all owned tensor names, not only bit-width keys or the first tensor per block.
- The generated index must make `artifact_ref_mode` evaluate to `full_tensor_index`, and incomplete tensor indexes must fail with a clear `missing_tensor_artifact` or equivalent error.
- Artifact manifests must distinguish full tensor-native artifacts from sampled/truncated probes; sampled artifacts with `accepted_as_full_quantized_inference_artifact: false` cannot satisfy this task.
- The focused pytest command above must pass, and the run must not load large LLaMA weights locally unless launched through `scripts/gpu_run.py` on the lab RTX 3090 server.
- After this task, the next evidence step is checked-in or generated LLaMA matrix configs for `fp16`, `static_8bit`, `static_4bit`, `qaq_on_demand_off`, and `qaq_on_demand_on` sharing the same comparison key and validated with `python -m qaq.config <config> --skip-output-dir-check --print-json`.
- For an accepted QAQ claim beyond the next task: full LLaMA artifacts, real held-out benchmark data, per-mode result artifacts with `mixed_precision_forward_applied: true` for quantized modes, `python -m qaq.report --results <all-five-result-artifacts> --print-json`, and GPU-selector records are required.

## Risks / Residual Risk

- The router objective is an implementation assumption. `router_cost_cross_entropy` estimates quantized-student behavior from reconstruction distortion and a bit-cost term, not from an official QAQ loss.
- The new weight-override path is uncommitted and only verified on a tiny CPU model in this audit, not on LLaMA.
- Sampled LLaMA artifacts and first-tensor runtime indexes can validate plumbing but cannot support accepted full-model inference claims.
- Optional model-loading dependencies are not declared in `pyproject.toml`, and README/run instructions are minimal.
- Documentation has a minor stale/conflicting note: the latest experiment report states the sampled LLaMA evaluation blocker was fixed and produced a result artifact, but its output-directory section still says no required LLaMA result artifact was produced.
- Full paper reproduction remains out of scope until LLaMA first-milestone evidence is accepted and Qwen plus full benchmark coverage are added.
