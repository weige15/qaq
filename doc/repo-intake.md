# Repository Intake

## Stack

- Main language/runtime: Python 3.12. Evidence: `pyproject.toml` declares `requires-python = ">=3.12"`.
- Project type: QAQ research prototype scaffold for query-adaptive mixed-precision LLM inference with bit-plane artifacts, router decisions, static baselines, and optional on-demand loading. Evidence: `AGENTS.md`, `doc/high-level-design.md`, and `doc/detailed-design.md`.
- Packaging/build backend: setuptools. Evidence: `pyproject.toml` declares `build-system.requires = ["setuptools>=68"]` and `build-backend = "setuptools.build_meta"`.
- Package manager: no canonical package manager is documented. Evidence: `README.md` only contains `# qaq`, `pyproject.toml` has no dependency groups, and inspection found no `requirements.txt`, `uv.lock`, `poetry.lock`, `Pipfile`, `environment.yml`, `tox.ini`, `noxfile.py`, `Makefile`, `Dockerfile`, or CI workflow.
- Runtime dependencies: none declared for the installed package. Evidence: `pyproject.toml` has `dependencies = []`.
- Test framework: pytest. Evidence: `pyproject.toml` configures `[tool.pytest.ini_options]`, and `AGENTS.md` documents `python -m pytest -q`.
- Optional real-model stack: PyTorch, Hugging Face Transformers, safetensors, huggingface_hub, CUDA, and `nvidia-smi`. Evidence: optional imports in `qaq/model_adapter.py`, `qaq/tensor_bitplanes.py`, `qaq/prepare_bitplanes.py`, `qaq/llama_bitplanes.py`, `qaq/runtime/adaptive.py`, `qaq/runtime/loader.py`, and the GPU policy in `AGENTS.md`.
- Research source of truth: `QAQ.pdf` and `doc/`. Evidence: `AGENTS.md` explicitly says to treat `QAQ.pdf` and documents under `doc/` as the research-intent source of truth.

## Entry Points

- `qaq/config.py`: validates JSON or TOML run configs. Evidence: `qaq/config.py` defines `ArgumentParser(description="Validate a QAQ run configuration.")`, a `main()` function, and `if __name__ == "__main__"`.
- `qaq/evaluate.py`: dispatches static/fixed or adaptive QAQ runtime checks and can emit result artifacts. Evidence: `qaq/evaluate.py` defines `--config`, `--artifact-index`, `--print-json`, `--result-output`, and `--print-result-json`.
- `qaq/router/train.py`: trains the router from a config or runs a diagnostic health check. Evidence: `qaq/router/train.py` defines mutually exclusive `--config` and `--health-check`.
- `qaq/report.py`: builds comparison reports from result artifact JSON files. Evidence: `qaq/report.py` defines `--results`, `--output`, and `--print-json`.
- `qaq/prepare_bitplanes.py`: prepares sampled QAQ bit-plane artifacts from local Hugging Face safetensors. Evidence: `qaq/prepare_bitplanes.py` defines `--model`, `--output-dir`, `--sample-values`, `--block-limit`, `--overwrite`, and `--print-json`.
- `qaq/llama_bitplanes.py`: generates LLaMA JSON or tensor-native `.qaq.safetensors` bit-plane artifacts. Evidence: `qaq/llama_bitplanes.py` defines `--artifact-format`, `--max-elements-per-tensor`, `--allow-full-tensor-json`, `--overwrite`, and `--print-json`.
- `scripts/gpu_run.py`: required wrapper for heavy ML commands on the lab GPU server. Evidence: `scripts/gpu_run.py` defines `--count`, `--min-free-mb`, `--physical-ids`, `--gpu-name-contains`, `--status-file`, `--dry-run`, and sets `CUDA_VISIBLE_DEVICES` for the child command.
- No package console scripts, web server entry point, Make target, Docker command, notebook entry point, or CI job was found. Evidence: `pyproject.toml`, top-level listing, and `rg --files`.

## Install Commands

- Documented install command: Unknown from repository files. Checked `README.md`, `pyproject.toml`, top-level files, and common dependency/environment files.
- Inferred editable install from `pyproject.toml`:

```bash
# cwd: /nfs/home/s314511048/qaq
python -m pip install -e .
```

- Inferred test dependency install is needed because tests import pytest while the package declares no dependencies. Evidence: `pyproject.toml` has `dependencies = []` and tests import `pytest`.
- Optional real-model dependencies must be installed separately if using LLaMA, CUDA, tensor-native artifacts, or Hugging Face paths. Evidence: optional imports of `torch`, `transformers`, `safetensors`, and `huggingface_hub` in `qaq/`.
- Warning: optional dependency installation can be network-dependent and environment-specific; no lockfile or pinned environment file exists.

## Build Commands

- Documented build command: Unknown from repository files. Checked `README.md`, `pyproject.toml`, top-level files, Makefile, Dockerfile, tox, nox, and CI locations.
- Inferred source/wheel build, only if the `build` frontend is installed:

```bash
# cwd: /nfs/home/s314511048/qaq
python -m build
```

- Build, lint, format, and static type-analysis gates are not configured as repository commands. Evidence: `doc/tasks/progress.md` lists build, lint, format, and type/static analysis gates as unchecked, and no Makefile or CI workflow was found.

## Run Commands

- Documented config validation command. Evidence: `AGENTS.md` and `qaq/config.py`:

```bash
# cwd: /nfs/home/s314511048/qaq
python -m qaq.config configs/smoke.json --skip-output-dir-check --print-json
```

- Documented fake/local smoke evaluation command. Evidence: `AGENTS.md` and `qaq/evaluate.py`:

```bash
# cwd: /nfs/home/s314511048/qaq
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.evaluate --config configs/smoke.json --skip-output-dir-check --print-json
```

- Documented diagnostic router-training health check. Evidence: `AGENTS.md`, `doc/router-training.md`, and `qaq/router/train.py`:

```bash
# cwd: /nfs/home/s314511048/qaq
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.router.train --health-check
```

- Documented non-diagnostic local-fixture router-training command. Evidence: `doc/router-training.md` and `configs/router_train_real.yaml`:

```bash
# cwd: /nfs/home/s314511048/qaq
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.router.train --config configs/router_train_real.yaml
```

- Documented checkpoint-loaded router evaluation command, after the router checkpoint exists. Evidence: `doc/router-training.md`, `configs/router_eval_real.json`, and `configs/router_eval_real_artifacts.json`:

```bash
# cwd: /nfs/home/s314511048/qaq
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.evaluate --config configs/router_eval_real.json --artifact-index configs/router_eval_real_artifacts.json --skip-output-dir-check --print-json
```

- Documented report command. Evidence: `qaq/report.py` and `doc/tasks/progress.md`:

```bash
# cwd: /nfs/home/s314511048/qaq
python -m qaq.report --results tests/golden/result_artifact_static.json --print-json
```

- Documented sampled LLaMA bit-plane preparation command. Evidence: `doc/router-training.md` and `qaq/prepare_bitplanes.py`:

```bash
# cwd: /nfs/home/s314511048/qaq
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.prepare_bitplanes --model meta-llama/Llama-3.1-8B --output-dir runs/llama31_8b_bitplanes_sampled --sample-values 16 --overwrite --print-json
```

- Documented LLaMA tensor-native artifact probe. Evidence: `doc/llama-bitplanes.md` and `qaq/llama_bitplanes.py`:

```bash
# cwd: /nfs/home/s314511048/qaq
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.llama_bitplanes --model meta-llama/Llama-3.1-8B --artifact-format safetensors --output-dir runs/llama31_8b_native_bitplanes_probe --block-limit 1 --tensor-limit-per-block 2 --max-elements-per-tensor 16 --overwrite --print-json
```

- Documented hardware-dependent LLaMA sampled-artifact router training command. Evidence: `doc/router-training.md` and `configs/router_train_llama31_8b_sampled.yaml`:

```bash
# cwd: /nfs/home/s314511048/qaq
python scripts/gpu_run.py --count 1 --min-free-mb 18000 -- python -m qaq.router.train --config configs/router_train_llama31_8b_sampled.yaml
```

- Verified real LLaMA-3.1-8B HellaSwag FP16 subset evaluation command. Evidence: `configs/benchmarks/llama_first_milestone/hellaswag/fp16.json`, `qaq/evaluate.py`, `qaq/results.py`, and `scripts/gpu_run.py`. This is real-path subset evidence and is rejected for full QAQ acceptance with `benchmark_subset_not_full_acceptance` until the full comparable five-mode matrix exists:

```bash
# cwd: /nfs/home/s314511048/qaq
python scripts/gpu_run.py --count 1 --min-free-mb 18000 --status-file runs/llama_first_milestone/hellaswag/fp16_subset/gpu_run_status.json -- python -m qaq.evaluate --config configs/benchmarks/llama_first_milestone/hellaswag/fp16.json --skip-output-dir-check --max-examples 128 --eval-batch-size 1 --hf-device-map single --result-output runs/llama_first_milestone/hellaswag/fp16_subset/result.json --print-result-json
```

- Warning: training, full inference, full evaluation, benchmarks, and large-model-loading commands are GPU-dependent and must run through `scripts/gpu_run.py` on the lab RTX 3090 server. Evidence: `AGENTS.md` ML Runtime Policy.
- Warning: the LLaMA sampled-artifact router training command is documented to fail on the visible local 6 GiB RTX 4050 because it needs at least 15.46 GiB free before activations. Evidence: `doc/router-training.md` and `doc/residual-risk.md`.

## Test Commands

- Documented full test suite. Evidence: `AGENTS.md`:

```bash
# cwd: /nfs/home/s314511048/qaq
python -m pytest -q
```

- Documented focused suites. Evidence: `AGENTS.md`:

```bash
# cwd: /nfs/home/s314511048/qaq
python -m pytest -q tests/unit
python -m pytest -q tests/integration
python -m pytest -q tests/e2e
```

- Inferred regression suite from repository layout. Evidence: `tests/regression/` exists:

```bash
# cwd: /nfs/home/s314511048/qaq
python -m pytest -q tests/regression
```

- Documented GPU wrapper unit test. Evidence: `doc/tasks/progress.md` and `tests/unit/test_gpu_run.py`:

```bash
# cwd: /nfs/home/s314511048/qaq
python -m pytest -q tests/unit/test_gpu_run.py
```

- Documented targeted tensor/native/router checks. Evidence: `doc/tasks/progress.md`:

```bash
# cwd: /nfs/home/s314511048/qaq
python -m pytest -q tests/integration/test_tensor_bitplane_artifacts.py tests/integration/test_llama_bitplane_generation.py tests/integration/test_router_checkpoint_contract.py
```

- Warning: optional-package tests may skip when `torch`, `safetensors`, or CUDA are unavailable. Evidence: `pytest.importorskip` and CUDA checks in `tests/integration/test_tensor_bitplane_artifacts.py`, `tests/integration/test_llama_bitplane_generation.py`, and `tests/integration/test_on_demand_loader_simulation.py`.
- No test suite was run during this intake; only read-only inspection commands and this documentation edit were performed.

## Existing Modules

- `qaq/config.py`: run config dataclasses, JSON/TOML loading, validation, output-directory checks, and config CLI. Evidence: `qaq/config.py` and `doc/tasks/experiment-configuration-and-run-manifest.md`.
- `qaq/errors.py`: shared categorized error model. Evidence: imports from validation/runtime modules.
- `qaq/manifest.py`: run manifest serialization and lifecycle state. Evidence: `qaq/manifest.py` and manifest references in `doc/tasks/progress.md`.
- `qaq/logging.py`, `qaq/progress.py`, `qaq/status.py`: structured JSONL events, progress state, timing helpers, and completion/failure status handling. Evidence: `doc/tasks/logging-and-progress-tracking.md`.
- `qaq/data.py` and `qaq/benchmark_adapter.py`: fake/local benchmark examples, prompt formatting, and tokenized batch construction. Evidence: `tests/integration/test_model_adapter_smoke.py`.
- `qaq/model_adapter.py`: dependency-free fake/local adapter plus optional Hugging Face LLaMA metadata/model loading hooks. Evidence: `qaq/model_adapter.py` imports `transformers` and `torch` only inside optional paths.
- `qaq/blocks.py` and `qaq/precision_plan.py`: MHA/FFN block discovery, stable block IDs, static/fixed/QAQ precision plans, and validation. Evidence: `tests/unit/test_block_registry.py`.
- `qaq/quantization.py`, `qaq/bitplanes.py`, and `qaq/artifacts.py`: small tensor quantization, MSB bit-plane decomposition/reconstruction, JSON artifact contracts, and checksums. Evidence: `tests/unit/test_bitplanes.py` and `tests/integration/test_quantized_artifact_roundtrip.py`.
- `qaq/tensor_bitplanes.py`: tensor-native `.qaq.safetensors` bit-plane artifacts, distortion helpers, and selected-plane materialization. Evidence: `doc/llama-bitplanes.md` and optional imports in `qaq/tensor_bitplanes.py`.
- `qaq/prepare_bitplanes.py`: sampled real-weight artifact preparation from local Hugging Face safetensors. Evidence: `doc/router-training.md`.
- `qaq/llama_bitplanes.py`: streaming LLaMA artifact generator for JSON and safetensors formats. Evidence: `doc/llama-bitplanes.md`.
- `qaq/router/types.py`, `qaq/router/policy.py`, `qaq/router/losses.py`, `qaq/router/checkpoint.py`, and `qaq/router/train.py`: router dataclasses, scoring/policy, loss records, checkpoint save/load, training config, preflight, health-check, and training CLI. Evidence: `doc/router-training.md`.
- `qaq/runtime/common.py`, `qaq/runtime/static.py`, `qaq/runtime/adaptive.py`, and `qaq/runtime/loader.py`: runtime output contracts, static/fixed runtime, adaptive QAQ runtime, and on-demand materialization for JSON/native artifacts. Evidence: `tests/e2e/test_smoke_modes.py` and `tests/integration/test_on_demand_loader_simulation.py`.
- `qaq/loader.py`: public loader-facing helper module. Evidence: source file and loader tests.
- `qaq/metrics.py`, `qaq/results.py`, and `qaq/report.py`: metric aggregation, result artifact schema, acceptance/comparison guards, and report CLI. Evidence: `tests/unit/test_results_schema.py` and `tests/regression/test_qaq_acceptance_guards.py`.
- `qaq/evaluate.py`: evaluation CLI dispatching static versus adaptive runtime and optional result artifact output. Evidence: `qaq/evaluate.py`.
- `scripts/gpu_run.py`: remote GPU selector/wrapper for ML commands. Evidence: `scripts/gpu_run.py`, `AGENTS.md`, and `tests/unit/test_gpu_run.py`.
- `tests/`: unit, integration, e2e, regression, fixture, and golden test directories. Evidence: `rg --files`.
- `doc/adr/`: Architecture Decision Records for staged prototype, bit-plane artifacts, block granularity, on-demand loading, adaptive routing separation, and baseline-comparable evaluation. Evidence: files under `doc/adr/`.
- `doc/tasks/`: task-level implementation/status docs for the major modules. Evidence: files under `doc/tasks/`.

## Data Files

- `QAQ.pdf`: paper/source material and should be treated as input. Evidence: top-level file and `AGENTS.md`.
- `tests/fixtures/benchmarks/fake_smoke.jsonl`: fake smoke benchmark fixture. Evidence: `rg --files`.
- `tests/fixtures/benchmarks/router_training_real.jsonl`: small file-backed router-training/evaluation fixture. Evidence: `configs/router_train_real.yaml` and `configs/router_eval_real.json`.
- `tests/fixtures/models/router_local_model.json` and `tests/fixtures/tokenizers/router_local_tokenizer.json`: local model/tokenizer metadata fixtures. Evidence: `configs/router_train_real.yaml`.
- `tests/fixtures/bitplanes/router_training_real/*.json`: checked-in bit-plane artifacts used by local router training/evaluation configs. Evidence: `configs/router_train_real.yaml` and `configs/router_eval_real_artifacts.json`.
- `tests/fixtures/configs/*.json`: config validation fixtures. Evidence: `rg --files`.
- `tests/golden/*.json`: golden bit-plane, result artifact, report row, and router-decision outputs. Evidence: `rg --files`.
- `configs/router_eval_real_artifacts.json`: artifact index mapping block IDs and bit-widths to fixture bit-plane artifacts. Evidence: file contents.
- `runs/router_train_llama31_8b_sampled/`: generated router-training outputs, including `manifest.json`, `logs/router_train.jsonl`, `checkpoints/router_step_0001.json`, and `router_targets.json`. Evidence: `find runs -maxdepth 3 -type f`.
- `runs/llama31_8b_bitplanes_sampled/`: generated sampled LLaMA bit-plane artifacts and `artifact_index.json`. Evidence: `find runs -maxdepth 3 -type f`.
- Generated outputs under `runs/` are ignored and should not be committed. Evidence: `.gitignore` contains `runs/`.
- Generated Python/cache/build outputs are ignored. Evidence: `.gitignore` lists `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `build/`, `dist/`, and `*.egg-info/`.
- LLaMA model weights, Hugging Face caches, and benchmark datasets are external and not checked in. Evidence: absence from `rg --files` plus references in `doc/router-training.md`, `doc/requirements.md`, and `doc/llama-bitplanes.md`.

## Config Files

- `pyproject.toml`: package metadata, setuptools build backend, Python version, and pytest config.
- `.gitignore`: ignore rules for caches, coverage, build outputs, and `runs/`.
- `AGENTS.md`: project-specific working agreement, validation commands, anti-smoke-test requirements, and local/remote ML runtime policy.
- `configs/README.md`: states checked-in smoke, first-milestone, and report configs belong under `configs/`.
- `configs/smoke.json`: fake/local CPU smoke config for `fp16`, `fake_smoke`, precision candidates `[4, 8]`, and `runs/smoke`.
- `configs/smoke.toml`: TOML smoke config for fake model/tokenizer and `toy_prompts`.
- `configs/router_train_smoke.yaml`: diagnostic router health/config path using fake smoke data and generated `runs/router_train_health` artifacts.
- `configs/router_train_real.yaml`: non-diagnostic local router-training config using checked-in fixture data, local model/tokenizer metadata, and fixture bit-plane artifacts.
- `configs/router_eval_real.json`: local router evaluation config expecting `runs/router_train_real/checkpoints/router_step_0003.json`.
- `configs/router_eval_real_artifacts.json`: artifact index for `configs/router_eval_real.json`.
- `configs/router_train_llama31_8b_sampled.yaml`: CUDA LLaMA sampled-artifact router-training config using `meta-llama/Llama-3.1-8B`, fixture prompts, and `runs/llama31_8b_bitplanes_sampled/artifact_index.json`.
- `configs/llama31_8b_first_milestone.json`: first-milestone LLaMA config stub for `fp16`, CUDA, `wikitext`, and perplexity; its `notes` field says runtime, datasets, artifacts, and model access are validated later.
- `configs/llama31_8b_first_milestone.toml`: first-milestone TOML config for `qaq_on_demand_off` expecting `runs/router/llama31_8b/router_checkpoint.json`.
- No `.env.example`, secret config, lockfile, Dockerfile, Makefile, CI workflow, tox, nox, pytest.ini, setup.py, or setup.cfg was found. Evidence: targeted `find` scan returned no matching files.

## Known Broken Parts

- `README.md` is effectively empty and does not document setup, dependencies, commands, or scope. Evidence: `README.md` contains only `# qaq`.
- Dependency management is incomplete. Evidence: `pyproject.toml` has `dependencies = []`, while tests and optional paths use `pytest`, `torch`, `transformers`, `safetensors`, and `huggingface_hub`.
- No reproducible environment file or lockfile exists. Evidence: no `requirements.txt`, `uv.lock`, `poetry.lock`, `Pipfile`, or `environment.yml` was found.
- No CI configuration was found. Evidence: `.github/` is absent from the repository file list.
- Build, lint, format, and type/static analysis gates remain unconfigured or unchecked. Evidence: `doc/tasks/progress.md`.
- `doc/test-plan.md` contains stale scaffold language saying the repository has no implementation, package manifest, source directory, test directory, scripts, fixtures, or CI configuration, even though `pyproject.toml`, `qaq/`, `tests/`, `scripts/`, `configs/`, and fixtures now exist.
- `doc/test-plan.md` still has TODOs for final paper-aligned reproduction, GPU memory/latency benchmark commands, install command confirmation, and final reproduction report command. Evidence: `rg -n "TODO" doc/test-plan.md`.
- Full paper-scale QAQ reproduction is not complete. Evidence: `AGENTS.md`, `doc/residual-risk.md`, and `doc/tasks/progress.md` distinguish local fixture/fake CPU evidence from LLaMA/Qwen benchmark-scale evidence.
- Local router-training acceptance uses small fixtures and is not paper-scale evidence. Evidence: `doc/router-training.md` and `doc/residual-risk.md`.
- The official QAQ router loss, calibration corpus, and hyperparameters remain unavailable. Evidence: `doc/router-training.md` and `doc/residual-risk.md`.
- The current `router_cost_cross_entropy` estimates quantized-student behavior partly from bit-plane reconstruction distortion rather than executing a full quantized transformer block. Evidence: `doc/router-training.md` and `doc/residual-risk.md`.
- Full LLaMA 3.1 8B router training is blocked locally by GPU capacity. Evidence: `doc/router-training.md` records `insufficient_cuda_memory` on the visible 6 GiB RTX 4050 and says lab RTX 3090-class memory is required.
- `configs/router_eval_real.json` is not standalone on a clean checkout until the router-training command creates `runs/router_train_real/checkpoints/router_step_0003.json`.
- `configs/llama31_8b_first_milestone.*` and `configs/router_train_llama31_8b_sampled.yaml` depend on external model access, generated artifacts, CUDA, and/or generated router checkpoints.
- CUDA on-demand materialization exists for selected JSON/native bit-plane tensors, but GPU memory and transfer claims still require a full QAQ runtime path applying materialized tensors to the model on intended RTX 3090 hardware. Evidence: `doc/residual-risk.md`.
- Working tree was already dirty before this intake edit. Evidence: `git status --short` showed modified `AGENTS.md`, `doc/llama-bitplanes.md`, `doc/repo-intake.md`, `doc/residual-risk.md`, `doc/router-training.md`, `doc/tasks/progress.md`, plus untracked `doc/debug-report.md`, `scripts/`, and `tests/unit/test_gpu_run.py`.

## External Dependencies

- Python `>=3.12`. Evidence: `pyproject.toml`.
- `setuptools>=68` for package build metadata. Evidence: `pyproject.toml`.
- `pytest` for tests. Evidence: `pyproject.toml` pytest config and test imports.
- Optional `torch` for CUDA checks, Hugging Face model execution, tensor-native bit-plane handling, CUDA materialization, and optional tests. Evidence: imports in `qaq/model_adapter.py`, `qaq/router/train.py`, `qaq/tensor_bitplanes.py`, `qaq/runtime/adaptive.py`, and `qaq/runtime/loader.py`.
- Optional `transformers` for Hugging Face LLaMA config/tokenizer/model loading. Evidence: `qaq/model_adapter.py`.
- Optional `safetensors` for local Hugging Face weight reads and `.qaq.safetensors` artifact I/O. Evidence: `qaq/prepare_bitplanes.py`, `qaq/llama_bitplanes.py`, and `qaq/tensor_bitplanes.py`.
- Optional `huggingface_hub` for resolving or downloading local Hugging Face snapshots. Evidence: `qaq/llama_bitplanes.py`.
- `nvidia-smi` for GPU selection. Evidence: `scripts/gpu_run.py`.
- CUDA-capable NVIDIA GPU for real LLaMA training/evaluation and GPU-memory claims. Evidence: `AGENTS.md`, `doc/router-training.md`, and `doc/residual-risk.md`.
- Remote lab RTX 3090 server for real ML workloads. Evidence: `AGENTS.md`.
- Hugging Face access/license and local model files for `meta-llama/Llama-3.1-8B`. Evidence: `doc/router-training.md`, `doc/llama-bitplanes.md`, and `configs/router_train_llama31_8b_sampled.yaml`.
- Paper target datasets are external: HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande, WikiText-2, and PTB. Evidence: `doc/requirements.md` and `doc/high-level-design.md`.

## What Not To Touch

- Do not modify `QAQ.pdf`; it is source material. Evidence: `AGENTS.md`.
- Do not delete or rewrite `tests/golden/*.json` unless intentionally updating golden expectations with matching tests. Evidence: golden files are used by result/report/bit-plane/router tests.
- Do not delete fixture data under `tests/fixtures/`; smoke, router-training, and validation tests depend on it. Evidence: `configs/router_train_real.yaml`, `configs/router_eval_real_artifacts.json`, and tests under `tests/`.
- Do not commit generated outputs under `runs/`. Evidence: `.gitignore`.
- Do not commit generated caches or build outputs such as `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `build/`, `dist/`, or `*.egg-info/`. Evidence: `.gitignore`.
- Do not modify `.git/`.
- Do not hardcode local Hugging Face cache paths, private model paths, API tokens, credentials, or lab-server paths. Evidence: `AGENTS.md`.
- Do not silently weaken acceptance guards that prevent fake, fixture, smoke, or diagnostic paths from being treated as final QAQ evidence. Evidence: `AGENTS.md` and `doc/adr/0006-require-baseline-comparable-evaluation-for-accepted-qaq-results.md`.
- Do not replace real-model, CUDA, dataset, or quantization failures with fake fallbacks; fail clearly or label runs diagnostic. Evidence: `AGENTS.md`.
- Do not run large local ML workloads on the local RTX 4050. Evidence: `AGENTS.md` ML Runtime Policy.
- Do not overwrite existing output directories unless config/CLI explicitly allows overwrite or uses a documented skip-output-dir check for smoke/health runs. Evidence: `AGENTS.md`, `qaq/config.py`, and `qaq/evaluate.py`.
- Do not treat sampled/truncated LLaMA artifacts as full quantized inference evidence. Evidence: `doc/router-training.md` and `doc/llama-bitplanes.md`.
- Do not change LLaMA/Qwen target semantics without updating the relevant requirements, router-training, residual-risk, and test documentation. Evidence: `doc/requirements.md`, `doc/router-training.md`, and `doc/residual-risk.md`.

## Open Questions

- What exact dependency-management approach should be adopted: requirements file, uv, Poetry, Conda, Docker, or package extras?
- Should optional dependencies become declared extras such as `qaq[dev]`, `qaq[hf]`, and `qaq[cuda]`?
- What are the lab-server SSH command, CUDA version, driver version, Python environment, model-cache paths, and dataset paths for the RTX 3090 environment?
- What free-memory margin is required for LLaMA 3.1 8B router training after activation and warm-up overhead are included?
- Which real calibration corpus and held-out split should replace the small local router-training fixture for first-milestone evidence?
- What is the closest defensible QAQ router objective if more paper details do not become available?
- Which command should become the canonical first-milestone LLaMA evaluation once artifact and checkpoint paths are available?
- What lint, format, type-check, and CI gates should be adopted?
- Should stale docs such as `doc/test-plan.md`, `doc/repo-map.md`, `doc/review.md`, and `doc/repair-plan.md` be refreshed before the next implementation task?
- Setup summary: from `/nfs/home/s314511048/qaq`, inferred setup is `python -m pip install -e .`, plus separate installation of pytest and optional Hugging Face/PyTorch/safetensors dependencies as needed.
- Run summary: use `python -m qaq.config` for config validation, `python -m qaq.evaluate` for runtime/evaluation checks, `python -m qaq.router.train` for router training, `python -m qaq.report` for reports, and artifact generators `python -m qaq.prepare_bitplanes` or `python -m qaq.llama_bitplanes` for bit-plane preparation.
- Test summary: documented primary test command is `python -m pytest -q`, with focused `tests/unit`, `tests/integration`, and `tests/e2e` commands in `AGENTS.md`; tests were not run during this intake.
- Still unknown summary: canonical dependency setup, CI/build/lint/type gates, full paper-scale evaluation commands, lab server environment details, real dataset paths, and paper-faithful router objective remain unresolved.
