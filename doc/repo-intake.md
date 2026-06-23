# Repository Intake

## Stack

- Language: Python 3.12.
- Project type: QAQ research prototype for query-adaptive mixed-precision LLM inference.
- Packaging/build metadata: `pyproject.toml` uses `setuptools.build_meta`.
- Test runner: pytest, configured in `pyproject.toml` with `testpaths = ["tests"]`, `pythonpath = ["."]`, and `addopts = "-ra"`.
- Declared runtime dependencies: none. `pyproject.toml` has `dependencies = []`.
- Optional runtime/test dependencies used by code and tests but not declared in project metadata:
  - `torch`
  - `transformers`
  - `safetensors`
  - `huggingface_hub`
  - `pytest`
- Repository style: dependency-light stdlib code with optional Hugging Face/PyTorch paths for real LLaMA metadata, weights, tensor-native artifacts, CUDA materialization, and router training preflight.

## Entry Points

- `python -m qaq.config <config>` validates JSON or TOML QAQ run configs.
- `python -m qaq.evaluate --config <config>` runs static or adaptive evaluation depending on the config mode.
- `python -m qaq.router.train --config <config>` runs non-diagnostic router training from JSON, TOML, or the repo's small YAML subset.
- `python -m qaq.router.train --health-check` runs the quick diagnostic router-training health check.
- `python -m qaq.report --results <result.json> ...` builds a comparison report from result artifacts.
- `python -m qaq.prepare_bitplanes --model <model> --output-dir <dir>` prepares sampled JSON bit-plane artifacts from local Hugging Face safetensors.
- `python -m qaq.llama_bitplanes --model <model> --output-dir <dir>` generates LLaMA bit-plane artifacts as JSON or tensor-native `.qaq.safetensors`.
- Tests are run through `python -m pytest`.

## Install Commands

- No canonical install command is documented in `README.md`; it only contains the project title.
- Inferred editable install command:

```bash
python -m pip install -e .
```

- Test and optional real-model commands also require installing undeclared packages such as `pytest`, `torch`, `transformers`, `safetensors`, and `huggingface_hub`.
- No lockfile, requirements file, uv file, Poetry file, Conda environment, or Dockerfile was found.

## Build Commands

- No required build command is documented.
- Inferred package build command, if the `build` package is available:

```bash
python -m build
```

- There is no Makefile, justfile, tox config, nox config, or CI workflow in the inspected tree.
- Build/lint/format/type project gates remain unchecked in `doc/tasks/progress.md`.

## Run Commands

Config validation:

```bash
python -m qaq.config configs/smoke.json --skip-output-dir-check --print-json
```

Smoke/fixture evaluation:

```bash
python -m qaq.evaluate --config configs/smoke.json --skip-output-dir-check --print-json
```

Router diagnostic health check:

```bash
python -m qaq.router.train --health-check
```

Local non-diagnostic router-training acceptance path:

```bash
python -m qaq.router.train --config configs/router_train_real.yaml
```

Checkpoint-loaded router evaluation path after the local acceptance checkpoint exists:

```bash
python -m qaq.evaluate --config configs/router_eval_real.json --artifact-index configs/router_eval_real_artifacts.json --skip-output-dir-check --print-json
```

Report generation from an existing golden result artifact:

```bash
python -m qaq.report --results tests/golden/result_artifact_static.json --print-json
```

Sampled real-weight LLaMA bit-plane artifact preparation:

```bash
python -m qaq.prepare_bitplanes --model meta-llama/Llama-3.1-8B --output-dir runs/llama31_8b_bitplanes_sampled --sample-values 16 --overwrite --print-json
```

Tensor-native LLaMA bit-plane artifact probe:

```bash
python -m qaq.llama_bitplanes --model meta-llama/Llama-3.1-8B --artifact-format safetensors --output-dir runs/llama31_8b_native_bitplanes_probe --block-limit 1 --tensor-limit-per-block 2 --max-elements-per-tensor 16 --overwrite --print-json
```

LLaMA 3.1 8B sampled-artifact router training command:

```bash
python -m qaq.router.train --config configs/router_train_llama31_8b_sampled.yaml
```

The LLaMA router-training command is hardware/model dependent. `doc/router-training.md` records that the current local RTX 4050 path fails CUDA preflight with insufficient memory, while the user has RTX 3090s on the lab server.

## Test Commands

Full suite:

```bash
python -m pytest -q
```

Focused suites:

```bash
python -m pytest -q tests/unit
python -m pytest -q tests/integration
python -m pytest -q tests/e2e
python -m pytest -q tests/regression
```

Targeted checks referenced by repo docs:

```bash
python -m pytest -q tests/integration/test_tensor_bitplane_artifacts.py tests/integration/test_llama_bitplane_generation.py tests/integration/test_router_checkpoint_contract.py
python -m qaq.config configs/smoke.json --skip-output-dir-check --print-json
python -m qaq.evaluate --config configs/smoke.json --skip-output-dir-check --print-json
python -m qaq.router.train --health-check
```

Tests that require optional packages use `pytest.importorskip` for `torch` and `safetensors.torch`; CUDA-specific checks skip when CUDA is unavailable.

No tests, evaluations, installs, or training commands were run during this intake because the request limited work to inspection plus this documentation update.

## Existing Modules

- `qaq/config.py`: run config dataclasses, JSON/TOML parsing, config validation CLI.
- `qaq/errors.py`: shared custom error model.
- `qaq/manifest.py`: run manifest creation and serialization.
- `qaq/logging.py`, `qaq/progress.py`, `qaq/status.py`: JSONL logs, progress events, run status enums.
- `qaq/data.py`, `qaq/benchmark_adapter.py`: benchmark/sample loading and tokenized batch construction.
- `qaq/model_adapter.py`: fake/local adapters plus optional Hugging Face LLaMA adapter and tokenizer/model metadata support.
- `qaq/blocks.py`, `qaq/precision_plan.py`: MHA/FFN block discovery and precision-plan construction.
- `qaq/quantization.py`, `qaq/bitplanes.py`, `qaq/artifacts.py`: small dependency-free quantization, bit-plane decomposition/reconstruction, JSON artifacts.
- `qaq/tensor_bitplanes.py`: tensor-native `.qaq.safetensors` bit-plane artifact creation, loading, distortion, and materialization helpers.
- `qaq/prepare_bitplanes.py`: sampled real-weight bit-plane artifact preparation from local safetensors.
- `qaq/llama_bitplanes.py`: LLaMA safetensors streaming artifact generator for JSON and tensor-native formats.
- `qaq/router/types.py`: router dataclasses.
- `qaq/router/policy.py`: router scoring, normalization, selection, checkpoint compatibility.
- `qaq/router/losses.py`: router loss records and `router_cost_cross_entropy`.
- `qaq/router/checkpoint.py`: router checkpoint save/load and validation.
- `qaq/router/train.py`: router-training config parser, preflight, health check, real local training path, checkpointing, validation metrics.
- `qaq/runtime/static.py`: static/fixed precision runtime path and artifact index loading.
- `qaq/runtime/adaptive.py`: QAQ adaptive runtime path, router checkpoint loading, route summaries, metrics, and on-demand mode integration.
- `qaq/runtime/loader.py`, `qaq/loader.py`: on-demand materialization for JSON and tensor-native bit-plane artifacts, including CUDA checks.
- `qaq/metrics.py`: score, latency, memory, and routing metric helpers.
- `qaq/results.py`: result artifact construction, comparison validation, paper-coverage guards.
- `qaq/report.py`: comparison report CLI.
- `qaq/evaluate.py`: evaluation CLI dispatching static versus adaptive runtime.

## Data Files

- `QAQ.pdf`: local paper source of truth.
- `tests/fixtures/benchmarks/fake_smoke.jsonl`: fake smoke benchmark fixture.
- `tests/fixtures/benchmarks/router_training_real.jsonl`: file-backed local router-training sample fixture.
- `tests/fixtures/bitplanes/router_training_real/*.json`: local router-training bit-plane artifacts.
- `tests/fixtures/models/router_local_model.json`: local fixture model metadata.
- `tests/fixtures/tokenizers/router_local_tokenizer.json`: local fixture tokenizer metadata.
- `tests/fixtures/configs/*.json`: config validation fixtures.
- `tests/golden/*.json`: golden bit-plane, router decision, report row, and result artifacts.
- `configs/router_eval_real_artifacts.json`: artifact index for local checkpoint-loaded router evaluation.
- Generated run outputs go under `runs/`, which is ignored by `.gitignore`.
- Generated Python cache files are present under `qaq/**/__pycache__` and `tests/**/__pycache__`; these are generated files and should not be committed.
- LLaMA model weights and Hugging Face caches are external data, not repository files.

## Config Files

- `pyproject.toml`: package metadata and pytest configuration.
- `.gitignore`: ignores bytecode, pytest/cache/build outputs, coverage, and `runs/`.
- `AGENTS.md`: project-specific working agreement, validation commands, and anti-smoke-test rules.
- `configs/smoke.json`: fake/local smoke evaluation config.
- `configs/smoke.toml`: fake/local smoke TOML variant.
- `configs/router_train_smoke.yaml`: diagnostic router health/config path.
- `configs/router_train_real.yaml`: local non-diagnostic router-training acceptance config.
- `configs/router_eval_real.json`: local router evaluation config that expects the trained checkpoint under `runs/router_train_real`.
- `configs/router_eval_real_artifacts.json`: artifact index for that local router evaluation.
- `configs/router_train_llama31_8b_sampled.yaml`: LLaMA 3.1 8B sampled-artifact router training config; hardware/model dependent.
- `configs/llama31_8b_first_milestone.json`: LLaMA first-milestone config marked with a stub note.
- `configs/llama31_8b_first_milestone.toml`: LLaMA first-milestone TOML config requiring external model/artifacts/router checkpoint.
- `configs/README.md`: minimal config directory note.
- `doc/*.md` and `doc/tasks/*.md`: source-of-truth planning, design, progress, risk, and task status docs.

## Known Broken Parts

- `README.md` is effectively empty and does not document setup, dependencies, commands, or scope.
- Optional packages used by real-model and tensor-native paths are not declared in `pyproject.toml`.
- No lockfile or reproducible environment file exists.
- No CI configuration was found.
- Build, lint, format, and type-check gates remain unchecked in `doc/tasks/progress.md`.
- Most project files are currently untracked according to `git status --short --untracked-files=all`; only `.gitignore` is tracked as modified in the current worktree view.
- `doc/test-plan.md` still contains TODOs for final paper-aligned reproduction, GPU memory/latency benchmark commands, install command confirmation, and final reproduction report command.
- Full paper-scale reproduction is not done. Current accepted local router training uses small dependency-free fixtures, not LLaMA/Qwen benchmark-scale evidence.
- `router_cost_cross_entropy` is a documented implementation assumption because the official QAQ router loss, calibration corpus, and hyperparameters are not specified in the available paper/code.
- The current router objective estimates quantized-student behavior using bit-plane reconstruction distortion rather than executing a full quantized transformer block.
- Full LLaMA 3.1 8B router training is blocked locally by the visible 6 GiB RTX 4050. The documented preflight expects at least 30.42 GiB free before activations for the current separate teacher/student BF16 loading path.
- `configs/router_train_llama31_8b_sampled.yaml` is not expected to pass on the local RTX 4050; it needs the lab RTX 3090 setup or a changed shared/sequential teacher-student execution path.
- `configs/router_eval_real.json` is not standalone on a clean checkout until `python -m qaq.router.train --config configs/router_train_real.yaml` has produced the checkpoint under `runs/router_train_real`.
- `configs/llama31_8b_first_milestone.*` require external LLaMA access, CUDA, artifacts, and router checkpoints; they are not self-contained smoke configs.
- CUDA on-demand materialization exists, but GPU memory and transfer claims still need a full QAQ runtime path that applies materialized tensors to the model on the intended hardware.
- `doc/test-plan.md` has stale language saying no implementation/package/test tree exists; the repository now has `qaq/`, `tests/`, `configs/`, and `pyproject.toml`.

## External Dependencies

- Python 3.12.
- `setuptools` for packaging.
- `pytest` for tests.
- Optional `torch` for Hugging Face model execution, tensor-native bit-plane artifact handling, CUDA checks, CUDA materialization, and some integration tests.
- Optional `transformers` for Hugging Face LLaMA config/tokenizer/model loading.
- Optional `safetensors` for local HF weight reading and `.qaq.safetensors` artifact I/O.
- Optional `huggingface_hub` for resolving/downloading local Hugging Face snapshots when needed.
- Hugging Face access/license and local model files for `meta-llama/Llama-3.1-8B`.
- CUDA-capable NVIDIA GPU for real LLaMA training/evaluation paths. The local machine has an RTX 4050; the user's lab server has RTX 3090s.
- Full target datasets remain external: HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande, WikiText-2, and PTB.

## What Not To Touch

- Do not modify `QAQ.pdf`; treat it as source material.
- Do not delete or rewrite `tests/golden/*.json` unless intentionally updating golden expectations with a matching test rationale.
- Do not delete fixture data under `tests/fixtures/`; tests and local acceptance configs depend on it.
- Do not commit generated run outputs under `runs/`.
- Do not commit generated `__pycache__` directories or pytest/cache/build outputs.
- Do not hardcode local Hugging Face cache paths, private model paths, API tokens, or lab-server paths.
- Do not silently weaken acceptance guards that prevent smoke/fake/diagnostic paths from being treated as final QAQ evidence.
- Do not change LLaMA/Qwen target config semantics without updating `doc/requirements.md`, `doc/router-training.md`, `doc/residual-risk.md`, and relevant tests.
- Do not replace real-model or real-data failures with fake fallbacks; fail clearly or label paths as diagnostic.
- Do not overwrite existing output directories unless the config or CLI explicitly sets overwrite or skips the output-dir check for a health/smoke run.

## Open Questions

- What exact dependency management approach should be used: requirements file, uv, Poetry, Conda, Docker, or documented manual install?
- Should optional dependencies become declared extras, for example `qaq[dev]`, `qaq[hf]`, and `qaq[cuda]`?
- What are the exact lab-server paths, CUDA versions, driver versions, and model-cache locations for the RTX 3090 environment?
- Should LLaMA router training be changed to shared/sequential teacher-student execution so it can fit a single 24 GiB RTX 3090?
- What is the accepted official or closest-defensible QAQ router objective if more paper details become available?
- Which real calibration corpus and split should replace the small local router-training fixture for first-milestone evidence?
- Which command should be the canonical first-milestone LLaMA evaluation once artifacts and checkpoint paths are available?
- What lint, format, type-check, and CI gates should be adopted?
- Should the currently untracked scaffold be committed as a baseline before further implementation work?
- Setup summary: install the package editably, then install pytest plus optional Hugging Face/PyTorch dependencies for real-model paths.
- Run summary: use `python -m qaq.config`, `python -m qaq.evaluate`, `python -m qaq.router.train`, `python -m qaq.prepare_bitplanes`, `python -m qaq.llama_bitplanes`, and `python -m qaq.report`.
- Test summary: run `python -m pytest -q`, with focused suites under `tests/unit`, `tests/integration`, `tests/e2e`, and `tests/regression`.
- Still unknown: reproducible dependency set, lab GPU environment details, paper-scale artifact generation cost, and full LLaMA/Qwen benchmark acceptance commands.
