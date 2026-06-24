# Agent Instructions

## Repository Context

This repository is a Python 3.12 research prototype scaffold for QAQ:
query-adaptive mixed-precision LLM inference with bit-plane artifacts,
query-conditioned router decisions, static baselines, and optional on-demand
loading.

Treat `QAQ.pdf` and the documents under `doc/` as the source of truth for
research intent. Start with:

- `doc/high-level-design.md`
- `doc/detailed-design.md`
- `doc/requirements.md`
- `doc/test-plan.md`
- `doc/tasks/progress.md`

Current code is intentionally dependency-free and mostly exercises fake/local
model adapters, small tensors, and CPU simulation. Do not describe these paths
as paper-scale QAQ evidence.

## Current Implementation Status

Before starting feature work, read `doc/tasks/progress.md` and the matching
`doc/tasks/*.md` task file. As of the current scaffold:

- Implemented diagnostic/prototype paths include config/manifest handling,
  logging/progress, block discovery, bit-plane artifacts, static/fixed fake CPU
  runtime, router policy, router training health checks, and a CPU-simulated
  dynamic loader.
- Still incomplete or not accepted as final paper evidence: adaptive inference
  runtime, full evaluation metrics/reporter, real Hugging Face/LLaMA loading,
  real GPU memory measurements, and paper-scale benchmark reproduction.
- `fixed_mixed`, diagnostic router checkpoints, fake datasets, and generated
  health-check artifacts are validation tools only.

## Development Conventions

- Keep the package dependency-free unless the task explicitly approves adding a
  dependency.
- Prefer small, typed stdlib modules using `dataclass` models, `pathlib.Path`,
  explicit validation, and deterministic JSON serialization where artifacts are
  written.
- Preserve existing error style: custom exceptions should carry stable error
  codes and clear messages, and CLIs should fail before expensive work when
  configs or artifacts are invalid.
- Do not silently fall back from unsupported real-model, CUDA, dataset, or
  quantization behavior to fake behavior. Fail clearly or label the run as
  diagnostic.
- Keep output under `runs/` or a caller-provided output directory. `runs/`,
  caches, build output, and bytecode are ignored and should not be committed.
- Do not overwrite existing run directories unless the config has
  `overwrite: true` or a command intentionally uses a skip-output-dir check for
  a smoke/health run.

## Validation Commands

Use targeted tests while developing, then run the full suite before claiming a
code change is ready:

```bash
python -m pytest -q
```

Useful focused checks:

```bash
python -m pytest -q tests/unit
python -m pytest -q tests/integration
python -m pytest -q tests/e2e
python -m qaq.config configs/smoke.json --skip-output-dir-check --print-json
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.evaluate --config configs/smoke.json --skip-output-dir-check --print-json
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.router.train --health-check
```

`configs/router_train_smoke.yaml` is useful for config parsing coverage, but do
not assume it is a complete runnable training command unless the referenced
student bit-plane artifacts exist. Prefer the router training health-check
command above for the self-contained diagnostic router-training check.

## Testing Expectations

- Add or update tests beside the changed behavior: `tests/unit` for pure
  validation and data contracts, `tests/integration` for cross-module artifact
  or logging behavior, and `tests/e2e` for CLI/runtime smoke coverage.
- Fixture-backed fake data is acceptable for early health checks and regression
  tests, but must be labeled as fake or diagnostic in metadata, logs, and docs.
- For quantization/runtime/router changes, include artifact reload or checkpoint
  reload coverage where applicable.
- Keep static baselines comparable. QAQ comparison claims require matching
  model, tokenizer, dataset split, prompt format, precision candidates, and
  metric settings across modes.

## Anti-Smoke-Test Completion Rule

Smoke tests are allowed only as early health checks. They are never sufficient
to mark a feature complete.

A task is not complete if the implementation only validates:

- orchestration
- logging
- CLI argument parsing
- checkpoint save/load contracts
- failure behavior
- tiny synthetic data
- fake labels
- diagnostic toy objectives
- mocked training loops

For ML, router, and training features, done requires a real minimal
implementation:

- real data loading path, not synthetic-only
- real objective/loss matching the design document
- real targets/labels or distillation signal
- trainer wired to the actual model/router modules
- validation on held-out real examples or a real subset
- saved checkpoint reload used by inference/evaluation
- metrics recorded in a reproducible run artifact
- explicit command to reproduce the real run

If full paper-scale training is too expensive, implement a small real-data run,
not a fake diagnostic run. The small run must use the same real objective and
data format as the full run.

Never claim complete while residual risk says:

- fake/tiny diagnostic path
- no approved real objective
- no real training data
- no paper-scale or real-subset evidence

## ML Runtime Policy

This project uses a remote GPU server as the real ML runtime.

Local machine:
- GPU: RTX 4050
- Purpose: code editing, static checks, small CPU tests, syntax checks only
- Never treat local ML results as valid experiment results

Remote target:
- Use the lab server with physical GPU devices 0-7, each an RTX 3090, for all
  real ML workloads
- Run training, full inference, evaluation, benchmark, and model-loading experiments only on the remote server
- Launch every training, inference, evaluation, benchmark, or large-model-loading
  command through the GPU selector:

```bash
python scripts/gpu_run.py --count <N> --min-free-mb <MB> -- <command>
```

GPU selector rules:
- Never hardcode `CUDA_VISIBLE_DEVICES=0`.
- Never assume physical GPU 0 is free.
- Inspect GPU status before running.
- Record the selected physical GPU IDs.
- By default, accept only devices whose `nvidia-smi` name contains `RTX 3090`;
  do not override this unless intentionally targeting a different remote GPU.
- Remember that PyTorch sees selected GPUs as `cuda:0`, `cuda:1`, and so on
  inside the child process, even when the selected physical IDs are different.
- If no suitable GPU is free, stop and report instead of running locally.

Forbidden local commands:
- `python train.py`
- `python inference.py` when it loads a large language model
- `torchrun`
- `accelerate launch`
- any script that loads HuggingFace LLM weights larger than 1GB
- any full dataset evaluation
- any GPU memory benchmark
- any performance benchmark
- direct `python -m qaq.router.train ...` for nontrivial training
- direct `python -m qaq.evaluate ...` for full inference or evaluation

Allowed local commands:
- `rg`, `ls`, `cat`, `sed`, `git status`, `git diff`
- formatting and linting
- unit tests that do not load large models
- syntax checks
- tiny smoke tests only if explicitly labeled as smoke tests

Before running any ML command:
1. Check whether the current host is the remote server.
2. Run `hostname`.
3. Use `python scripts/gpu_run.py --count <N> --min-free-mb <MB> -- <command>`
   so `nvidia-smi` is queried and free physical GPU IDs are selected.
4. Confirm the selected physical GPU IDs and the child-process
   `CUDA_VISIBLE_DEVICES` mapping are recorded.
5. If not on the remote server, or if no suitable GPU is free, do not run the
   command locally. Report the exact command that should be run on the lab
   server instead.

Experiment reports must include:
- hostname
- GPU name
- CUDA_VISIBLE_DEVICES
- selected physical GPU IDs
- command
- git commit
- Python environment
- dataset path
- output path
- metric / score
