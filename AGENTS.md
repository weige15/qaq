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
python -m qaq.evaluate --config configs/smoke.json --skip-output-dir-check --print-json
python -m qaq.router.train --health-check
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
