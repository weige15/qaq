# Repository Map

## Repository Summary

QAQ is a Python 3.12 research prototype scaffold for query-adaptive mixed-precision LLM inference. The current tree contains dependency-free local/fake execution paths, optional local Hugging Face LLaMA adapter hooks, router training, bit-plane artifacts, static/adaptive runtimes, reporting, configs, and pytest coverage. The project docs state that local fixture evidence is not paper-scale QAQ evidence.

## Directory Structure

- `qaq/`: package source.
- `qaq/router/`: router checkpoint, policy, loss, and training code.
- `qaq/runtime/`: static, adaptive, loader, and shared runtime contracts.
- `configs/`: smoke, router-training, router-eval, and first-milestone config stubs.
- `tests/`: unit, integration, e2e, regression, golden, and fixture files.
- `doc/`: requirements, design, task progress, ADRs, review notes, and router-training docs.
- `runs/`: ignored output directory used by local commands when generated.

## Main Source Files

- `qaq/config.py`: run config validation and config CLI.
- `qaq/model_adapter.py`: fake/local adapter plus optional local Hugging Face LLaMA adapter.
- `qaq/data.py` and `qaq/benchmark_adapter.py`: benchmark fixture loading and tokenization.
- `qaq/blocks.py` and `qaq/precision_plan.py`: MHA/FFN block descriptors and precision plans.
- `qaq/bitplanes.py`, `qaq/quantization.py`, and `qaq/artifacts.py`: quantization and bit-plane artifact contracts.
- `qaq/router/train.py`, `qaq/router/losses.py`, `qaq/router/policy.py`, and `qaq/router/checkpoint.py`: router training, objective, routing, and checkpoint serialization.
- `qaq/runtime/static.py`, `qaq/runtime/adaptive.py`, and `qaq/runtime/loader.py`: runtime paths and on-demand materialization simulation.
- `qaq/results.py`, `qaq/metrics.py`, `qaq/report.py`, and `qaq/evaluate.py`: result artifacts, metrics, reporting, and evaluation CLI.

## Existing Tests

- `tests/unit/`: config, bit-plane, block registry, loader, logging, router policy, and result schema tests.
- `tests/integration/`: artifact roundtrip, model adapter, static profiles, loader simulation, router checkpoint/training, and incomplete-run tests.
- `tests/e2e/test_smoke_modes.py`: fake/local static and QAQ smoke coverage.
- `tests/regression/test_qaq_acceptance_guards.py`: QAQ acceptance guard coverage.
- `tests/golden/`: stable JSON fixtures for bit-planes, reports, routing, and results.

## Build System

`pyproject.toml` uses `setuptools>=68`, declares package `qaq`, requires Python `>=3.12`, and has no required runtime dependencies.

## Runtime or CLI Entry Points

- `python -m qaq.config`
- `python -m qaq.evaluate`
- `python -m qaq.router.train`
- `python -m qaq.report`

## Data and Assets

- `QAQ.pdf`: local paper source.
- `tests/fixtures/benchmarks/*.jsonl`: fake smoke and router-training fixture samples.
- `tests/fixtures/bitplanes/router_training_real/*.json`: checked-in bit-plane artifacts for local router-training acceptance.
- `tests/fixtures/models/*.json` and `tests/fixtures/tokenizers/*.json`: local fake metadata.

## Existing Documentation

Primary docs include `doc/requirements.md`, `doc/high-level-design.md`, `doc/detailed-design.md`, `doc/test-plan.md`, `doc/router-training.md`, `doc/residual-risk.md`, `doc/tasks/progress.md`, `doc/review.md`, `doc/repair-plan.md`, and ADRs under `doc/adr/`.

## Detected Dependencies

The package has no required production dependencies. Optional real-model paths import `torch` and `transformers` only when the Hugging Face adapter is used.

## Important Scripts

No standalone scripts directory was found. Operational commands are module entry points and pytest commands documented in `AGENTS.md` and `doc/tasks/progress.md`.

## Current Git State

At discovery time, `.gitignore` was modified and `AGENTS.md`, `configs/`, `doc/`, `pyproject.toml`, `qaq/`, and `tests/` were untracked. This repo appears to be an in-progress scaffold rather than a clean committed baseline.

## Missing or Ambiguous Areas

- Real CUDA on-demand materialization is not implemented.
- Paper-scale LLaMA/Qwen bit-plane artifact generation and training data are not present.
- The official QAQ router loss, training corpus, and hyperparameters are unavailable.
- Lint, formatting, and static type-check commands are not configured.

## Notes for Future Skills

Use local pytest and module CLIs first. Treat fake/local fixture commands as health or local acceptance checks only, and require GPU-backed runs before making memory or on-demand loading claims.
