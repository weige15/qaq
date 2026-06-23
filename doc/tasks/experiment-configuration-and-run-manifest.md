# Experiment Configuration and Run Manifest

## Goal

Implement validated run configuration and immutable run manifests so invalid QAQ runs fail before model loading and every run is reproducible from saved metadata.

## Inputs

- `doc/proposal.md`: Stage 1 requires reproducible baselines with model, dataset, mode, precision, seed, hardware, latency, and memory captured.
- `doc/high-level-design.md`: Configuration is the foundation for all runtime, training, evaluation, and logging modules.
- `doc/detailed-design.md`: Defines Run Configuration, Run Manifest, validation order, output overwrite policy, and incomplete-run status.
- `doc/test-plan.md`: Requires config validation tests for invalid modes, missing fields, precision candidates, GPU IDs, unsafe output reuse, and missing router checkpoints.

## Write Scope

Create or edit proposed paths: `qaq/config.py`, `qaq/manifest.py`, `qaq/errors.py`, `configs/`, `tests/unit/test_config_validation.py`, and fixtures under `tests/fixtures/configs/`.

## Read Scope

Inspect `doc/detailed-design.md` Shared Data Contracts and the Experiment Configuration module, plus any package layout or CLI files that exist when implementation begins.

## Dependencies

None for core validation. Coordinate field names with Logging and Progress Tracking, Evaluation Metrics and Results Reporter, and Router Policy Module.

## Tasks

- [x] Define a run configuration schema with model, tokenizer, dataset, split, mode, precision candidates, max bit-width, block granularity, GPU IDs, seed, output directory, overwrite policy, logging settings, and router checkpoint fields.
- [x] Implement semantic validation for mode-specific requirements, precision candidates, output reuse, selected GPUs, and required QAQ router checkpoint metadata.
- [x] Implement manifest creation with resolved config, hardware metadata, artifact paths, start status, completion status, failure status, and incomplete-run marker fields.
- [x] Add clear validation error categories and non-zero failure behavior for invalid configs.
- [x] Add sample smoke and first-milestone config stubs under `configs/` once CLI/package conventions are chosen.
- [x] Add unit tests and fixtures covering valid configs and all required invalid config cases.

## Tests and Quality Gates

- [x] Run `pytest -q tests/unit/test_config_validation.py` when pytest is configured.
- [x] Verify invalid configs fail before any model loading or artifact creation.
- [x] Verify existing output directories require explicit overwrite permission.

## Done When

- [x] A valid config creates a manifest with all required reproducibility fields.
- [x] Invalid mode, precision, GPU, output, and router-checkpoint cases fail with clear errors.
- [x] Config unit tests pass.
