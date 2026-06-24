# Model and Benchmark Adapter

## Goal

Implement a model and benchmark adapter that loads the causal LLM/tokenizer, creates comparable benchmark inputs, exposes FP16/reference execution, and returns hidden features for router decisions.

## Inputs

- `doc/proposal.md`: Baselines and router training require full-precision teacher/reference behavior and benchmark-compatible evaluation.
- `doc/high-level-design.md`: The adapter owns model/tokenizer references, benchmark prompt/tokenization behavior, and frozen base model execution for router training.
- `doc/detailed-design.md`: Defines tokenized batches, hidden-state bundles keyed by block ID, model architecture metadata, and benchmark comparability rules.
- `doc/test-plan.md`: Requires smoke E2E, public/sample validation, edge-case validation, and performance tests using shared checkpoint, tokenizer, prompt format, and metric code.

## Write Scope

Create or edit proposed paths: `qaq/model_adapter.py`, `qaq/benchmark_adapter.py`, `qaq/data.py`, `tests/integration/test_model_adapter_smoke.py`, and benchmark fixtures under `tests/fixtures/`.

## Read Scope

Inspect config and manifest APIs, block registry interfaces, router feature requirements, and any selected model/evaluation library documentation.

## Dependencies

Experiment Configuration and Run Manifest. Coordinate model metadata with Block Registry and Precision Plan. External dependency choice for model loading and benchmark datasets must be approved before full implementation.

## Tasks

- [x] Implement tokenizer and model loading from validated config, with explicit errors for missing or inaccessible checkpoints.
- [x] Implement benchmark example loading/tokenization with recorded dataset, split, prompt format, batch size, and context-length policy.
- [x] Expose FP16/reference execution outputs needed by static baselines, router training, and evaluation metrics.
- [x] Expose hidden representations at a documented feature point for each controlled block.
- [x] Provide model architecture metadata to the Block Registry without leaking framework-specific details into downstream modules.
- [x] Preserve fake/tiny adapter tests as diagnostic regression coverage only.
- [x] Add real local Hugging Face LLaMA-family adapter verification path and CLI.
- [x] Add real-benchmark local-root tokenization verification using non-fake adapter provenance.
- [x] Add lab-server GPU verification command for opt-in large checkpoint loading.

## Tests and Quality Gates

- [x] Run `pytest -q tests/integration/test_model_adapter_smoke.py` when implemented.
- [x] Verify tokenizer, prompt format, dataset split, and context policy are recorded in run metadata.
- [x] Verify unsupported model architecture or unavailable dataset fails clearly.

## Done When

- [x] Diagnostic fake adapter tests still pass, and adapter output now marks them diagnostic-only.
- [x] A real local Hugging Face LLaMA-family adapter verification path exists and fails clearly when dependencies, local files, tokenizer, model config, or CUDA are unavailable.
- [ ] The adapter can resolve a local cached `meta-llama/Llama-3.1-8B` snapshot without network access.
- [x] The adapter records model id, tokenizer id, dataset, split, prompt format, context policy, selected GPU IDs when applicable, benchmark/data provenance, and whether the run is fake/diagnostic.
- [x] At least one non-fake local or lab-server verification command is documented.
- [x] The task is not marked complete from `fake_smoke`, `TinyHFModel`, mocked model objects, synthetic tensors, or fixture-only tests.

Verification commands:

```bash
python -m qaq.model_adapter --config configs/benchmarks/llama_first_milestone/hellaswag/fp16.json --limit 8 --print-json
python scripts/gpu_run.py --count 1 --min-free-mb 18000 --status-file runs/gpu-selector/model-adapter-load-weights.json -- python -m qaq.model_adapter --config configs/benchmarks/llama_first_milestone/hellaswag/fp16.json --limit 1 --load-weights --print-json
```

The first command verifies local metadata, tokenizer, and benchmark batching without loading weights. The second command is the lab-server-only large-checkpoint weight-load verification path.
