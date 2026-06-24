# Benchmark Integration Plan

QAQ does not yet have accepted real benchmark support. The checked-in first-milestone configs under `configs/benchmarks/llama_first_milestone/` validate with `qaq.config` and name real benchmarks, but they do not by themselves prove benchmark support. No QAQ benchmark result is accepted until at least one real benchmark produces a result artifact through `python -m qaq.evaluate` and passes `doc/acceptance-contract.md`.

The repo-native benchmark loader now resolves supported real benchmark names from local JSON/JSONL files or a cached Hugging Face `datasets` copy. This fixes the earlier `dataset_unavailable` failure for names such as `hellaswag`, but it is still only data-loading support. It is not accepted benchmark support until the QAQ runtime, metric implementation, all five modes, GPU selector record, and report contract pass together.

## Current Repo-Native Loader

For supported benchmark names, place local data under one of these layouts and set `QAQ_BENCHMARK_DATA_ROOT`:

```text
<root>/<benchmark>/<split>.jsonl
<root>/<benchmark>/<split>.json
<root>/<benchmark>_<split>.jsonl
<root>/<benchmark>.jsonl
```

The loader also tries `datasets.load_dataset(..., download_config=DownloadConfig(local_files_only=True))`, so an already-prepared Hugging Face datasets cache can be used without network download. Set `QAQ_DISABLE_HF_DATASETS=1` to require local files only.

## Chosen Direction

Use `lm-evaluation-harness` as the first optional benchmark backend, then wire QAQ runtime execution into the harness task loop. The repo-native loader is an interim real-data input path; exact lm-eval scoring and task semantics are still TODO.

## Initial Benchmark Names

- HellaSwag: `hellaswag`, split `validation`, metric `acc_norm` or exact-match equivalent chosen by lm-eval.
- PIQA: `piqa`, split `validation`, metric `acc`.
- ARC-Easy: `arc_easy`, split `validation`, metric `acc_norm` or accepted lm-eval default.
- ARC-Challenge: `arc_challenge`, split `validation`, metric `acc_norm` or accepted lm-eval default.
- WinoGrande: `winogrande`, split `validation`, metric `acc`.
- WikiText-2: `wikitext2`, split `test`, metric `perplexity`.
- PTB: TODO. The current repo-native benchmark path only supports built-in fake/local datasets or JSON/JSONL files. Add a PTB adapter or lm-eval task mapping before creating accepted PTB configs.

## Command Templates

Run real experiments only on the lab RTX 3090 server through the GPU selector. Example templates:

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 18000 --status-file runs/gpu-selector/hellaswag-fp16.json -- python -m qaq.evaluate --config configs/benchmarks/llama_first_milestone/hellaswag/fp16.json --result-output runs/llama_first_milestone/hellaswag/fp16/result_artifact.json

python scripts/gpu_run.py --count 1 --min-free-mb 18000 --status-file runs/gpu-selector/hellaswag-static-8bit.json -- python -m qaq.evaluate --config configs/benchmarks/llama_first_milestone/hellaswag/static_8bit.json --artifact-index runs/llama31_8b_full_tensor_bitplanes/runtime_artifact_index.json --result-output runs/llama_first_milestone/hellaswag/static_8bit/result_artifact.json

python scripts/gpu_run.py --count 1 --min-free-mb 18000 --status-file runs/gpu-selector/hellaswag-qaq-on.json -- python -m qaq.evaluate --config configs/benchmarks/llama_first_milestone/hellaswag/qaq_on_demand_on.json --artifact-index runs/llama31_8b_full_tensor_bitplanes/runtime_artifact_index.json --result-output runs/llama_first_milestone/hellaswag/qaq_on_demand_on/result_artifact.json

python -m qaq.report --results runs/llama_first_milestone/hellaswag/fp16/result_artifact.json runs/llama_first_milestone/hellaswag/static_8bit/result_artifact.json runs/llama_first_milestone/hellaswag/static_4bit/result_artifact.json runs/llama_first_milestone/hellaswag/qaq_on_demand_off/result_artifact.json runs/llama_first_milestone/hellaswag/qaq_on_demand_on/result_artifact.json --output runs/llama_first_milestone/hellaswag/report.json --print-json
```

## TODOs

- Add an optional dependency path for `lm-eval` without making local schema tests require it.
- Implement a benchmark adapter that maps lm-eval examples into QAQ prompt/token/target batches while preserving model, tokenizer, split, prompt format, and metric metadata.
- Ensure `qaq.evaluate` records `benchmark_name`, `benchmark_split`, `gpu_selector_record`, and non-fake dataset provenance in every result artifact.
- Run at least one real benchmark through all five required modes before claiming benchmark support.
- Add PTB support through lm-eval or a repo-native adapter before PTB first-milestone configs can be accepted.
