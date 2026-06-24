# Configs

Checked-in smoke, first-milestone, and report configuration files belong here.

## First-Milestone Benchmark Stubs

`configs/benchmarks/llama_first_milestone/` contains structural LLaMA-3.1-8B configs for real benchmark names and all required modes. They validate with `qaq.config`, but they are not accepted benchmark support until the backend described in `doc/benchmark-integration-plan.md` produces real result artifacts through QAQ evaluation and `qaq.report` accepts the five-mode matrix.


The named benchmark loader looks for local real-data files under `QAQ_BENCHMARK_DATA_ROOT`, for example `QAQ_BENCHMARK_DATA_ROOT=/data/qaq-benchmarks` with `hellaswag/validation.jsonl`. If local files are absent, it tries a cached Hugging Face `datasets` copy with local files only. These inputs still do not make a result accepted unless the full acceptance contract passes.
