# 2026-06-24 QAQ Experiment Report

## Summary

The workflow ran on `basic-2`, which is the lab RTX 3090 server. The GPU
selector detected eight `NVIDIA GeForce RTX 3090` devices and selected physical
GPU `6` for all GPU-wrapped commands in this run.

The correctness gates passed, the diagnostic smoke path passed, the
non-diagnostic local-fixture router path passed, sampled real-weight LLaMA
bit-plane artifacts were regenerated, and a one-step sampled LLaMA router
training run completed on an RTX 3090.

No accepted paper-scale QAQ result was produced. The initial checkpoint-loaded
LLaMA evaluation attempt was blocked at result aggregation, but a later
Model and Benchmark Adapter pass added Hugging Face target-token losses and the
same command now writes a result artifact. That result remains small fixture
evidence over sampled artifacts, not accepted QAQ benchmark evidence.

## Hostname

- Hostname: `basic-2`
- Host classification: lab GPU server, not the local RTX 4050 machine.
- Evidence: `python scripts/gpu_run.py --dry-run --count 1 --min-free-mb 1000 -- python -c "print('gpu selector ok')"` detected eight eligible RTX 3090 GPUs.

## Git Commit

- Commit: `d5bd3ce0eb205c58a4cc93001ff63420ab6672f7`
- `git status --short`: clean before report edits.

## Python Environment

- Python: `Python 3.12.3`
- `python -m pip list | sed -n '1,120p'` was run for the environment record.
- Relevant packages:
  - `torch 2.4.0+cu124`
  - `transformers 5.12.1`
  - `safetensors 0.8.0`
  - `huggingface_hub 1.19.0`
  - `pytest 9.0.3`

## GPU Selector Dry Run

Command:

```bash
python scripts/gpu_run.py --dry-run --count 1 --min-free-mb 1000 -- python -c "print('gpu selector ok')"
```

Result:

- Status: `selected`
- Selected physical GPU IDs: `[6]`
- `CUDA_VISIBLE_DEVICES`: `6`
- PyTorch logical mapping: `cuda:0 -> physical 6`
- Eligible physical IDs: `[0, 1, 2, 3, 4, 5, 6, 7]`
- GPU name filter: `RTX 3090`

## Commands Run

### Phase 0: Audit

```bash
hostname
git status --short
git rev-parse HEAD
python --version
python -m pip list | sed -n '1,120p'
python scripts/gpu_run.py --dry-run --count 1 --min-free-mb 1000 -- python -c "print('gpu selector ok')"
rg --files -uu
python -m pip show torch transformers safetensors huggingface_hub pytest
```

Result: host and environment audit completed. `rg --files -uu` showed the
repository source, docs, configs, tests, `.git`, caches, and ignored `runs/`
artifacts.

### Phase 1: Diagnostic Gates

```bash
python -m pytest -q
```

Result: `125 passed in 9.56s`.

```bash
python -m qaq.config configs/smoke.json --skip-output-dir-check --print-json
```

Result: passed. Config is fake CPU `fp16` smoke validation.

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.evaluate --config configs/smoke.json --skip-output-dir-check --print-json
```

Result: passed. Selected physical GPU IDs `[6]`, `CUDA_VISIBLE_DEVICES=6`,
logical mapping `cuda:0 -> physical 6`. Runtime metadata was fake CPU
`qaq.runtime.static.fake_cpu` over `fake_smoke`, with 2 examples and
`peak_gpu_memory_gb=0.0`.

Evidence level: diagnostic only.

### Phase 2: Non-Diagnostic Local-Fixture Router Training

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.router.train --config configs/router_train_real.yaml
```

Result: passed. Selected physical GPU IDs `[6]`, `CUDA_VISIBLE_DEVICES=6`,
logical mapping `cuda:0 -> physical 6`.

Outputs:

- `runs/router_train_real/checkpoints/router_step_0003.json`
- `runs/router_train_real/router_targets.json`
- `runs/router_train_real/manifest.json`
- `runs/router_train_real/logs/router_train.jsonl`

Metrics and evidence:

- Objective: `router_cost_cross_entropy`
- Training samples: `3`
- Target records: `12`
- Validation loss: `0.693113872343573`
- Checkpoint metadata `diagnostic: false`
- Target audit `diagnostic_training: false`
- Parameter update metadata is present.
- Manifest status: `completed`
- Log last event: `completion`, status `completed`

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.evaluate --config configs/router_eval_real.json --artifact-index configs/router_eval_real_artifacts.json --skip-output-dir-check --print-json
```

Result: passed. Selected physical GPU IDs `[6]`, `CUDA_VISIBLE_DEVICES=6`,
logical mapping `cuda:0 -> physical 6`.

Evaluation evidence:

- `router_checkpoint_loaded: true`
- Runtime mode: `qaq_on_demand_off`
- Adaptive traces: `2`
- Routing decisions: `8`
- Routing summary `diagnostic: false`
- `constant_global_precision: false`
- Precision counts: `4-bit=7`, `8-bit=1`

Evidence level: non-diagnostic small real run using checked-in local fixtures.

### Phase 3: LLaMA Sampled Artifact Preparation

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.prepare_bitplanes --model meta-llama/Llama-3.1-8B --output-dir runs/llama31_8b_bitplanes_sampled --sample-values 16 --overwrite --print-json
```

Result: passed. Selected physical GPU IDs `[6]`, `CUDA_VISIBLE_DEVICES=6`,
logical mapping `cuda:0 -> physical 6`.

Outputs:

- `runs/llama31_8b_bitplanes_sampled/artifact_index.json`
- `runs/llama31_8b_bitplanes_sampled/manifest.json`
- `runs/llama31_8b_bitplanes_sampled/artifacts/*.json`

Artifact evidence:

- Artifact count: `64`
- Model ID: `meta-llama/Llama-3.1-8B`
- Resolved snapshot: `/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- Sample artifact metadata records real safetensor source shards such as `model-00001-of-00004.safetensors`.
- Sample artifact metadata records real tensor names such as `model.layers.0.self_attn.q_proj.weight`.
- Source dtype: `BF16`
- Artifact scope: `sampled_weight_values`
- `accepted_as_full_quantized_inference_artifact: false`
- `full_tensor_values_stored: false`
- Sample count per artifact: `16`

Evidence level: real-weight sampled artifact preparation, not full quantized
inference artifact evidence.

### Phase 4: LLaMA Sampled Router Training

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 18000 -- python -m qaq.router.train --config configs/router_train_llama31_8b_sampled.yaml
```

Result: passed. Selected physical GPU IDs `[6]`, `CUDA_VISIBLE_DEVICES=6`,
logical mapping `cuda:0 -> physical 6`.

Outputs:

- `runs/router_train_llama31_8b_sampled/checkpoints/router_step_0001.json`
- `runs/router_train_llama31_8b_sampled/router_targets.json`
- `runs/router_train_llama31_8b_sampled/manifest.json`
- `runs/router_train_llama31_8b_sampled/logs/router_train.jsonl`

Metrics and evidence:

- Model ID: `meta-llama/Llama-3.1-8B`
- Objective: `router_cost_cross_entropy`
- Training samples: `1`
- Target records: `64`
- Validation loss: `0.6929037649795032`
- Validation distillation cost: `0.0017881646347167363`
- Validation efficiency penalty: `0.029587437074223757`
- Parameter update L2: `0.0020548994143244448`
- Checkpoint metadata `diagnostic: false`
- Target audit `diagnostic_training: false`
- Shared reference path: `shared_teacher_student_reference: true`
- Manifest status: `completed`
- Log last event: `completion`, status `completed`

The manifest records child-process logical GPU ID `[0]`; the GPU selector
record maps that logical `cuda:0` to physical GPU `6`.

Run-local evaluation config created under ignored output:

- `runs/router_train_llama31_8b_sampled/eval_qaq_on_demand_off.json`

Config validation command:

```bash
python -m qaq.config runs/router_train_llama31_8b_sampled/eval_qaq_on_demand_off.json --print-json
```

Result: passed.

Checkpoint-loaded evaluation attempt:

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 18000 -- python -m qaq.evaluate --config runs/router_train_llama31_8b_sampled/eval_qaq_on_demand_off.json --artifact-index runs/llama31_8b_bitplanes_sampled/artifact_index.json --result-output runs/router_train_llama31_8b_sampled/eval_qaq_on_demand_off/result_artifact.json
```

Initial result: failed with exit code `5` after model loading because the Hugging
Face reference output had no target-derived losses for `router_acceptance`.

Follow-up result after the Model and Benchmark Adapter loss fix: passed and
wrote `runs/router_train_llama31_8b_sampled/eval_qaq_on_demand_off/result_artifact.json`.
The result artifact records completion status `completed`, metric
`router_acceptance`, fixture score `0.0`, peak GPU memory about `14.965 GB`, and
128 routing decisions over 2 fixture validation examples.

Evidence level: training and checkpoint-loaded evaluation are non-diagnostic
small real LLaMA sampled-artifact runs; they are not accepted paper-scale QAQ
evidence.

### Phase 5: Optional Tensor-Native Probe

Not run. Phase 3 and Phase 4 did not fail due artifact-format compatibility.
The remaining blocker is metric/result aggregation for LLaMA evaluation, not
tensor-native artifact format.

## Output Directories

- `runs/router_train_real`
- `runs/llama31_8b_bitplanes_sampled`
- `runs/router_train_llama31_8b_sampled`
- `runs/router_train_llama31_8b_sampled/eval_qaq_on_demand_off.json`

No required result artifact was produced for LLaMA evaluation because the
evaluation command failed before writing
`runs/router_train_llama31_8b_sampled/eval_qaq_on_demand_off/result_artifact.json`.

## Artifacts Produced

- Local-fixture router checkpoint:
  `runs/router_train_real/checkpoints/router_step_0003.json`
- Local-fixture router targets:
  `runs/router_train_real/router_targets.json`
- Local-fixture manifest/log:
  `runs/router_train_real/manifest.json`,
  `runs/router_train_real/logs/router_train.jsonl`
- Sampled LLaMA artifact index:
  `runs/llama31_8b_bitplanes_sampled/artifact_index.json`
- Sampled LLaMA bit-plane artifacts:
  `runs/llama31_8b_bitplanes_sampled/artifacts/*.json`
- LLaMA sampled router checkpoint:
  `runs/router_train_llama31_8b_sampled/checkpoints/router_step_0001.json`
- LLaMA sampled router targets:
  `runs/router_train_llama31_8b_sampled/router_targets.json`
- LLaMA sampled router manifest/log:
  `runs/router_train_llama31_8b_sampled/manifest.json`,
  `runs/router_train_llama31_8b_sampled/logs/router_train.jsonl`

## Metrics Observed

| Run | Metric | Value |
| --- | --- | --- |
| Full test suite | pytest | `125 passed in 9.56s` |
| Diagnostic smoke eval | processed examples | `2` |
| Diagnostic smoke eval | peak GPU memory | `0.0 GB`, fake CPU path |
| Local-fixture router train | validation loss | `0.693113872343573` |
| Local-fixture router train | target records | `12` |
| Local-fixture checkpoint eval | routing decisions | `8` |
| Local-fixture checkpoint eval | routing variation | `constant_global_precision=false` |
| LLaMA sampled artifact prep | artifact count | `64` |
| LLaMA sampled router train | validation loss | `0.6929037649795032` |
| LLaMA sampled router train | target records | `64` |
| LLaMA sampled router train | parameter update L2 | `0.0020548994143244448` |
| LLaMA checkpoint eval | result | completed: `result_artifact.json`, fixture score `0.0` |

## Diagnostic Only

- `configs/smoke.json` validation.
- `python -m qaq.evaluate --config configs/smoke.json ...`.
- Fake CPU runtime outputs and `peak_gpu_memory_gb=0.0`.

These prove CLI/config/runtime health only. They are not accepted QAQ evidence.

## Non-Diagnostic But Small-Scale

- `configs/router_train_real.yaml` router training over checked-in
  file-backed fixtures.
- `configs/router_eval_real.json` checkpoint-loaded evaluation over checked-in
  file-backed fixtures.
- `configs/router_train_llama31_8b_sampled.yaml` one-step LLaMA sampled router
  training using real model weights and sampled real-weight artifacts.

These runs are useful QAQ implementation evidence, but they are not paper-scale
benchmark evidence.

## Blocked

- Full QAQ inference claims remain blocked because artifacts are sampled
  16-value weight slices, not full quantized LLaMA inference artifacts.
- Paper-style acceptance remains blocked until comparable FP16, static 8-bit,
  static 4-bit, `qaq_on_demand_off`, and `qaq_on_demand_on` runs exist on the
  same real model, tokenizer, dataset split, prompt format, precision
  candidates, and metric implementation.

## What Can Be Claimed

- The workflow ran on the intended lab RTX 3090 server through
  `scripts/gpu_run.py`.
- Correctness tests passed.
- The diagnostic smoke path is healthy.
- The non-diagnostic local-fixture router path trains, saves, reloads, and
  emits non-diagnostic routing decisions.
- Sampled real-weight LLaMA bit-plane artifacts can be generated from cached
  base `meta-llama/Llama-3.1-8B` safetensors.
- A one-step non-diagnostic sampled LLaMA router training run completes on an
  RTX 3090 and writes checkpoint, target audit, manifest, logs, and validation
  loss.
- The sampled LLaMA checkpoint-loaded `qaq_on_demand_off` evaluation now writes
  a result artifact with target-derived fixture losses.

## What Cannot Be Claimed

- No accepted QAQ benchmark result was produced.
- No paper-scale QAQ accuracy, perplexity, latency, or memory comparison was
  produced.
- No full quantized LLaMA inference artifact was produced.
- The successful LLaMA checkpoint-loaded evaluation result artifact is not an accepted benchmark result.
- No on-demand LLaMA memory reduction or latency-overhead claim is supported.
- No full paper reproduction claim is supported.

## Next Experiment

The next experiment is to move beyond the sampled-artifact fixture path and run
comparable real benchmark modes. The immediate work is to create or approve a
first-milestone LLaMA evaluation config matrix for `fp16`, `static_8bit`,
`static_4bit`, `qaq_on_demand_off`, and `qaq_on_demand_on` using the same model,
tokenizer, dataset split, prompt format, precision candidates, and metric.

The sampled checkpoint evaluation command now runs, but it remains small-scale
fixture evidence:

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 18000 -- python -m qaq.evaluate --config runs/router_train_llama31_8b_sampled/eval_qaq_on_demand_off.json --artifact-index runs/llama31_8b_bitplanes_sampled/artifact_index.json --result-output runs/router_train_llama31_8b_sampled/eval_qaq_on_demand_off/result_artifact.json
```

Do not treat this fixture score or the sampled 16-value artifacts as accepted
QAQ benchmark evidence.
