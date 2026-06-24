# Router Training

## Objective Assumption

The QAQ design requires frozen base models, full-precision teacher signals, quantized student signals, hidden-state router features, and knowledge distillation, but it does not specify the exact router loss.

This implementation uses `router_cost_cross_entropy` as the smallest concrete objective consistent with that design:

1. Load file-backed training samples.
2. Run teacher and student reference forwards with base parameters frozen.
3. For every sample and controlled block, use the teacher hidden feature as the router input.
4. For every candidate bit-width, estimate candidate cost as:
   - teacher/student logit MSE for the sample
   - plus bit-plane reconstruction distortion for that block and bit-width, scaled by hidden-feature sensitivity
   - plus a documented bit-cost penalty `bit_cost_weight * bit_width / max_bit_width`
5. Convert lower candidate costs into target probabilities with `softmax(-cost / target_temperature)`.
6. Train the linear router with cross-entropy from those target probabilities to router softmax probabilities.

The bit-cost penalty is an implementation assumption. Without a cost term, the defensible minimal distillation target would usually prefer maximum precision for every block.

Feature extraction is also an implementation assumption. The dependency-free
local adapter exposes deterministic block-level pooled features for each MHA and
FFN block. The optional Hugging Face LLaMA adapter exposes pooled per-layer
hidden states, and the current block registry shares that layer feature between
the layer's MHA and FFN controlled blocks. This is acceptable for the minimal
real objective because routing remains per block and per sample, but it is not a
claim that the official QAQ implementation used the same feature tap.

For Hugging Face Llama-family checkpoints, the adapter exposes MHA and FFN block metadata from the model config. The current minimal router feature uses the pooled transformer layer output for both the layer's MHA and FFN router decisions because the Hugging Face hidden-state API exposes layer boundaries directly, while separate MHA/FFN activation capture requires hooks that are outside this router-training workstream.

## Acceptance Command

All training commands must be launched on the lab GPU server through the GPU
selector so the run inspects free physical devices, records the selected IDs,
and avoids assuming physical GPU 0 is available.

The non-diagnostic local-fixture acceptance command is:

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.router.train --config configs/router_train_real.yaml
```

The checkpoint-loaded evaluation command is:

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.evaluate --config configs/router_eval_real.json --artifact-index configs/router_eval_real_artifacts.json --skip-output-dir-check --print-json
```

For non-diagnostic training, both the run manifest and router checkpoint metadata must record the run as non-diagnostic. The manifest uses an `fp16` reference run config because router training is not itself a QAQ inference pass and has no router checkpoint at manifest creation time.

The training command writes `router_targets.json` in the run output directory. That audit artifact records the file-backed sample counts, sample/block target records, candidate costs, target distributions, objective name, and diagnostic flag so acceptance tests can prove target generation without relying only on checkpoint metadata.

When `overwrite: true` is set, the first log writer for a run truncates its previous JSONL log. This keeps reruns reproducible and prevents stale progress or completion events from being counted as current evidence.

Quick diagnostic health checks are separate:

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.router.train --health-check
```

The health-check path may use built-in fake data and generated temporary artifacts. It is not an acceptance gate.

## Llama 3.1 8B Base Model Status

The base checkpoint target is `meta-llama/Llama-3.1-8B`, not the Instruct variant. The local adapter can now read Llama-family Hugging Face metadata and expose paper-aligned MHA/FFN block IDs and tensor names for that base model.

A full Llama 3.1 8B router-training command additionally requires:

- a file-backed calibration/training split,
- Llama-compatible bit-plane artifacts for every controlled block in `student_quantized_path`,
- enough GPU memory to run teacher/student reference forwards when `reference_forward` is invoked.

If those artifacts or local Hugging Face files are absent, the run must fail clearly instead of falling back to fake/local metadata.

The current router trainer intentionally performs a CUDA capacity preflight
before loading Hugging Face weights. When `teacher_model` and `student_model`
are the same exact model reference, the trainer uses a shared frozen reference
adapter and reuses the teacher forward output as the student reference output.
This is valid for the current minimal objective because quantized-student
effects enter through bit-plane artifact reconstruction distortion rather than
through a separately executed quantized transformer. Distinct teacher and
student model references still use separate adapters and separate reference
forwards.

On the currently visible single RTX 4050 Laptop GPU, the Llama sampled-artifact
training command must not be run directly. On the lab server, use the GPU
selector and require enough free memory before model loading:

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 18000 -- python -m qaq.router.train --config configs/router_train_llama31_8b_sampled.yaml
```

Observed escalated preflight result after the shared-reference change on
2026-06-24:

```text
insufficient_cuda_memory: requires at least 15.46 GiB free before activations; cuda:0 reports 4.96 GiB free of 6.00 GiB total
```

This is a hardware/capacity blocker, not a router objective or artifact
compatibility failure. Do not report this Llama command as a successful training
run until it completes on a sufficiently large GPU setup.

## Llama Bit-Plane Artifact Preparation

The minimal local artifact-preparation command for the base checkpoint is:

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.prepare_bitplanes --model meta-llama/Llama-3.1-8B --output-dir runs/llama31_8b_bitplanes_sampled --sample-values 16 --overwrite --print-json
```

This command reads the local Hugging Face safetensors snapshot, discovers the
Llama MHA/FFN block registry through the real model metadata, samples real
weight values from one owned tensor per controlled block, and writes valid QAQ
bit-plane artifacts plus `artifact_index.json`.

The generated `artifact_index.json` is the path to use as
`student_quantized_path` for router-training preflight or training configs:

```text
runs/llama31_8b_bitplanes_sampled/artifact_index.json
```

The artifact scope is `sampled_weight_values`. These artifacts are real-data
router-training target inputs for the current distortion term, but they are not
full tensor bit-plane artifacts and are not accepted as full quantized Llama
inference evidence. Each artifact records the source shard, source tensor name,
source tensor shape, source dtype, sample policy, sample count, and
`accepted_as_full_quantized_inference_artifact: false`.

For tensor-native artifacts, use the safetensors generator:

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.llama_bitplanes \
  --model meta-llama/Llama-3.1-8B \
  --artifact-format safetensors \
  --output-dir runs/llama31_8b_native_bitplanes_probe \
  --block-limit 1 \
  --tensor-limit-per-block 2 \
  --max-elements-per-tensor 16 \
  --overwrite \
  --print-json
```

The trainer accepts `.qaq.safetensors` artifacts through the same
`student_quantized_path` directory or index mechanism used by JSON artifacts.
The real objective still uses `router_cost_cross_entropy`; only the artifact
storage and distortion-read path change.

The router trainer preflight check for the generated artifact index is:

```bash
python - <<'PY'
import json
from qaq.router.train import RouterTrainingConfig, validate_router_training_preflight
config = RouterTrainingConfig.from_mapping({
    "model": "meta-llama/Llama-3.1-8B",
    "tokenizer": "meta-llama/Llama-3.1-8B",
    "data_source": "tests/fixtures/benchmarks/router_training_real.jsonl",
    "split": "train",
    "teacher_model": "meta-llama/Llama-3.1-8B",
    "student_model": "meta-llama/Llama-3.1-8B",
    "student_quantized_path": "runs/llama31_8b_bitplanes_sampled/artifact_index.json",
    "distillation_loss": "router_cost_cross_entropy",
    "precision_candidates": [4, 8],
    "max_bit_width": 8,
    "block_granularity": "mha_ffn",
    "device": "cpu",
    "gpu_ids": [],
    "seed": 0,
    "output_dir": "runs/llama31_router_train_preflight",
    "overwrite": True,
    "prompt_format": "question_answer_v1",
    "training_data_limit": 1,
    "validation_data_limit": 1,
    "diagnostic": False,
    "logging": {"console": False},
})
validate_router_training_preflight(config)
print(json.dumps({"preflight": "ok", "objective": config.distillation_loss}, sort_keys=True))
PY
```
