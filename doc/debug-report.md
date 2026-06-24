# Debug Report

## Symptom
`qaq.prepare_bitplanes` now finds a local Hugging Face snapshot for `meta-llama/Llama-3.1-8B`, but fails while loading model metadata because the snapshot does not contain `config.json`.

## Reproduction Command
Working directory: `/nfs/home/s314511048/qaq`
Shell: `bash`
Runtime: `Python 3.12.3`
Environment: `/nfs/home/s314511048/.venv/bin/python`
Relevant environment variables:
```text
HF_HOME=/nfs/home/s314511048/.cache/huggingface
HF_HUB_CACHE=/nfs/home/s314511048/.cache/huggingface/hub
TRANSFORMERS_CACHE=<unset>
HF_HUB_OFFLINE=<unset>
TRANSFORMERS_OFFLINE=<unset>
```

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.prepare_bitplanes \
--model meta-llama/Llama-3.1-8B --output-dir runs/llama31_8b_bitplanes_sampled --sample-values 16 --overwrite --print-json
```

The full artifact-preparation command was not rerun during diagnosis. The metadata-only failure was reproduced with:

```bash
python -c "from transformers import AutoConfig; p='/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b'; print(AutoConfig.from_pretrained(p, local_files_only=True))"
```

## Expected Behavior
Transformers should read `config.json`, detect `model_type: llama`, and allow QAQ to discover the Llama block metadata before reading sampled safetensor weights.

## Actual Behavior
The snapshot contains model shards, tokenizer files, and `model.safetensors.index.json`, but no `config.json`, so Transformers cannot identify the model architecture.

## Error Log
```text
model_metadata_unavailable: model_unavailable: model '/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b' is not a supported fake model or local Hugging Face config: Unrecognized model in /nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b. Should have a `model_type` key in its config.json.
```

Metadata-only reproduction:

```text
ValueError: Unrecognized model in /nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b. Should have a `model_type` key in its config.json.
```

## Failure Layer Classification
Most likely layer:

* Command problem: no
* Permission problem: no
* Shell/script invocation problem: no
* Environment problem: yes
* Dependency problem: no
* Python/package/import problem: no
* GPU/CUDA problem: no
* Distributed/torchrun problem: no
* Filesystem/path problem: no
* Data/checkpoint/model file problem: yes
* Code logic problem: no
* Configuration problem: yes
* Resource problem: no
* Concurrency/race problem: no
* Unknown/insufficient evidence: no

Final classification: incomplete local Hugging Face snapshot; required model metadata file is missing.

## Hypotheses

### Hypothesis 1: The downloaded snapshot is incomplete
Why it could explain the symptom: `qaq.model_adapter._load_hf_config` first looks for `config.json` in the resolved local snapshot, then falls back to `AutoConfig.from_pretrained(..., local_files_only=True)`. Without `config.json`, Transformers cannot infer the Llama architecture.
Evidence for: `config.json` does not exist in `/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`. The snapshot contains safetensor shards, tokenizer files, and `model.safetensors.index.json`.
Evidence against: none found.
How to verify: download `config.json` into the same Hugging Face cache entry and rerun the metadata-only `AutoConfig.from_pretrained(..., local_files_only=True)` probe.

### Hypothesis 2: The earlier download command selected weights but missed metadata
Why it could explain the symptom: The model shards and tokenizer files are present, but `config.json` is not. A pattern-based or interrupted download can leave a cache snapshot that is usable for weight files but not for model metadata.
Evidence for: `hf download meta-llama/Llama-3.1-8B config.json --dry-run --json` reports that `config.json` exists on the Hub and is 826 bytes.
Evidence against: the exact prior download command output is not available in this diagnosis.
How to verify: run the explicit filename download command for `config.json`, then check that the snapshot contains a `config.json` symlink.

## Most Likely Root Cause
The base model cache is now present, but it is missing the required `config.json` metadata file. The repo did move past the previous "no local snapshot" failure. The new error is caused by a partial Hugging Face cache entry: QAQ/Transformers can see the snapshot directory, but cannot determine that it is a Llama model because the architecture config file is absent.

## Minimal Fix
Download the missing metadata file explicitly:

```bash
hf download meta-llama/Llama-3.1-8B config.json
```

Do not redownload all safetensor shards unless later verification shows they are corrupt or incomplete.

## Verification
```bash
ls -la /nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/config.json
python -c "from transformers import AutoConfig; p='/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b'; c=AutoConfig.from_pretrained(p, local_files_only=True); print(c.model_type, c.num_hidden_layers, c.hidden_size)"
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.prepare_bitplanes --model meta-llama/Llama-3.1-8B --output-dir runs/llama31_8b_bitplanes_sampled --sample-values 16 --overwrite --print-json
```

Expected verification result:

```text
config.json exists, the metadata probe prints llama 32 4096, and prepare_bitplanes proceeds past model metadata loading.
```
