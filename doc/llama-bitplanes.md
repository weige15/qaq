# LLaMA Bit-Plane Artifacts

`python -m qaq.llama_bitplanes` generates QAQ bit-plane artifacts from a local
Hugging Face LLaMA checkpoint without loading the full model for inference. It
streams tensors from local safetensor shards and writes either JSON artifacts
or tensor-native safetensors artifacts:

- per-tensor QAQ bit-plane artifact JSON files, or `.qaq.safetensors`
  artifacts with packed `torch.uint8` quantized values and QAQ metadata
- `tensor_artifact_index.json` for router training over all generated tensors
- `runtime_artifact_index.json` for the current per-block runtime materializer
- `generation_manifest.json` with source tensor shapes, dtype, element counts,
  truncation status, storage layout, artifact format, and source snapshot

The generated indexes contain absolute artifact paths so they can be consumed
from router-training configs without being resolved relative to the index file.

Probe command for a small real-weight artifact sample. Because this touches
real LLaMA weight files, launch it through the lab-server GPU selector:

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

`--max-elements-per-tensor` creates truncated artifacts from real LLaMA tensor
values. These are useful for format, metadata, loader, and router-objective
acceptance checks, but they are not full-model QAQ artifacts.

Full tensor-native artifact generation for a bounded tensor can be run without
the JSON guard:

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 1000 -- python -m qaq.llama_bitplanes \
  --model meta-llama/Llama-3.1-8B \
  --artifact-format safetensors \
  --output-dir runs/llama31_8b_native_bitplanes_full_one_tensor \
  --block-limit 1 \
  --tensor-limit-per-block 1 \
  --max-elements-per-tensor 0 \
  --overwrite \
  --print-json
```

The native format stores one `quantized_values` tensor in safetensors metadata
layout `packed_uint8_bitplanes`. Runtime materialization packs the selected MSB
planes on demand for CPU or CUDA targets. JSON remains available for small
fixtures and backwards compatibility; full JSON generation is still guarded by
`--allow-full-tensor-json`.

For paper-scale runs, the remaining work is not artifact storage format itself.
The unresolved scale step is generating all controlled-block artifacts and
running router training/evaluation on the intended GPU setup.
