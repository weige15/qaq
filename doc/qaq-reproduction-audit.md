# QAQ Reproduction Audit

## Paper Requirements

From `QAQ.pdf`, reproducing QAQ requires the following components as paper semantics, not just similarly named code paths:

- **Bit-plane decomposition**: decompose each LLM weight tensor into bit-planes and reconstruct/select precision from the most significant planes.
- **Trainable router**: train a lightweight query-conditioned router that predicts precision sensitivity from query/block representations.
- **Block-wise precision decision**: choose candidate bit-widths per transformer block, specifically at MHA/FFN granularity in this repository.
- **Static 8-bit baseline**: run the model with a fixed 8-bit quantized execution path comparable to QAQ.
- **Static 4-bit baseline**: run the model with a fixed 4-bit quantized execution path comparable to QAQ.
- **QAQ on-demand off**: use router-selected block precision with selected bit-plane weights resident for inference, without CPU-to-GPU on-demand loading.
- **QAQ on-demand on**: use router-selected block precision and dynamically load only selected CPU-resident bit-planes to GPU.
- **Latency and GPU memory measurement**: report comparable end-to-end latency and peak GPU memory across FP16, static 8-bit, static 4-bit, QAQ on-demand off, and QAQ on-demand on, matching the kind of claims in Table 1.

## Current Implementation Behavior

### `fp16`

- **Code path**: `qaq.evaluate` dispatches non-QAQ modes to `qaq.runtime.static.run_static_runtime`. For `fp16`, `require_artifacts` is false, no bit-plane materialization is attempted, and the adapter runs `reference_forward`.
- **Real bit-plane artifacts used**: No.
- **Full FP16/BF16 weights resident**: Yes for the Hugging Face adapter. `HuggingFaceCausalLMAdapter._ensure_model_loaded()` loads `AutoModelForCausalLM.from_pretrained(...)` with dtype resolved to the normal HF torch dtype path, then moves the model to CPU/CUDA or uses HF `device_map`.
- **Selected low-bit weights used for actual computation**: No.
- **Optimized low-bit GEMM or equivalent low-bit execution exists**: No.
- **CPU-GPU on-demand loading affects real model residency**: No. This mode does not use the on-demand loader.

### `static_8bit`

- **Code path**: `qaq.evaluate` dispatches to `run_static_runtime`. The runtime builds a static precision plan, materializes bit-plane artifacts with `_materialize_plan_artifacts`, then, only if the adapter supports overrides and every block has a full tensor artifact index, calls `build_weight_overrides`. The forward is still `adapter.reference_forward(...)`.
- **Real bit-plane artifacts used**: Partially yes. Full tensor-native artifacts can be loaded and reconstructed when a `full_tensor_index` is supplied. Legacy or partial artifact indexes can be materialized for records but are not accepted for real mixed-forward execution.
- **Full FP16/BF16 weights resident**: Yes. The HF model is loaded as a normal full model. `_temporary_weight_overrides` stores the original `parameter.data`, assigns reconstructed tensors to parameters during the forward, and restores the originals afterward.
- **Selected low-bit weights used for actual computation**: Only after dequantization/reconstruction into regular torch tensors. The selected bit-plane artifact determines the temporary floating-point tensor value, but the matrix multiply is still the ordinary HF model forward using the parameter dtype/device.
- **Optimized low-bit GEMM or equivalent low-bit execution exists**: No. There is no INT8/INT4 packed-kernel path, custom CUDA kernel, bitsandbytes/GPTQ/AWQ backend, or fused bit-plane GEMM.
- **CPU-GPU on-demand loading affects real model residency**: No. Static mode does not use `OnDemandLoader`, and full HF weights remain resident.

### `static_4bit`

- **Code path**: Same as `static_8bit`, except `build_precision_plan` selects 4-bit for all applicable blocks.
- **Real bit-plane artifacts used**: Partially yes under the same conditions as `static_8bit`: full tensor-native artifact indexes can reconstruct 4-bit-selected weights.
- **Full FP16/BF16 weights resident**: Yes. The base HF model remains loaded, and temporary overrides are cast back to the original parameter dtype in `_temporary_weight_overrides`.
- **Selected low-bit weights used for actual computation**: Only as dequantized temporary tensors in the normal HF forward. The computation is not low-bit.
- **Optimized low-bit GEMM or equivalent low-bit execution exists**: No.
- **CPU-GPU on-demand loading affects real model residency**: No.

### `qaq_on_demand_off`

- **Code path**: `qaq.evaluate` dispatches QAQ modes to `qaq.runtime.adaptive.run_adaptive_runtime`. The runtime first runs a normal reference forward with hidden-state collection, routes those hidden states through the checkpoint, validates selected artifacts, materializes selected plans without `OnDemandLoader`, then, if possible, runs per-query `adapter.reference_forward(...)` with `build_weight_overrides`-generated temporary weight overrides.
- **Real bit-plane artifacts used**: Partially yes. Runtime-selected artifacts can be loaded and reconstructed per routed plan when full tensor-native artifact indexes are present.
- **Full FP16/BF16 weights resident**: Yes. The routing pass and final pass use the normal HF model. The base model remains resident and temporary override tensors are swapped into `parameter.data` only for the forward.
- **Selected low-bit weights used for actual computation**: Only as reconstructed/dequantized tensors during the ordinary HF forward. The router changes which reconstructed tensors are temporarily assigned, but execution remains floating-point HF computation.
- **Optimized low-bit GEMM or equivalent low-bit execution exists**: No.
- **CPU-GPU on-demand loading affects real model residency**: No. This mode does not use `OnDemandLoader`; it simulates GPU-resident selected materialization in metadata.

### `qaq_on_demand_on`

- **Code path**: Same adaptive routing setup as `qaq_on_demand_off`, but `_materialize_adaptive_plans` calls `OnDemandLoader.load(...)` for each selected block/tensor and records loader summary/events. After that, `_run_adaptive_weight_overridden_forward` independently calls `build_weight_overrides`, which reloads/reconstructs artifact tensors again and runs `adapter.reference_forward(...)`.
- **Real bit-plane artifacts used**: Partially yes. `OnDemandLoader` can load selected packed bit-planes to CPU or CUDA tensors, and `build_weight_overrides` can reconstruct selected weights from full tensor artifacts.
- **Full FP16/BF16 weights resident**: Yes. The normal HF model remains loaded for routing and final forward. The loader does not replace the HF model residency model.
- **Selected low-bit weights used for actual computation**: Not as low-bit tensors. Loader materialization does not feed the actual forward path. The final forward uses reconstructed floating-point temporary weight overrides through the ordinary HF model.
- **Optimized low-bit GEMM or equivalent low-bit execution exists**: No.
- **CPU-GPU on-demand loading affects real model residency**: No, not in the paper sense. The loader records selected plane loads and resident bytes, but the HF model still keeps full weights resident, and the final forward rebuilds separate floating-point overrides instead of consuming loader-resident bit-planes.

## Reproduction Gap

- **Correctness gap**: The current static and QAQ modes do not execute quantized low-bit layers. They reconstruct selected bit-plane weights into floating-point tensors and call the ordinary HF forward. This can test value substitution, not true QAQ quantized execution semantics.
- **Correctness gap**: `qaq_on_demand_on` records loader activity, but loader outputs are not the tensors consumed by the final model forward. The execution path and loader path are not integrated.
- **Correctness gap**: Adaptive routing is based on a preliminary full-model reference forward to collect hidden states before running the selected-weight forward. The paper describes the router as part of the dynamic inference system; current execution pays for a full reference pass first.
- **Memory gap**: Full FP16/BF16 HF weights remain resident. Temporary overrides additionally keep original parameter data references alive during the override context. This cannot support Table 1-style claims that static/QAQ modes reduce model residency relative to FP16.
- **Memory gap**: `OnDemandLoader` tracks bytes for selected bit-plane planes, but it does not control actual model parameter residency or unload full precision weights from the HF model.
- **Latency gap**: Timings include standard HF forwards and Python-level artifact reconstruction/parameter swapping, not a production quantized backend. They cannot establish QAQ latency improvements or the paper's on-demand latency tradeoff.
- **Kernel/backend gap**: There is no optimized INT4/INT8 GEMM, bit-plane GEMM, fused dequantization kernel, custom CUDA backend, or integration with an existing quantized inference backend.
- **Router training gap**: Router training exists, but the target construction is an implementation assumption: it uses teacher/student reference logits plus reconstruction distortion and bit-cost terms. It is not yet demonstrated as the paper's knowledge-distillation training against actual quantized student execution at scale.
- **Benchmark gap**: The docs state that accepted full LLaMA-3.1-8B execution, full five-mode matrix, real paper benchmark adapters including PTB, Qwen3 paths, and accepted latency/GPU memory comparisons are not yet implemented.

## Required Fixes Before Claiming QAQ Reproduction

1. **Implement a true quantized execution backend** for static 8-bit and static 4-bit baselines. The selected weights must remain packed or quantized through the compute path, with INT8/INT4 kernels or an equivalent low-bit backend.
2. **Integrate bit-plane artifacts with real layer execution** so reconstructed or selected bit-planes drive MHA/FFN computation without falling back to full-model floating-point HF parameters.
3. **Remove full FP16/BF16 residency from quantized modes** or isolate it as an explicit teacher/reference-only path. Static and QAQ result runs must not keep the full precision model resident while claiming quantized memory usage.
4. **Wire `OnDemandLoader` into the actual forward path**. `qaq_on_demand_on` must execute from loader-resident selected planes or quantized tensors, and loader residency must govern what is actually present on GPU.
5. **Implement block-wise dynamic precision inside the model runtime** so each routed MHA/FFN block executes at the selected precision during the same inference path.
6. **Train and validate the router against the real quantized student path**, using real held-out benchmark examples and a documented distillation objective aligned with QAQ.
7. **Generate accepted full tensor-native artifacts** for every required tensor/block of each reproduced model, with no sampled/truncated artifacts.
8. **Run a complete comparable five-mode matrix** for each target model/benchmark: `fp16`, `static_8bit`, `static_4bit`, `qaq_on_demand_off`, and `qaq_on_demand_on`, with identical model, tokenizer, dataset split, prompt format, precision candidates, block granularity, metric, and seed.
9. **Measure latency and GPU memory on the lab RTX 3090 server** through `scripts/gpu_run.py`, recording hostname, selected physical GPU IDs, CUDA mapping, command, commit, environment, dataset path, output path, and metrics.
10. **Add acceptance tests or validators for backend semantics**, specifically proving that quantized modes do not call the ordinary full-weight HF forward with temporary floating-point parameter overrides and that on-demand loader residency affects the tensors used by computation.

## Conclusion

`static_8bit`, `static_4bit`, `qaq_on_demand_off`, and `qaq_on_demand_on` currently do **not** implement QAQ paper reproduction semantics. They can use bit-plane artifacts to reconstruct selected weights, and the adaptive modes can use a trained router checkpoint to choose per-block precisions, but the actual model computation is ordinary Hugging Face FP16/BF16 forward through temporary floating-point weight overrides. The current implementation cannot honestly support the latency or GPU memory claims in Table 1 of the QAQ paper.
