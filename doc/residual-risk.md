# Residual Risk

## Current Router Training Status

Router training now uses a minimal real implementation with `router_cost_cross_entropy`. It loads file-backed samples, validates bit-plane artifacts, produces cost-derived router targets, writes a target audit artifact, updates router parameters, saves reloadable checkpoints, emits validation metrics, records non-diagnostic acceptance runs consistently in the manifest and checkpoint metadata, and has a checkpoint-loaded evaluation command.

## Remaining Research Limitations

- The official QAQ router loss, calibration corpus, and hyperparameters remain unavailable in the PDF and official code is not available in this repo.
- The current objective estimates quantized-student behavior from bit-plane reconstruction distortion rather than executing a full quantized transformer block.
- Local router-training acceptance uses small dependency-free fixtures. It is implementation evidence, not a paper-scale LLaMA/Qwen reproduction.
- Llama 3.1 8B base metadata and local safetensor weights can be discovered through the Hugging Face adapter. `qaq.prepare_bitplanes` can prepare trainer-compatible sampled real-weight artifacts for all 64 controlled MHA/FFN blocks, and `qaq.llama_bitplanes --artifact-format safetensors` can write tensor-native `.qaq.safetensors` artifacts without JSON expansion. The verified native run includes a full 16,777,216-element Llama q-projection tensor, but not the complete model artifact set.
- Full Llama router training still requires a GPU-backed reference forward run. For identical teacher/student model refs, the trainer now uses a shared frozen reference adapter and the preflight reports at least 15.46 GiB free before activations for the base Llama 3.1 8B sampled-artifact config. The current visible GPU is a 6 GiB RTX 4050, so the run still cannot complete locally. Distinct teacher/student model refs still require separate model loads unless another execution strategy is implemented.
- CUDA on-demand materialization exists for selected JSON and tensor-native bit-plane tensors when torch can access CUDA. GPU memory and transfer claims still require a full QAQ runtime path that applies the materialized tensors to the model, plus the intended RTX 3090 hardware. The currently visible escalated device is a single 6 GiB RTX 4050, not the 8 RTX 3090 setup in `doc/requirements.md`.

These are true research and scale limitations, not blockers for the local minimal router-training implementation.
