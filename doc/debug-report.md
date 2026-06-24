# Debug Report

## Symptom
The GPU-wrapped HellaSwag FP16 LLaMA evaluation selected an RTX 3090 successfully, then exited immediately with `Invalid device argument` before producing the requested result artifact.

## Reproduction Command
Working directory: `/nfs/home/s314511048/qaq`
Shell: `bash`
Runtime: `Python 3.12.3` from `/nfs/home/s314511048/.venv/bin/python`
Environment: active `.venv`; host `basic-2`
Relevant environment variables:
```text
Parent process CUDA_VISIBLE_DEVICES=None
Parent process QAQ_SELECTED_PHYSICAL_GPUS=None
Child process CUDA_VISIBLE_DEVICES=1
Child process QAQ_SELECTED_PHYSICAL_GPUS=1
```

```bash
python scripts/gpu_run.py --count 1 --min-free-mb 18000 --status-file runs/gpu-selector/hellaswag-fp16-debug-subset.json -- python -m qaq.evaluate --config configs/benchmarks/llama_first_milestone/hellaswag/fp16.json --skip-output-dir-check --max-examples 128 --eval-batch-size 1 --hf-device-map single --result-output runs/llama_first_milestone/hellaswag/fp16_debug_subset/result_artifact.json --print-result-json
```

## Expected Behavior
The selector should choose a free physical RTX 3090, expose it to the child as logical `cuda:0`, load the local LLaMA checkpoint, run the 128-example HellaSwag subset, and write a result artifact.

## Actual Behavior
The selector chose physical GPU 1 and mapped it to child `cuda:0`, but the evaluator failed before model loading with `Invalid device argument`.

## Error Log
```text
Traceback from direct runtime reproduction:
  File "/nfs/home/s314511048/qaq/qaq/runtime/static.py", line 55, in run_static_runtime
    _reset_cuda_peak_memory_if_available(config)
  File "/nfs/home/s314511048/qaq/qaq/runtime/static.py", line 394, in _reset_cuda_peak_memory_if_available
    torch.cuda.reset_peak_memory_stats(index)
  File "/nfs/home/s314511048/.venv/lib/python3.12/site-packages/torch/cuda/memory.py", line 322, in reset_peak_memory_stats
    return torch._C._cuda_resetPeakMemoryStats(device)
RuntimeError: Invalid device argument
```

## Failure Layer Classification
Most likely layer:

* Command problem: no
* Permission problem: no
* Shell/script invocation problem: no
* Environment problem: no
* Dependency problem: no
* Python/package/import problem: no
* GPU/CUDA problem: yes
* Distributed/torchrun problem: no
* Filesystem/path problem: no
* Data/checkpoint/model file problem: no
* Code logic problem: yes
* Configuration problem: no
* Resource problem: no
* Concurrency/race problem: no
* Unknown/insufficient evidence: no

Final classification: CUDA memory-stat handling in the runtime code. The selected GPU and logical CUDA mapping were valid.

## Hypotheses

### Hypothesis 1: Wrong physical GPU ID was passed into PyTorch
Why it could explain the symptom: `gpu_run.py` selected physical GPU 1, while PyTorch inside the child only sees it as logical `cuda:0`.
Evidence for: physical/logical remapping is active: `CUDA_VISIBLE_DEVICES=1`, `pytorch_logical_mapping={"cuda:0": 1}`.
Evidence against: the config uses `gpu_ids: [0]`, and a selector probe successfully allocated a tensor on child `cuda:0`.
How to verify: run a GPU selector probe that allocates `torch.empty(..., device="cuda:0")`. This passed.

### Hypothesis 2: PyTorch rejects startup peak-memory reset before allocator state exists
Why it could explain the symptom: the direct traceback points to `torch.cuda.reset_peak_memory_stats(0)` before LLaMA loading or inference.
Evidence for: bypassing `qaq.evaluate` showed the exact failing call; a one-example run passed after making the reset path tolerant of this exception.
Evidence against: none after traceback capture.
How to verify: run the same runtime path with one real HellaSwag example under `gpu_run`. This passed after the fix.

## Most Likely Root Cause
`qaq.runtime.static` and `qaq.runtime.adaptive` treated CUDA peak-memory reset as mandatory. On this PyTorch/CUDA/server combination, `torch.cuda.reset_peak_memory_stats(0)` can raise built-in `RuntimeError: Invalid device argument` even when the selected logical CUDA device is valid and usable. The modules also import the project `RuntimeError` class, so a first attempt to catch `RuntimeError` would have caught the wrong class unless it explicitly targeted `builtins.RuntimeError`.

## Minimal Fix
Keep CUDA memory measurement, but make startup memory-stat reset and peak-stat reads tolerant only of PyTorch's `invalid device argument` memory-stat failure. Re-raise all other runtime errors so real CUDA/model failures remain visible.

Applied changes:

* `qaq/runtime/static.py`: added safe wrappers around `reset_peak_memory_stats` and `max_memory_allocated`.
* `qaq/runtime/adaptive.py`: applied the same safe wrappers for adaptive CUDA paths.
* `tests/unit/test_runtime_cuda_memory_stats.py`: added regression tests for the PyTorch exception and for re-raising unrelated errors.

## Verification
```bash
python -m pytest -q tests/unit/test_runtime_cuda_memory_stats.py
python -m py_compile qaq/runtime/static.py qaq/runtime/adaptive.py tests/unit/test_runtime_cuda_memory_stats.py
python scripts/gpu_run.py --count 1 --min-free-mb 18000 --status-file runs/gpu-selector/hellaswag-fp16-trace-one-after-fix.json -- python -c "from dataclasses import replace; from qaq.config import load_config_file; from qaq.runtime.static import run_static_runtime; from qaq.results import build_result_artifact; c=load_config_file('configs/benchmarks/llama_first_milestone/hellaswag/fp16.json', validate_output=False); c=replace(c, max_examples=1, eval_batch_size=1, hf_device_map='single'); r=run_static_runtime(c); a=build_result_artifact(c,r); print(a.as_dict()['metrics']); print(r.metadata.get('model_device_map'), r.metadata.get('peak_gpu_memory_gb'))"
python scripts/gpu_run.py --count 1 --min-free-mb 18000 --status-file runs/gpu-selector/hellaswag-fp16-debug-subset.json -- python -m qaq.evaluate --config configs/benchmarks/llama_first_milestone/hellaswag/fp16.json --skip-output-dir-check --max-examples 128 --eval-batch-size 1 --hf-device-map single --result-output runs/llama_first_milestone/hellaswag/fp16_debug_subset/result_artifact.json --print-result-json
```

Expected verification result:

```text
Focused regression tests pass.
Changed files compile.
One-example LLaMA/HellaSwag runtime completes on physical GPU 1 mapped to cuda:0.
The requested 128-example command completes and writes runs/llama_first_milestone/hellaswag/fp16_debug_subset/result_artifact.json.
```

Actual verification result:

```text
3 passed in 0.11s.
py_compile passed.
One-example GPU run completed, loaded LLaMA, used cuda:0, and recorded peak_gpu_memory_gb=15.002574920654297.
The requested 128-example command completed, processed 128 / 10042 HellaSwag validation examples, wrote the result artifact, selected physical GPU 1, mapped cuda:0 -> physical 1, recorded peak_gpu_memory_gb=15.09602975845337, and rejected the artifact as accepted QAQ evidence for diagnostic_result and subset_debug_run.
```
