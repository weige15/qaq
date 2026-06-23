# Requirements

## Functional Requirements

- Confirmed: The rebuild must target QAQ, "Query-adaptive Mixed-precision Quantization for Large Language Models", using the local `QAQ.pdf` as the primary source.
- Confirmed by user: Official QAQ code is private and not available for this rebuild; `QAQ.pdf` is the only supplementary material available to the project.
- Confirmed: The system must support dynamic precision selection per input query rather than only static, uniform quantization across all inputs.
- Confirmed: The system must decompose model weights into bit-planes so that a lower-precision or higher-precision reconstruction can be selected from the same maximum-bit representation.
- Confirmed: The bit-plane representation must support a maximum bit-width `B`, with the paper giving 8 bits as an example.
- Confirmed: The system must selectively use the most significant bit-planes for lower-precision inference.
- Confirmed: The system must include a trainable router that makes query-conditioned precision decisions.
- Confirmed: The router must produce a per-block importance score or probability distribution over candidate bit-widths.
- Confirmed: The router must be lightweight relative to the base LLM; the paper describes it as a lightweight MLP.
- Confirmed: Router training must use a full-precision teacher LLM and a quantized student LLM with a knowledge distillation loss.
- Confirmed: The base LLM parameters are frozen during router training; the router is trainable.
- Confirmed: Inference must support block-wise mixed precision over transformer structure.
- Inferred: The block abstraction should map to MHA and FFN blocks because Figure 1 describes transformer layers as MHA and FFN blocks quantized at block level.
- Needs user confirmation: If framework constraints make MHA/FFN-level quantization too costly for the first milestone, whole-layer granularity may be accepted only if documented as a simplification.
- Confirmed: The system must support an on-demand loading mode that transfers only router-selected bit-planes or precision variants from CPU memory into GPU memory when needed.
- Confirmed: The system must support a non-on-demand QAQ mode so that adaptive quantization can be evaluated without CPU-to-GPU loading overhead.
- Confirmed: The evaluation must compare QAQ against full FP16, static 8-bit, and static 4-bit baselines.
- Confirmed: The full paper reproduction target includes Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B.
- Confirmed by user: The first implementation target is LLaMA-3.1-8B.
- Inferred: Qwen3-4B and Qwen3-8B remain full paper-reproduction targets after the LLaMA-3.1-8B path is working.
- Confirmed: The full paper reproduction target includes HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande, WikiText-2, and PTB metrics.
- Assumption: The first implementation milestone may use a smaller benchmark subset, but it must include at least one accuracy metric and one perplexity or language-modeling metric if feasible.
- Confirmed: The evaluation must report accuracy or benchmark score, end-to-end latency, and GPU memory usage for every evaluated mode.
- Confirmed: Latency must be measured end-to-end for a single evaluation pass on WikiText-2 for paper-aligned reproduction.
- Inferred: The implementation must log enough routing decisions to verify that different queries or blocks can select different precisions.
- Confirmed by user: Training and inference must include logging or progress tracking.
- Inferred: Training progress logs must include at least current step or epoch, loss values, learning rate if applicable, elapsed time, and checkpoint/save events.
- Inferred: Inference/evaluation progress logs must include current benchmark or dataset progress, current mode, processed example count, elapsed time, latency summary, and memory summary.
- Confirmed by user: The project owner does not currently know how the router should be trained, so router training must be treated as an unresolved research/design requirement rather than an assumed implementation detail.
- Inferred: Before implementation begins, the proposal or design must choose and document a concrete router-training method that is consistent with the paper's teacher/student knowledge-distillation requirement.
- Unknown: The paper does not specify an official command-line interface, Python API, config schema, or repository layout.
- Unknown: The paper does not specify the exact router training dataset, distillation target, loss formula, optimizer, schedule, or hyperparameters.
- Unknown: The paper does not specify the exact candidate bit-width set beyond low/mid/high notation and an 8-bit maximum example.

## Non-Functional Requirements

- Confirmed: The primary trade-off to evaluate is accuracy versus GPU memory versus latency.
- Confirmed: QAQ should preserve accuracy comparable to static 8-bit quantization.
- Needs user confirmation: "Comparable" will be treated as within 1 percentage point on classification benchmarks or within 5 percent relative perplexity on language-modeling benchmarks until a stricter tolerance is chosen.
- Confirmed: QAQ with on-demand loading should reduce GPU memory footprint compared with static quantization or QAQ without on-demand loading.
- Needs user confirmation: The paper-level memory target should be at least a 5 percent reduction in peak GPU memory for on-demand QAQ, based on the reported 5.6 percent claim, unless hardware or framework constraints make this impossible.
- Confirmed: On-demand loading is allowed to increase latency, but the increase must be measured and reported.
- Confirmed: Sequential CPU-to-GPU transfer overhead is a known limitation; asynchronous prefetching is not required for the first milestone.
- Inferred: Results must be reproducible with pinned model identifiers, dataset versions, dependency versions, random seeds where applicable, and saved run configs.
- Inferred: The system must not claim paper reproduction unless the same model families, baseline modes, benchmark set, and metric types are evaluated.
- Assumption: The implementation may depend on common LLM tooling, such as PyTorch and Hugging Face libraries, unless later constraints forbid them.
- Confirmed by user: Available development hardware is 8 GPUs with the same specification, each an NVIDIA GeForce RTX 3090 with 24 GiB VRAM.
- Confirmed by user: A reported idle snapshot for device 0 showed 0.442 GiB / 24.000 GiB memory in use, 23 C temperature, 30 percent fan, and 21 W / 300 W power.
- Confirmed by user: The reported PCIe status for device 0 was GEN 1@16x with RX 390.0 KiB/s and TX 488.0 KiB/s at the time of observation.
- Inferred: The implementation must support CUDA execution on the available RTX 3090 GPUs and must record selected GPU IDs and per-GPU memory usage in run metadata.
- Confirmed by user: The first milestone does not have to use all 8 GPUs.
- Inferred: GPU selection must be configurable, and valid runs may use one GPU or a subset of the 8 available GPUs.
- Inferred: Multi-GPU execution may be used if useful for training or baseline evaluation, but it is not required for acceptance.
- Inferred: Full-scale LLaMA-3.1-8B experiments should fit the stated hardware for static quantized inference; FP16 teacher/student training or simultaneous teacher-student execution may require careful device placement across GPUs.
- Assumption: CPU-only execution may be acceptable for unit tests and small smoke tests, but not for validating QAQ's GPU memory and loading claims.
- Inferred: Evaluation scripts must avoid data leakage between router training/calibration data and held-out benchmark evaluation data.
- Inferred: The implementation must preserve enough numeric determinism to make repeated metric runs comparable, while accepting normal GPU nondeterminism.
- Confirmed by user: Progress tracking is required during both training and inference.
- Inferred: Progress tracking must be usable from a terminal session and must also write durable logs to disk so interrupted or long-running runs can be audited.
- Inferred: Logging must not materially distort latency measurements; timing-sensitive benchmarks should separate progress/logging overhead from measured inference latency where possible.

## Input Format

- Confirmed: The primary source input is a pretrained causal LLM checkpoint and its matching tokenizer.
- Confirmed: Paper-aligned model inputs are Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B checkpoints.
- Confirmed by user: Development should start with LLaMA-3.1-8B.
- Assumption: Smaller compatible causal LLMs may still be used for unit tests or smoke tests, but not as the primary rebuild target.
- Confirmed: Inference inputs are natural-language text queries or benchmark prompts encoded by the model tokenizer.
- Confirmed: Router decisions require query-dependent features, such as hidden representations at block `j`, but the exact feature extraction point is unspecified.
- Unknown: The official router training corpus, calibration set, and benchmark splits are not specified in the PDF.
- Needs user confirmation: A run configuration must identify the model, tokenizer, precision candidates, block granularity, QAQ mode, dataset, batch size, sequence length, device placement, output directory, and random seed.
- Inferred: A run configuration must also identify logging settings, progress reporting interval, checkpoint interval, and selected GPU IDs.
- Confirmed by user: The selected GPU IDs may specify any usable subset of the 8 RTX 3090 GPUs; specifying all 8 is not required.
- Assumption: Configuration may be provided through a machine-readable config file, CLI flags, or both; the exact format is not defined by the paper.
- Needs user confirmation: Candidate precision values must include static 4-bit and static 8-bit baselines, plus at least two adaptive choices for QAQ.
- Needs user confirmation: The default QAQ precision candidates should be defined before implementation, for example 4/6/8-bit or low/mid/high mapped to concrete bit-widths.
- Inferred: The dynamic loader input must include CPU-resident bit-planes or precision variants and metadata mapping each block to its available precision representation.
- Inferred: Evaluation inputs must identify benchmark names and splits so results can be compared across runs.

## Output Format

- Confirmed: Evaluation output must include benchmark score or perplexity, latency, and GPU memory usage for each mode.
- Confirmed: Required evaluation modes are full FP16, static 8-bit, static 4-bit, QAQ on-demand off, and QAQ on-demand on.
- Inferred: Output must include the model identifier, dataset, split, mode, precision candidates, block granularity, router checkpoint identifier if used, seed, device, and dependency/version metadata.
- Confirmed by user: Output must include training and inference logs or progress-tracking artifacts.
- Inferred: Durable logs must include run start/end time, model, mode, dataset, selected devices, progress counters, losses or metrics, latency summaries, memory summaries, warnings, and failure status when applicable.
- Inferred: QAQ output must include routing summaries, such as per-block precision selections or aggregate precision frequencies, sufficient to verify adaptive behavior.
- Inferred: On-demand output must include memory-transfer or loading summaries sufficient to verify that selected weights or bit-planes were loaded from CPU to GPU.
- Needs user confirmation: The canonical machine-readable results format should be JSON, JSONL, CSV, or a combination.
- Assumption: A results artifact should be machine-readable and stable enough for later plotting or table generation.
- Inferred: Human-readable logs may be printed to stdout/stderr, but machine-readable metrics must be written to an output path for comparison.
- Inferred: Router training output must include the trained router weights, training config, and training metrics.
- Inferred: Quantization output must include enough metadata to reconstruct which bit-planes or precision variants correspond to each model block.
- Unknown: The paper does not specify required filenames, directory names, numeric precision, table formatting, or packaging format.

## Edge Cases

- Confirmed: Harder queries may require higher precision to avoid quantization error; the router must be able to select higher precision when needed.
- Confirmed: Simpler queries may be processable with lower precision; the router must be able to select lower precision when appropriate.
- Inferred: If all queries choose the same precision, the run should be flagged because it no longer demonstrates query-adaptive behavior.
- Inferred: If all blocks choose the same precision, the run should be flagged because it does not demonstrate block-wise adaptation.
- Inferred: Router probability ties must be resolved deterministically when converting probabilities to discrete precision choices.
- Inferred: Invalid precision candidates, such as non-positive bit-widths or bit-widths above the stored maximum `B`, must be rejected.
- Inferred: Missing bit-plane data for a requested block or bit-width must be treated as an invalid artifact.
- Inferred: Empty input queries must not crash the evaluator; they should either be rejected with a clear error or processed according to tokenizer behavior.
- Inferred: Very long queries must respect the model context length and should be truncated or rejected according to the configured policy.
- Inferred: Unsupported model layers must be reported clearly instead of silently falling back to full precision.
- Inferred: CPU/GPU transfer timing may dominate latency in on-demand mode; this must be reported rather than hidden.
- Inferred: GPU memory measurement must be taken consistently across modes, including after relevant cache clearing or warm-up policy is applied.
- Inferred: Progress reporting must handle long-running training and evaluation without losing the current run state after terminal interruption or log rotation.
- Inferred: Multi-GPU logs must distinguish per-GPU memory usage and utilization instead of reporting only a single aggregate value.
- Unknown: The paper does not define behavior for batching multiple queries with different precision decisions.
- Unknown: The paper does not define behavior for generation workloads versus benchmark scoring/perplexity workloads.

## Error Behavior

- Inferred: Missing model, tokenizer, dataset, router checkpoint, or quantized artifact paths must produce a clear error and a non-zero exit status.
- Inferred: Invalid config values must fail before model loading where possible.
- Inferred: On-demand mode must fail clearly if CUDA/GPU transfer support is unavailable, unless an explicit CPU-only simulation mode is selected.
- Inferred: If GPU memory is insufficient, the run must fail with a clear out-of-memory message and must not report partial metrics as successful.
- Inferred: If a benchmark dependency or dataset is unavailable, the run must mark that benchmark as not run instead of fabricating a score.
- Inferred: If static 8-bit or static 4-bit baselines cannot run, QAQ results must not be presented as a valid comparison.
- Inferred: If router training fails or no router checkpoint exists, QAQ evaluation must either stop or explicitly use an untrained-router baseline labeled as such.
- Inferred: If no router-training method has been selected in the active config or design, training must fail before launching long-running jobs rather than silently using an undocumented heuristic.
- Inferred: If logging or progress tracking fails, the run must continue only if core metrics can still be written durably; otherwise it must fail clearly rather than silently losing training or inference state.
- Inferred: If a selected GPU is unavailable or has insufficient free VRAM, the run must fail clearly or select another configured GPU only when automatic device fallback is explicitly enabled.
- Inferred: Existing output directories must not be overwritten unless an explicit overwrite option is provided.
- Inferred: Malformed result files from interrupted runs must be marked incomplete.
- Unknown: The paper does not specify official error messages, exit codes, retry behavior, or partial-result behavior.

## Acceptance Criteria

- Confirmed: The project has a requirements document at `doc/requirements.md` before proposal, design, or implementation work begins.
- Confirmed: A completed rebuild must demonstrate the three QAQ components from the paper: bit-plane weight representation, trainable query-conditioned router, and on-demand CPU-to-GPU loading.
- Confirmed: A completed rebuild must evaluate full FP16, static 8-bit, static 4-bit, QAQ on-demand off, and QAQ on-demand on modes.
- Confirmed: A completed rebuild must report accuracy or benchmark score, latency, and GPU memory for every evaluated mode.
- Confirmed: Full paper-aligned reproduction must evaluate Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B on HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande, WikiText-2, and PTB.
- Confirmed by user: First-milestone model acceptance is based on LLaMA-3.1-8B.
- Assumption: First-milestone acceptance requires LLaMA-3.1-8B, at least one held-out evaluation task, and all five required modes if hardware permits.
- Confirmed by user: First-milestone acceptance does not require using all 8 GPUs.
- Assumption: First-milestone acceptance requires unit tests or equivalent checks showing that bit-plane reconstruction at lower and higher precision produces expected tensor shapes and numerically plausible reconstructions.
- Assumption: First-milestone acceptance requires a router test showing valid probability distributions or precision choices for every configured block.
- Assumption: First-milestone acceptance requires an evaluation artifact that proves QAQ routing decisions vary by query or block on at least one run.
- Inferred: First-milestone acceptance requires a documented router-training strategy before any trained-router result is claimed as QAQ.
- Confirmed by user: First-milestone acceptance requires logging or progress tracking during training and inference.
- Inferred: Acceptance requires durable training logs, durable inference/evaluation logs, and machine-readable metrics for at least one LLaMA-3.1-8B run.
- Needs user confirmation: QAQ accuracy acceptance should be within 1 percentage point of static 8-bit for classification or within 5 percent relative perplexity for language modeling unless a different tolerance is selected.
- Needs user confirmation: QAQ on-demand memory acceptance should target at least 5 percent lower peak GPU memory than the comparable non-on-demand mode unless the run is explicitly marked as a constrained-hardware exception.
- Confirmed: Latency overhead in QAQ on-demand mode is acceptable only if reported and compared against non-on-demand QAQ and static baselines.
- Inferred: Results are not accepted if static baselines and QAQ modes use different model checkpoints, datasets, prompts, tokenization, or metric implementations.
- Inferred: Results are not accepted if generated reports omit config, seed, model, dataset, mode, or hardware metadata needed to reproduce the run.

## Examples

No official example found in the repository or `QAQ.pdf`.

Minimal inferred run shape, subject to implementation design:

```bash
python -m qaq.evaluate --config configs/example.yaml --mode qaq_on_demand
```

Minimal inferred config fields, subject to user confirmation:

```yaml
model: <model-id-or-path>
tokenizer: <tokenizer-id-or-path>
dataset: <benchmark-name>
precision_candidates: [4, 8]
block_granularity: mha_ffn
mode: qaq_on_demand
device: cuda
gpu_ids: [0]
seed: 0
logging:
  progress_interval_steps: 10
  checkpoint_interval_steps: 500
output_dir: runs/example
```

Minimal inferred result fields:

```json
{
  "model": "<model-id-or-path>",
  "dataset": "<benchmark-name>",
  "mode": "qaq_on_demand",
  "score": 0.0,
  "latency_seconds": 0.0,
  "peak_gpu_memory_gb": 0.0,
  "routing_summary": {},
  "hardware": {
    "gpu_model": "NVIDIA GeForce RTX 3090",
    "gpu_count": 8,
    "selected_gpu_ids": [0]
  },
  "logs": {
    "training_log": "runs/example/train.log",
    "inference_log": "runs/example/eval.log"
  },
  "seed": 0
}
```
