# Detailed Design

## Purpose

This document defines the implementation design for rebuilding QAQ, a query-adaptive mixed-precision quantization research prototype for LLM inference. It translates `QAQ.pdf`, `doc/proposal.md`, `doc/high-level-design.md`, `doc/requirements.md`, `doc/problem-brief.md`, and `doc/test-plan.md` into module responsibilities, data contracts, algorithm contracts, failure handling, and independently testable work packages. Official QAQ code is private and unavailable; `QAQ.pdf` is the only supplementary source material available to this rebuild.

This is a design document only. It does not implement code.

## Source Proposal Summary

The proposal frames QAQ as a staged rebuild of the paper's central behavior:

- Establish reproducible FP16, static 8-bit, static 4-bit, and fixed mixed-precision baselines.
- Build bit-plane-backed quantized weights so lower and higher effective bit-widths can be reconstructed from one maximum-bit representation.
- Add block-level mixed precision before learned routing so execution correctness is separable from router quality.
- Train a lightweight router using full-precision teacher and quantized student signals.
- Evaluate adaptive QAQ with on-demand loading disabled first, then enable CPU-to-GPU on-demand loading.
- Report accuracy or perplexity, latency, peak GPU memory, routing behavior, and loader behavior for all required comparison modes.

The proposal explicitly treats production serving, asynchronous prefetching, and exact paper-number reproduction as non-goals for the first implementation stage.

## HLD Summary

The HLD defines a reproducible experiment pipeline with these modules:

- Experiment Configuration and Run Manifest
- Model and Benchmark Adapter
- Block Registry and Precision Plan
- Quantization and Bit-Plane Store
- Static and Fixed Mixed-Precision Runtime
- Router Policy Module
- Router Training Pipeline
- Adaptive Inference Runtime
- Dynamic Loader and Memory Residency Manager
- Evaluation Metrics and Results Reporter
- Logging and Progress Tracking

The HLD fixes the first implementation target as LLaMA-3.1-8B on a configurable subset of the available RTX 3090 GPUs. Full reproduction remains Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B on HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande, WikiText-2, and PTB. The primary block boundary is MHA and FFN, matching the QAQ figure. Whole-layer control is only a documented fallback if MHA/FFN-level control proves infeasible.

## Design Goals

- Preserve the paper-defined QAQ components: bit-plane weight representation, trainable query-conditioned router, block-wise mixed precision, and optional on-demand CPU-to-GPU loading.
- Keep static baselines mandatory and comparable to QAQ modes.
- Make all run settings, artifacts, routing decisions, loader events, and results reproducible through configs, manifests, logs, and machine-readable outputs.
- Keep module boundaries independently testable with small tensors, fake models, and simulated devices before large GPU runs.
- Fail clearly when a run cannot support a valid QAQ comparison.
- Record unresolved paper details as assumptions or open questions instead of silently choosing hidden behavior.

## Non-Goals

- Production serving, autoscaling, scheduling, UI, or API service design.
- Training or fine-tuning base LLM weights.
- Asynchronous prefetching or advanced memory scheduling in the first rebuild.
- Claiming full paper reproduction before all paper model families, benchmarks, modes, and metric types are evaluated or deviations are clearly labeled.
- Treating fixed or heuristic routing as an accepted QAQ result unless it is explicitly labeled as diagnostic.

## Architecture Overview

QAQ is organized as a staged experiment pipeline:

1. Configuration validates the run and writes an immutable manifest.
2. The model and benchmark adapter loads the tokenizer, model, datasets, prompt formatting, and hidden-state access.
3. The block registry maps model internals into stable MHA/FFN controlled blocks.
4. The bit-plane store creates or loads quantized artifacts and reconstructs requested bit-widths.
5. Static and fixed mixed-precision runtime validates FP16, static 8-bit, static 4-bit, and fixed profiles.
6. Router training learns only router parameters from teacher/student signals.
7. Router policy emits per-query, per-block precision probabilities and deterministic decisions.
8. Adaptive runtime applies the precision plan with on-demand loading disabled or enabled.
9. Dynamic loader handles CPU-resident bit-planes and GPU materialization in on-demand mode.
10. Evaluation reporter aggregates metrics across modes and rejects invalid comparisons.
11. Logging tracks long-running work and persists failure or incomplete-run state.

Dependency direction is from configuration and model/block metadata toward quantization, training, runtime, and evaluation. Evaluation consumes runtime outputs; it does not own execution behavior. Logging receives events from all long-running modules but must not redefine latency measurement windows.

## Shared Data Contracts

Exact class names, filenames, and serialization formats are still open until package layout and tooling are approved. The contracts below define required fields and semantics independent of implementation names.

### Run Configuration

Required fields:

- `model`: model ID or local path.
- `tokenizer`: tokenizer ID or local path, or an explicit instruction to use the model tokenizer.
- `dataset`: benchmark or dataset name.
- `split`: dataset split or benchmark subset.
- `mode`: one of `fp16`, `static_8bit`, `static_4bit`, `fixed_mixed`, `qaq_on_demand_off`, or `qaq_on_demand_on`.
- `precision_candidates`: positive bit-widths not exceeding `max_bit_width`.
- `max_bit_width`: maximum bit-plane width, with 8 as the paper example.
- `block_granularity`: expected first value is MHA/FFN; whole-layer is only a documented fallback.
- `gpu_ids`: selected GPU IDs; first milestone does not require all 8 GPUs.
- `seed`: run seed where deterministic behavior is configurable.
- `output_dir`: run output location.
- `overwrite`: explicit output reuse policy.
- `logging`: console monitor settings, progress interval, durable log paths, and checkpoint interval where applicable.
- `router_checkpoint`: required for trained-router QAQ evaluation unless the mode is explicitly diagnostic.

### Run Manifest

Required fields:

- Resolved config values.
- Model, tokenizer, dataset, split, and prompt formatting identifiers.
- Hardware metadata, including selected GPU IDs and per-GPU memory metadata when available.
- Dependency metadata where available.
- Artifact references for bit-planes, router checkpoint, logs, metrics, and result summaries.
- Start time, completion status, failure status, and incomplete-run marker when applicable.

### Block Descriptor

Required fields:

- Stable `block_id`.
- Layer index or model-order position.
- Block type, expected to include `mha` and `ffn`.
- Source module path or model location.
- Owned weight tensor references or tensor names.
- Supported bit-widths.
- Quantized artifact references.
- Validation status.

Block IDs must be stable across repeated discovery for the same model, checkpoint, block granularity, and implementation version.

### Precision Plan

Required fields:

- Query or example identifier.
- Runtime mode.
- Candidate bit-widths.
- Per-block precision decision.
- Decision source: static, fixed profile, trained router, or explicitly diagnostic router.
- Router checkpoint and temperature when routing is active.
- Deterministic tie-break policy reference.

### Bit-Plane Artifact Metadata

Required fields:

- Model identity and checkpoint reference.
- Block ID and tensor identity.
- Original tensor shape and dtype.
- Quantized representation metadata.
- Maximum bit-width `B`.
- Available bit-plane indices.
- Reconstruction policy.
- Artifact version.
- Checksum or integrity marker where tooling supports it.
- Validation status.

The exact signed quantization scheme, scale shape, zero-point handling, and group size are unresolved implementation choices. They must be documented before artifact compatibility is treated as stable.

### Router Trace

Required fields:

- Query or example identifier.
- Block ID.
- Candidate bit-widths.
- Router feature source.
- Raw scores where available.
- Probability distribution over candidate bit-widths.
- Selected bit-width.
- Temperature.
- Tie-break result if a tie occurs.
- Router checkpoint identifier.

### Loader Event

Required fields:

- Query or runtime step identifier.
- Block ID and requested bit-width.
- Requested bit-plane subset or derived precision artifact.
- Source residency, expected to be CPU in on-demand mode.
- Target GPU ID or device.
- Transfer start/end timing or duration.
- Result: loaded, cache hit, skipped, released, or failed.
- Failure reason when applicable.

### Result Artifact

Required fields:

- Model, tokenizer, dataset, split, prompt formatting, and mode.
- Precision candidates, block granularity, seed, selected GPU IDs, and hardware metadata.
- Router checkpoint if used.
- Score or perplexity as applicable.
- End-to-end latency.
- Peak GPU memory and per-GPU memory where relevant.
- Routing summary for QAQ modes.
- Loader summary for `qaq_on_demand_on`.
- Log paths.
- Completion status.

The canonical machine-readable result format is open. The test plan allows JSON, JSONL, CSV, or a combination, but tests must validate whichever format is selected.

## Module Designs

### Experiment Configuration and Run Manifest

#### Responsibility

Validate user-provided run settings, resolve environment and hardware metadata, prevent invalid or unsafe runs before expensive model loading, and write an immutable run manifest.

#### Non-Responsibility

This module does not load model weights, create bit-plane artifacts, train routers, execute inference, or compute benchmark metrics.

#### Inputs and Outputs

Inputs:

- Config file and optional CLI overrides if CLI support is adopted.
- Environment metadata and GPU availability.
- Existing output directory state.

Outputs:

- Validated run configuration.
- Immutable run manifest.
- Early validation errors.

#### Public Interface

Must expose these capabilities:

- Load and merge configuration inputs.
- Validate required fields and cross-field constraints.
- Resolve selected devices and output paths.
- Write a run manifest before long-running work begins.
- Mark run completion, failure, or incompleteness.

Exact Python API names and CLI command names are open.

#### Data Structures

- Run Configuration.
- Run Manifest.
- Validation error categories.
- Output directory state record.

#### Internal Design

Validation should happen in layers:

1. Parse user config.
2. Validate schema-level requirements.
3. Validate semantic constraints, including mode, precision candidates, `max_bit_width`, block granularity, selected GPU IDs, output overwrite policy, and router checkpoint requirements.
4. Resolve environment metadata.
5. Write a manifest with status `started`.

The config module owns the rule that QAQ results cannot be accepted when static 8-bit and static 4-bit comparison settings are absent from the same evaluation plan.

#### Algorithm Details

Validation is deterministic:

```text
load config
apply overrides if supported
validate required fields
validate precision candidates are unique, positive, and <= max_bit_width
validate mode-specific artifacts, including router checkpoint for trained QAQ
validate selected GPU IDs if CUDA execution is required
validate output directory policy
write manifest with status started
```

#### Dependencies

No project module dependency. It may use standard configuration parsing and hardware discovery libraries once approved.

#### Failure Handling

- Invalid config fails before model loading.
- Existing output directories fail unless overwrite is explicit.
- Invalid GPU IDs fail unless automatic fallback is explicitly configured.
- Missing router checkpoint fails for trained QAQ evaluation.
- Failure after manifest creation must update status or write an incomplete marker.

#### Independent Test Plan

Map to `tests/unit/test_config_validation.py`, result schema tests, and E2E edge-case validation from `doc/test-plan.md`.

Required cases:

- Invalid mode.
- Missing model or tokenizer.
- Non-positive or duplicate precision candidates.
- Candidate bit-width greater than `max_bit_width`.
- Invalid GPU ID.
- Unsafe output reuse.
- Missing router checkpoint for QAQ evaluation.

#### Open Questions

- What implementation language and config library should be used?
- Should CLI flags, config files, or both be the authoritative input?
- What exact result format should the manifest reference?

### Model and Benchmark Adapter

#### Responsibility

Load the pretrained causal LLM and tokenizer, expose benchmark-compatible tokenization and prompt formatting, provide FP16/reference execution, and provide hidden representations for router features.

#### Non-Responsibility

This module does not decide precision, quantize weights, train router parameters, manage on-demand loading, or aggregate benchmark reports.

#### Inputs and Outputs

Inputs:

- Validated run configuration.
- Model and tokenizer IDs or paths.
- Dataset examples and benchmark settings.
- Device placement settings.

Outputs:

- Model execution handle.
- Tokenized batches.
- Hidden representations at or near controlled blocks.
- Logits, losses, predictions, or labels required by metrics.

#### Public Interface

Must expose these capabilities:

- Load tokenizer and model.
- Freeze base model parameters for router training.
- Tokenize benchmark examples consistently.
- Execute FP16 teacher/reference passes.
- Return hidden states for each controlled block.
- Provide model architecture metadata to the Block Registry.

Exact method names and framework-specific adapter boundaries are open.

#### Data Structures

- Tokenized batch.
- Benchmark example metadata.
- Model architecture metadata.
- Hidden-state bundle keyed by block ID.
- Teacher/student output bundle.

#### Internal Design

The adapter should isolate framework-specific model details from the rest of QAQ. The Block Registry should not inspect arbitrary framework internals without going through metadata or discovery hooks supplied by this adapter.

For benchmark comparability, all modes must share the same tokenizer, prompt formatting, dataset split, batch size, sequence length policy, and metric implementation. The adapter records these settings in the run manifest and result artifacts.

#### Algorithm Details

Feature extraction is source-supported as `h_j(x)`, the hidden representation at block `j`. The exact tensor position is open and must be recorded. Candidate feature points include the block input, block output, or a pooled representation near the block. The detailed design does not choose among them without a router-training decision.

#### Dependencies

- Experiment Configuration and Run Manifest.
- Block Registry and Precision Plan for controlled block metadata.
- Approved model loading, tokenizer, and dataset libraries.

#### Failure Handling

- Missing or inaccessible model/tokenizer paths fail clearly.
- Unsupported model architecture fails before quantization if block discovery cannot proceed.
- Dataset or benchmark access failures mark the benchmark not run rather than fabricating metrics.
- Context-length policy violations reject or truncate inputs according to configured policy.

#### Independent Test Plan

Map to E2E smoke tests, public/sample validation tests, edge-case validation tests, and performance tests from `doc/test-plan.md`.

Unit tests can use fake models and fake tokenizers. GPU and LLaMA-3.1-8B runs are required only for milestone acceptance, not for isolated adapter tests.

#### Open Questions

- Which model loading framework is approved?
- What exact hidden-state feature point feeds the router?
- Are generation workloads in scope for the first milestone, or only benchmark scoring and perplexity?

### Block Registry and Precision Plan

#### Responsibility

Map the model structure into QAQ-controlled blocks, assign stable block IDs, validate block granularity, validate supported precision candidates, and produce static, fixed, or routed precision plans.

#### Non-Responsibility

This module does not quantize tensors, reconstruct weights, compute router probabilities, or execute model inference.

#### Inputs and Outputs

Inputs:

- Model architecture metadata.
- Candidate bit-widths.
- Fixed precision profiles.
- Router decisions.

Outputs:

- Block descriptors.
- Validated precision plans.
- Block-to-artifact mapping requirements.

#### Public Interface

Must expose these capabilities:

- Discover controlled blocks.
- Validate block granularity.
- Produce stable block IDs.
- Validate that every block has a precision decision for the current mode.
- Validate that precision decisions are supported by available bit-plane artifacts.

Exact discovery hooks and class names are open.

#### Data Structures

- Block Descriptor.
- Precision Plan.
- Fixed profile specification.
- Unsupported-layout error.

#### Internal Design

The primary target is MHA/FFN block control. Discovery should preserve model order and produce IDs stable enough to join block metadata, bit-plane artifacts, router traces, loader events, and result summaries.

Precision plans are generated by mode:

- `fp16`: no quantized block decisions required for execution, but block metadata can still be recorded.
- `static_8bit`: every controlled block is assigned 8-bit.
- `static_4bit`: every controlled block is assigned 4-bit.
- `fixed_mixed`: every controlled block is assigned by the configured fixed profile.
- QAQ modes: every controlled block is assigned by Router Policy Module output.

#### Algorithm Details

Discovery contract:

```text
read model architecture metadata
for each transformer layer in model order:
    identify MHA block if supported
    identify FFN block if supported
    create stable block_id from layer index and block type
validate block count and ordering
```

Precision validation contract:

```text
for each block:
    require a precision decision for active runtime mode
    require decision in precision_candidates
    require decision <= max_bit_width
    require bit-plane artifact support when quantized execution is requested
```

#### Dependencies

- Experiment Configuration and Run Manifest.
- Model and Benchmark Adapter.
- Router Policy Module for adaptive decisions.
- Quantization and Bit-Plane Store for artifact availability metadata.

#### Failure Handling

- Unsupported layer layouts fail clearly.
- Missing block decisions fail before inference.
- Unsupported bit-widths fail before weight reconstruction.
- Silent fallback to full precision is forbidden for quantized modes.

#### Independent Test Plan

Map to `tests/unit/test_block_registry.py`, integration static-equivalent profile tests, and regression tests for all-blocks-same precision behavior.

Required cases:

- Fake transformer with MHA and FFN blocks produces stable IDs.
- Unsupported layout fails clearly.
- Repeated discovery preserves ordering.
- Missing or invalid precision decisions fail.

#### Open Questions

- Is whole-layer control acceptable as a temporary simplification if MHA/FFN control is too costly?
- What exact block naming scheme should become stable for artifacts?
- How should batched queries with different precision plans be represented?

### Quantization and Bit-Plane Store

#### Responsibility

Create, load, validate, serialize, and reconstruct quantized bit-plane artifacts for controlled blocks.

#### Non-Responsibility

This module does not choose runtime precision, train the router, load data to GPU on demand, or compute benchmark metrics.

#### Inputs and Outputs

Inputs:

- Full-precision weights.
- Block descriptors.
- Maximum bit-width `B`.
- Requested effective bit-width.
- Artifact paths and metadata.

Outputs:

- Bit-plane artifacts.
- Reconstructed block weights.
- Artifact validation summaries.

#### Public Interface

Must expose these capabilities:

- Quantize supported block tensors to a maximum-bit representation.
- Decompose quantized tensors into bit-planes.
- Load and validate bit-plane artifacts.
- Reconstruct requested effective precision from selected planes.
- Report artifact metadata and validation status.

Exact storage format and API names are open.

#### Data Structures

- Bit-Plane Artifact Metadata.
- Quantized tensor metadata.
- Bit-plane tensor bundle.
- Reconstruction request and response.

#### Internal Design

The paper defines bit-plane decomposition conceptually as a sum over binary planes up to maximum bit-width `B`, with lower precision using selected most significant bit-planes. For signed or scaled real-valued weights, the implementation must combine this with an explicit quantization scheme. That scheme is not specified in the paper and must be recorded in artifact metadata.

The store should support static-equivalent reconstruction for all-8-bit and all-4-bit profiles before adaptive routing is evaluated. This makes bit-plane correctness independently testable.

#### Algorithm Details

Bit-plane creation:

```text
for each controlled block tensor:
    quantize full-precision tensor using approved quantization scheme
    represent quantized integer values at max_bit_width B
    split integer representation into B binary planes
    write bit-plane tensors and metadata
    validate full-width reconstruction against quantized representation
```

Effective precision reconstruction:

```text
validate requested bit_width <= B
select the top bit_width planes according to artifact reconstruction policy
recombine selected planes into quantized integer representation
dequantize or materialize runtime weight representation as required by runtime
return reconstructed weight plus metadata
```

Numeric tolerances for static-equivalent comparison are open and must be confirmed before acceptance tests enforce exact thresholds.

#### Dependencies

- Experiment Configuration and Run Manifest.
- Block Registry and Precision Plan.
- Approved tensor and serialization libraries.

#### Failure Handling

- Missing planes fail clearly.
- Requests for non-positive bit-widths or bit-widths above `B` fail.
- Artifact model/block mismatch fails before reconstruction.
- Checksum or metadata validation failure prevents use in accepted runs.

#### Independent Test Plan

Map to `tests/unit/test_bitplanes.py`, integration artifact roundtrip tests, static-equivalent profile tests, golden bit-plane tensor tests, and property-based bit-plane tests.

Required cases:

- Known tensor decomposes and reconstructs as expected.
- Full-width reconstruction preserves quantized representation.
- 4-bit and 8-bit reconstructions preserve shape.
- Missing or invalid planes fail.
- All-4-bit and all-8-bit QAQ reconstruction paths match static quantized baselines within approved tolerance.

#### Open Questions

- What exact quantization scheme, group size, scale shape, and zero-point behavior should be used?
- What artifact file format should be used?
- What numeric tolerance defines static-equivalent reconstruction success?

### Static and Fixed Mixed-Precision Runtime

#### Responsibility

Execute FP16, static 8-bit, static 4-bit, and fixed mixed-precision profiles without learned routing, producing baseline outputs, latency samples, memory samples, and validation records.

#### Non-Responsibility

This module does not train routers, choose adaptive precision, manage CPU-to-GPU on-demand loading, or decide result acceptance.

#### Inputs and Outputs

Inputs:

- Validated run configuration.
- Model and tokenizer adapter.
- Block descriptors.
- Bit-plane store.
- Static or fixed precision profile.
- Benchmark examples.

Outputs:

- Predictions, logits, losses, or generated evaluation outputs.
- Baseline timing and memory events.
- Static-equivalent validation records.

#### Public Interface

Must expose these capabilities:

- Run FP16 reference inference.
- Run static 8-bit inference.
- Run static 4-bit inference.
- Run fixed mixed-precision inference.
- Emit raw outputs and measurement events for the evaluation reporter.

Exact command and Python API names are open.

#### Data Structures

- Precision Plan.
- Runtime output bundle.
- Latency event.
- Memory event.
- Baseline validation record.

#### Internal Design

Static baselines are comparison anchors. QAQ results are not accepted if static 8-bit or static 4-bit baselines are missing under the same model, tokenizer, dataset, prompt formatting, and metric implementation.

Fixed mixed precision is a diagnostic mode. It verifies that block-level mixed execution works before routing is enabled. It does not replace paper-required QAQ modes.

#### Algorithm Details

Runtime flow:

```text
load model and benchmark batch
construct precision plan for mode
for each controlled block:
    use FP16 weights or reconstruct configured bit-width
execute benchmark pass
record raw outputs, latency, and memory events
write runtime status
```

Latency and memory measurement points must be consistent across modes and recorded in the manifest or result artifact.

#### Dependencies

- Experiment Configuration and Run Manifest.
- Model and Benchmark Adapter.
- Block Registry and Precision Plan.
- Quantization and Bit-Plane Store.
- Evaluation Metrics and Results Reporter.
- Logging and Progress Tracking.

#### Failure Handling

- Missing static artifacts fail static quantized modes.
- OOM fails clearly and marks the run incomplete.
- Mixed profiles with missing block decisions fail.
- Different tokenizer, dataset split, or prompt formatting across modes invalidates comparison.

#### Independent Test Plan

Map to integration static-equivalent profile tests, E2E smoke tests, performance smoke checks, and regression tests for missing static baselines.

Required cases:

- One prompt runs through all baseline and fixed modes in smoke configuration.
- All-8-bit fixed profile matches static 8-bit path within tolerance.
- All-4-bit fixed profile matches static 4-bit path within tolerance.
- Missing static baselines reject QAQ acceptance.

#### Open Questions

- Which runtime framework supports the selected quantized execution path?
- What warm-up and cache-clearing policy should be used for latency and memory measurements?
- What exact tolerance is accepted for static-equivalent outputs?

### Router Policy Module

#### Responsibility

Compute query- and block-dependent precision scores, normalize them over candidate bit-widths, choose deterministic discrete precision decisions, and emit routing traces and summaries.

#### Non-Responsibility

This module does not train itself, quantize weights, execute block inference, or load weights from CPU to GPU.

#### Inputs and Outputs

Inputs:

- Hidden representations `h_j(x)` from the model adapter.
- Candidate bit-widths.
- Router parameters/checkpoint.
- Temperature parameter.
- Block descriptors.

Outputs:

- Per-block raw scores where available.
- Per-block probability distributions.
- Per-block discrete precision decisions.
- Router trace summaries.

#### Public Interface

Must expose these capabilities:

- Load router parameters and metadata.
- Score each configured block for a query.
- Normalize scores into bit-width probabilities.
- Convert probabilities into deterministic precision decisions.
- Emit routing traces.
- Validate checkpoint compatibility with active model, block IDs, and candidate precision set.

Exact model class and function names are open.

#### Data Structures

- Router checkpoint metadata.
- Router Trace.
- Precision Plan.
- Probability distribution record.

#### Internal Design

The paper defines router scoring as a lightweight MLP over block hidden representation `h_j(x)`. The module should keep router parameters separate from base LLM parameters and must support traceability from every decision to checkpoint, block ID, input example, candidate set, and temperature.

Discrete decisions must be deterministic. If probability ties occur, the tie-break rule must be stable and documented.

#### Algorithm Details

Router inference:

```text
for each query:
    for each controlled block:
        read hidden feature h_j(x)
        compute router scores using lightweight MLP
        normalize scores over candidate bit-widths using configured temperature
        select bit-width by deterministic decision policy
        record trace
return precision plan
```

The exact choice between argmax, expected precision rounding, sampling, or another decision policy is open. Accepted evaluation should use a deterministic policy.

#### Dependencies

- Experiment Configuration and Run Manifest.
- Model and Benchmark Adapter.
- Block Registry and Precision Plan.
- Logging and Progress Tracking.

#### Failure Handling

- Checkpoint candidate set mismatch fails.
- Missing block IDs fail.
- Non-finite probabilities fail.
- Probability sums outside tolerance fail.
- Constant global precision across all queries or blocks is flagged and cannot satisfy QAQ acceptance unless explicitly diagnostic.

#### Independent Test Plan

Map to `tests/unit/test_router_policy.py`, router checkpoint contract tests, golden router decision tests, routing benchmark tests, and regression tests for constant precision.

Required cases:

- Probabilities are finite, non-negative, and sum to 1.
- Temperature changes distribution sharpness.
- Tie-breaking is deterministic.
- Outputs exist for every configured block.
- Routing summaries show variation by query or block for accepted QAQ runs.

#### Open Questions

- What exact feature point should provide `h_j(x)`?
- Should the first precision set be `{4, 8}` or include a mid precision such as `{4, 6, 8}`?
- What deterministic decision policy should be used for accepted runs?

### Router Training Pipeline

#### Responsibility

Train router parameters while keeping base LLM parameters frozen, using full-precision teacher and quantized student signals with a knowledge-distillation objective.

#### Non-Responsibility

This module does not train the base model, evaluate final benchmark acceptance, or implement dynamic loading.

#### Inputs and Outputs

Inputs:

- Training or calibration examples.
- Model adapter with frozen teacher/reference path.
- Quantized student path.
- Router configuration.
- Bit-plane artifacts.
- Logging and checkpoint settings.

Outputs:

- Router checkpoint.
- Router checkpoint metadata.
- Training metrics.
- Durable progress logs.
- Incomplete-run marker on failure.

#### Public Interface

Must expose these capabilities:

- Validate training configuration.
- Run teacher/student forward passes.
- Compute the approved router-training objective.
- Update only router parameters.
- Save checkpoints and training metadata.
- Resume or mark incomplete runs if resume support is adopted.

Exact training command, optimizer API, and checkpoint format are open.

#### Data Structures

- Router training config.
- Teacher/student output bundle.
- Loss record.
- Router checkpoint metadata.
- Training progress event.

#### Internal Design

The paper states that router training uses a full-precision teacher, a quantized student, frozen base parameters, and knowledge distillation. It does not specify the exact loss formula, dataset, labels, optimizer, or schedule. Therefore, implementation must select and document a concrete training method before any trained-router result is claimed as QAQ.

The training pipeline should be isolated from evaluation so diagnostic routers can be labeled separately and trained-router checkpoints can be validated before use.

#### Algorithm Details

Training flow:

```text
validate router training config
load teacher/reference path
load quantized student path
freeze base model parameters
for each training batch:
    collect block hidden features
    compute router probabilities
    execute or estimate quantized student behavior according to approved objective
    compute knowledge-distillation loss
    update router parameters only
    log loss, progress, elapsed time, and checkpoint events
save checkpoint and metadata
```

The concrete loss composition is open and must be decided before implementation. Any efficiency regularization is an implementation assumption unless confirmed by a source or user decision.

#### Dependencies

- Experiment Configuration and Run Manifest.
- Model and Benchmark Adapter.
- Quantization and Bit-Plane Store.
- Router Policy Module.
- Logging and Progress Tracking.

#### Failure Handling

- Missing training data fails before launch.
- Missing concrete router-training method fails before long-running training.
- Base model parameters accidentally marked trainable should fail validation where detectable.
- Checkpoint write failures mark the run incomplete.
- Training failure must preserve logs and partial status.

#### Independent Test Plan

Map to router checkpoint contract tests, logging and incomplete-run tests, performance tests, and manual verification.

Required cases:

- Tiny router checkpoint saves and reloads.
- Checkpoint metadata matches model, block IDs, and candidate precision set.
- Training logs include progress, loss, elapsed time, and checkpoint events.
- Failure writes incomplete markers.

#### Open Questions

- What exact knowledge-distillation loss should be used?
- What training or calibration dataset should train the router?
- What optimizer, schedule, and hyperparameters are accepted?
- Is an efficiency regularizer required or deferred?

### Adaptive Inference Runtime

#### Responsibility

Execute query-adaptive QAQ inference by collecting router features, applying router precision decisions, materializing selected block precision, running block-wise mixed-precision inference, and emitting adaptive traces.

#### Non-Responsibility

This module does not train the router, create bit-plane artifacts, own loader residency policy, or aggregate final benchmark reports.

#### Inputs and Outputs

Inputs:

- Tokenized query or benchmark batch.
- Validated router checkpoint.
- Block descriptors.
- Bit-plane store.
- Runtime mode: `qaq_on_demand_off` or `qaq_on_demand_on`.
- Dynamic loader when on-demand mode is enabled.

Outputs:

- Predictions, logits, losses, or benchmark outputs.
- Per-query precision plans.
- Routing traces.
- Runtime latency and memory events.
- Loader request events when on-demand mode is enabled.

#### Public Interface

Must expose these capabilities:

- Run QAQ on-demand off.
- Run QAQ on-demand on.
- Request router decisions for each query.
- Request reconstructed or loaded weights for each controlled block.
- Emit raw outputs and adaptive trace metadata.

Exact execution APIs are open.

#### Data Structures

- Precision Plan.
- Router Trace.
- Runtime output bundle.
- Adaptive trace.
- Loader request.

#### Internal Design

`qaq_on_demand_off` and `qaq_on_demand_on` must use the same routing semantics. The only intended difference is whether selected bit-planes or reconstructed weights are already GPU-resident or loaded from CPU on demand.

Accepted QAQ runs must preserve comparability with static baselines. They must share checkpoint, tokenizer, dataset split, prompt formatting, precision candidates, and metric implementation.

#### Algorithm Details

On-demand off:

```text
tokenize query
collect hidden features for controlled blocks
compute router precision plan
for each controlled block:
    reconstruct or access selected precision from GPU-resident artifacts
execute mixed-precision inference
record output, routing trace, latency, and memory
```

On-demand on:

```text
tokenize query
collect hidden features for controlled blocks
compute router precision plan
for each controlled block:
    request selected bit-planes from Dynamic Loader
    wait for synchronous materialization
execute block using loaded selected precision
record output, routing trace, loader events, latency, and memory
```

Batching behavior for examples with different precision plans is unresolved. The first accepted design may process adaptive decisions per query if batching semantics remain unclear.

#### Dependencies

- Model and Benchmark Adapter.
- Block Registry and Precision Plan.
- Router Policy Module.
- Quantization and Bit-Plane Store.
- Dynamic Loader and Memory Residency Manager.
- Evaluation Metrics and Results Reporter.
- Logging and Progress Tracking.

#### Failure Handling

- Missing or invalid router checkpoint fails unless diagnostic mode is explicit.
- Missing bit-plane artifacts fail.
- Loader failure fails `qaq_on_demand_on`.
- Constant precision behavior is flagged and rejected for accepted QAQ claims.
- OOM fails clearly and marks the run incomplete.

#### Independent Test Plan

Map to E2E smoke tests, output format validation, edge-case validation, reproducibility tests, routing benchmark tests, and performance tests.

Required cases:

- Smoke run completes for QAQ on-demand off and on.
- Routing summaries are present.
- Loader summaries are present for on-demand on.
- Invalid router or artifact metadata fails.
- Repeated small runs with fixed seed produce stable deterministic fields.

#### Open Questions

- How should multiple queries with different precision decisions be batched?
- Which execution strategy applies reconstructed block weights without hidden full-precision fallback?
- What measured region defines end-to-end latency for adaptive inference?

### Dynamic Loader and Memory Residency Manager

#### Responsibility

Keep bit-plane artifacts CPU-resident when on-demand mode is configured, materialize only requested selected planes or precision artifacts onto GPU, release or offload inactive data according to policy, and record loader timing and residency state.

#### Non-Responsibility

This module does not decide precision, train routers, compute benchmark metrics, or own model block discovery.

#### Inputs and Outputs

Inputs:

- Router-selected block and bit-width requests.
- CPU-resident bit-plane artifacts.
- Selected GPU IDs.
- Residency and release policy.

Outputs:

- GPU-resident selected weight subset.
- Loader events.
- Residency map.
- Transfer timing summary.
- Loader warnings or failures.

#### Public Interface

Must expose these capabilities:

- Validate load requests.
- Load selected bit-planes or derived precision artifacts from CPU to GPU.
- Report cache hits, loads, releases, and failures.
- Report current residency state.
- Clear or release resident data according to policy.

Exact cache and memory APIs are open.

#### Data Structures

- Loader Event.
- Residency map.
- Loader request.
- Loader summary.

#### Internal Design

The first design uses synchronous loading because the paper reports latency overhead from sequential CPU-to-GPU transfers and treats overlap/prefetching as future work. The loader should make this overhead measurable rather than hiding it.

The loader must distinguish CPU-resident source artifacts from GPU-resident selected artifacts. In on-demand mode, accepted results must show loader activity or explicitly explain why the run is a constrained simulation.

#### Algorithm Details

Synchronous loading:

```text
validate request block_id and bit_width
resolve required bit-plane subset
if subset already GPU-resident:
    record cache hit
else:
    transfer selected subset from CPU to target GPU
    record transfer timing and bytes if available
update residency map
return GPU-resident selected representation
```

Release policy is open. The first milestone can use a simple explicit release or bounded cache policy if documented.

#### Dependencies

- Quantization and Bit-Plane Store.
- Adaptive Inference Runtime.
- Logging and Progress Tracking.
- Approved CUDA/tensor runtime.

#### Failure Handling

- Missing CPU-resident artifact fails.
- Invalid block or bit-width fails.
- CUDA unavailable fails unless CPU-only simulation is explicit.
- Insufficient GPU memory fails clearly and marks the run incomplete.
- Missing loader summary invalidates `qaq_on_demand_on` result acceptance.

#### Independent Test Plan

Map to loader request validation tests, on-demand loader simulation tests, loader benchmark tests, memory benchmark tests, and regression tests for missing loader summary.

Required cases:

- Simulated small tensors load only requested planes.
- Missing plane fails.
- Transfer events are logged.
- Loader summary exists for on-demand mode.
- On-demand memory and latency are reported separately.

#### Open Questions

- What release/cache policy should be used first?
- Should loader requests transfer bit-planes, reconstructed tensors, or framework-specific precision variants?
- What exact GPU memory measurement points are accepted?

### Evaluation Metrics and Results Reporter

#### Responsibility

Aggregate benchmark scores, perplexity, latency, peak GPU memory, routing summaries, loader summaries, manifests, logs, and completion status into machine-readable artifacts and comparison tables.

#### Non-Responsibility

This module does not execute model internals, train routers, make routing decisions, or load weights.

#### Inputs and Outputs

Inputs:

- Runtime outputs.
- Benchmark labels or loss values.
- Timing and memory events.
- Routing traces.
- Loader events.
- Run manifest.

Outputs:

- Result artifacts.
- Comparison tables.
- Human-readable summaries.
- Validation failures for invalid comparisons.

#### Public Interface

Must expose these capabilities:

- Compute accuracy, benchmark score, or perplexity as configured.
- Aggregate latency and memory metrics.
- Validate result schema.
- Validate cross-mode comparability.
- Emit machine-readable results.
- Emit paper-aligned comparison tables where feasible.

Exact report command and file format are open.

#### Data Structures

- Result Artifact.
- Metric event.
- Comparison group.
- Acceptance validation record.
- Report row.

#### Internal Design

Evaluation must treat scientific validity as a first-class contract. A run that completes but lacks static baselines, loader summaries, routing summaries, or comparable configs must be marked invalid for acceptance.

Per-mode results should be separate artifacts that can be grouped into comparison tables by shared model, tokenizer, dataset, prompt format, precision candidates, seed policy, and metric implementation.

#### Algorithm Details

Comparison validation:

```text
group results by model, tokenizer, dataset, split, prompt format, metric, and precision candidates
require FP16, static_8bit, static_4bit, qaq_on_demand_off, and qaq_on_demand_on for paper-required comparison where hardware permits
require routing summary for QAQ modes
require loader summary for qaq_on_demand_on
verify static baseline settings match QAQ settings
mark comparison accepted, incomplete, diagnostic, or invalid
```

Metric calculations must use benchmark-appropriate implementations. The exact benchmark framework is open.

#### Dependencies

- Static and Fixed Mixed-Precision Runtime.
- Adaptive Inference Runtime.
- Router Policy Module traces.
- Dynamic Loader events.
- Experiment Configuration and Run Manifest.
- Logging and Progress Tracking.

#### Failure Handling

- Missing required result fields fail schema validation.
- Missing static baselines invalidates QAQ acceptance.
- Mixed tokenizer, prompt format, or dataset split invalidates comparison.
- Interrupted runs are marked incomplete and excluded from accepted comparisons.
- Missing loader summary invalidates `qaq_on_demand_on` acceptance.

#### Independent Test Plan

Map to `tests/unit/test_results_schema.py`, output format validation tests, golden result artifact tests, golden report row tests, regression tests, performance tests, and manual verification.

Required cases:

- Result artifact includes all required fields.
- Invalid comparisons are rejected.
- Diagnostic modes are labeled and cannot satisfy QAQ acceptance.
- Paper-table row generation is stable for fixed fake metrics.

#### Open Questions

- What canonical result artifact format should be used?
- Which benchmark framework should compute HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande, WikiText-2, and PTB metrics?
- What exact acceptance thresholds supersede current assumptions?

### Logging and Progress Tracking

#### Responsibility

Provide a console monitor plus durable logs for configuration, quantization, training, inference, evaluation, loader activity, warnings, failures, checkpoint events, and incomplete status.

#### Non-Responsibility

This module does not compute benchmark metrics, choose precision, train models, or manage GPU memory residency.

#### Inputs and Outputs

Inputs:

- Progress events.
- Metric events.
- Checkpoint events.
- Loader events.
- Warnings and errors.
- Run manifest metadata.

Outputs:

- Durable training logs.
- Durable inference/evaluation logs.
- Loader and routing log records where configured.
- Failure and incomplete-run markers.
- Console progress output.

#### Public Interface

Must expose these capabilities:

- Record structured progress events.
- Record warnings and failures.
- Flush durable logs.
- Mark incomplete and complete run states.
- Provide a live console view without corrupting timing-sensitive metrics.

Exact logging library, format, and sink names are open.

#### Data Structures

- Log event.
- Progress counter.
- Console monitor state.
- Failure record.
- Incomplete-run marker.
- Log path manifest entry.

#### Internal Design

Training logs and the console monitor must include step or epoch, loss values, learning rate when applicable, elapsed time, checkpoint events, warnings, and failure status. Inference and evaluation logs and the console monitor must include mode, benchmark progress, processed examples, elapsed time, latency summary, memory summary, routing summary, loader summary where applicable, warnings, and failure status.

Latency-sensitive measurements must separate measured inference windows from incidental logging overhead where practical.

#### Algorithm Details

Logging flow:

```text
on run start: create durable log and record manifest reference
on progress event: update console monitor and append durable record
on warning: append warning with module and run context
on failure: append error, flush, and mark incomplete
on run complete: append final summary and mark complete
```

#### Dependencies

All long-running modules emit events to this module. It should not create reverse dependencies that make core modules depend on logging implementation details.

#### Failure Handling

- If logging fails but core metrics can still be written durably, the run may continue with warning.
- If no durable metrics or status can be written, the run must fail clearly.
- Interrupted runs must leave incomplete markers.

#### Independent Test Plan

Map to logging event formatting tests, integration logging and incomplete-run tests, E2E output validation, manual verification, and performance tests checking logging does not distort latency.

Required cases:

- Training and inference logs include required fields.
- Console monitor shows training and inference progress fields without being the only durable record.
- Controlled failure writes incomplete marker.
- Logs distinguish selected GPUs and per-GPU memory when available.
- Timing summaries distinguish loader overhead where required.

#### Open Questions

- What structured log format should be used?
- What progress display library, if any, is approved?
- Should logs be JSONL, plain text, or both?

## Cross-Module Contracts

- Configuration owns validated run settings. Downstream modules must not reinterpret missing or invalid fields silently.
- Model Adapter owns tokenizer, prompt formatting, benchmark batch construction, and hidden-state extraction.
- Block Registry owns block IDs. Bit-plane artifacts, router checkpoints, loader events, and routing traces must reference those IDs.
- Bit-Plane Store owns artifact compatibility and reconstruction semantics.
- Router Policy owns probability normalization and deterministic decision semantics.
- Adaptive Runtime applies router decisions but does not alter them.
- Dynamic Loader owns on-demand residency and transfer timing.
- Evaluation Reporter owns acceptance validation and comparison grouping.
- Logging records module events but does not change benchmark semantics.

No module should depend on hidden shared state. Every cross-module interaction should pass explicit config, metadata, artifact IDs, or event records.

## End-to-End Workflow

### Preparation and Baseline Validation

1. Validate config and create run manifest.
2. Load model, tokenizer, benchmark adapter, and selected dataset split.
3. Discover MHA/FFN blocks and assign stable block IDs.
4. Create or load bit-plane artifacts.
5. Validate all-8-bit and all-4-bit reconstruction paths.
6. Run FP16, static 8-bit, static 4-bit, and fixed mixed-precision smoke paths.
7. Write baseline results and logs.

### Router Training

1. Validate router-training config and data split.
2. Load teacher/reference path and quantized student path.
3. Freeze base model parameters.
4. Collect hidden features and teacher/student outputs.
5. Train only router parameters with the approved distillation objective.
6. Save router checkpoint, checkpoint metadata, training metrics, and logs.

### QAQ On-Demand Off

1. Load validated router checkpoint.
2. For each query, collect router features.
3. Produce per-block precision plan.
4. Reconstruct or access selected precision from GPU-resident artifacts.
5. Run mixed-precision inference.
6. Record output metrics, routing summaries, latency, memory, and logs.

### QAQ On-Demand On

1. Use the same router semantics and candidate precision set as on-demand off.
2. Keep source bit-plane artifacts CPU-resident according to loader policy.
3. For each query and block, request selected planes or precision artifacts from the loader.
4. Synchronously materialize requested data on GPU.
5. Run inference and record loader transfer timing, memory residency, latency, routing, and output metrics.

### Reporting

1. Group results by comparable model, tokenizer, dataset, split, prompt format, precision set, and metric implementation.
2. Require FP16, static 8-bit, static 4-bit, QAQ on-demand off, and QAQ on-demand on for accepted paper-style comparison where hardware permits.
3. Validate routing and loader summaries.
4. Produce machine-readable results and comparison tables.
5. Mark deviations, diagnostic runs, incomplete runs, and constrained-hardware exceptions clearly.

## Test Strategy Mapping

### Test-Plan Section Coverage

| Test-plan section | Design coverage |
| --- | --- |
| Unit Tests | Independent test plans under Experiment Configuration, Block Registry, Quantization Store, Router Policy, Evaluation Reporter, Logging, and Dynamic Loader. |
| Integration Tests | Cross-module contracts for artifact roundtrip, static-equivalent profiles, router checkpoint compatibility, loader event logging, and incomplete-run handling. |
| End-to-End Tests | End-to-End Workflow plus Quality Gates for smoke, first-milestone LLaMA-3.1-8B, and paper-aligned report paths. |
| Golden Tests | Quantization Store covers golden bit-plane tensors; Router Policy covers golden router decisions; Evaluation Reporter covers golden result artifacts and report rows. |
| Regression Tests | Evaluation Reporter, Router Policy, Adaptive Runtime, and Dynamic Loader reject invalid comparisons, constant precision claims, missing baselines, and missing loader summaries. |
| Property-Based Tests | Quantization Store, Router Policy, Experiment Configuration, and Evaluation Reporter define invariants over precision sets, tensors, probabilities, and result schemas. |
| Performance Tests | Static Runtime, Adaptive Runtime, Dynamic Loader, and Evaluation Reporter own accuracy/perplexity, latency, memory, router overhead, loader overhead, and routing variation checks. |
| Manual Verification | Quality Gates and Evaluation Reporter require review of model, tokenizer, dataset split, prompt format, routing summaries, loader summaries, and paper-claim wording. |
| Required Commands | Quality Gates records the test plan's proposed commands as unconfirmed until implementation tooling is selected. |
| CI / Automation Recommendation | Quality Gates separates fast CPU checks, full local checks, and GPU checks; exact CI tooling remains open. |
| Acceptance Gate | Quality Gates restates first-milestone acceptance requirements and current assumed thresholds. |

### Requirement Coverage

| Test-plan requirement | Design coverage |
| --- | --- |
| Config validation rejects invalid modes, precision candidates, GPU IDs, missing artifacts, and unsafe output reuse. | Experiment Configuration and Run Manifest independent tests. |
| Block Registry maps target model into stable controlled blocks and rejects unsupported layouts. | Block Registry and Precision Plan independent tests. |
| Bit-plane reconstruction has expected shape and plausible lower/higher precision behavior. | Quantization and Bit-Plane Store unit, golden, and property tests. |
| Static-equivalent all-8-bit and all-4-bit profiles match static baselines within approved tolerance. | Quantization Store plus Static and Fixed Runtime integration tests. |
| Router emits valid probability distributions and deterministic decisions for every configured block. | Router Policy Module unit, golden, and property tests. |
| Router checkpoint metadata matches active config. | Router Policy Module and Router Training Pipeline integration tests. |
| Dynamic loader reports missing planes, invalid bit-widths, CUDA availability, and insufficient memory clearly. | Dynamic Loader independent and integration tests. |
| Smoke run executes FP16, static 8-bit, static 4-bit, fixed mixed, QAQ on-demand off, and QAQ on-demand on. | Static Runtime, Adaptive Runtime, Dynamic Loader, and Evaluation Reporter E2E tests. |
| Routing summaries demonstrate variation by query or block. | Router Policy Module, Adaptive Runtime, Evaluation Reporter, and regression tests. |
| Training and inference produce console progress, durable logs, and metrics. | Logging and Progress Tracking plus E2E output validation. |
| Evaluation artifacts include metadata required to reproduce the run. | Result Artifact contract and Evaluation Reporter schema tests. |
| Static baselines are mandatory for QAQ acceptance. | Static Runtime and Evaluation Reporter regression tests. |
| On-demand loader latency and memory trade-off are reported. | Dynamic Loader and Evaluation Reporter performance tests. |
| Interrupted runs are marked incomplete. | Experiment Configuration, Logging, and Evaluation Reporter integration tests. |
| Full paper reproduction is not claimed until paper-aligned models and benchmarks are run or deviations are labeled. | Evaluation Reporter acceptance validation and Manual Verification. |

## Quality Gates

`doc/quality-gates.md` is not present. Until it exists, quality gates are defined by `doc/test-plan.md`, `doc/requirements.md`, and the HLD quality alignment.

First-milestone gate:

- Unit tests pass for config validation, bit-plane reconstruction, block registry, router policy, result schema, and loader request validation.
- Integration tests pass for bit-plane artifact roundtrip, static-equivalent 4-bit and 8-bit profiles, router checkpoint compatibility, loader event logging, and incomplete-run handling.
- Smoke E2E runs complete for `fp16`, `static_8bit`, `static_4bit`, `fixed_mixed`, `qaq_on_demand_off`, and `qaq_on_demand_on`.
- LLaMA-3.1-8B first-milestone run completes for required modes where hardware permits.
- Accepted runs write durable logs, machine-readable metrics, selected GPU IDs, latency, peak GPU memory, routing summaries, loader summaries for on-demand mode, and completion status.
- Training and inference runs expose a console monitor for live progress and also persist the same essential status to durable logs.
- QAQ comparisons use the same checkpoint, tokenizer, dataset split, prompt format, seed policy, precision candidates, and metric implementation as static baselines.
- QAQ accuracy uses the currently assumed tolerance: within 1 percentage point of static 8-bit for classification or within 5 percent relative perplexity for language modeling, unless superseded.
- QAQ on-demand memory target is currently at least 5 percent lower peak GPU memory than comparable non-on-demand mode unless constrained-hardware exceptions are documented.
- Routing varies by query or block in accepted QAQ results.

Proposed future commands from the test plan remain unconfirmed:

```bash
pytest -q
python -m qaq.evaluate --config configs/smoke.yaml --modes fp16 static_8bit static_4bit fixed_mixed qaq_on_demand_off qaq_on_demand_on
python -m qaq.evaluate --config configs/llama31_8b_first_milestone.yaml --modes fp16 static_8bit static_4bit qaq_on_demand_off qaq_on_demand_on
```

## Risks and Mitigations

- Router training is under-specified. Mitigation: isolate router training, require a documented loss before trained QAQ claims, and label diagnostic routers.
- Quantization scheme is under-specified. Mitigation: store quantization metadata explicitly and validate static-equivalent reconstruction before adaptive inference.
- MHA/FFN-level control may be difficult. Mitigation: preserve it as the primary target and record whole-layer control only as an approved fallback.
- On-demand loading may increase latency. Mitigation: report loader timing and end-to-end latency separately where practical.
- Exact paper-scale reproduction may be blocked by model, dataset, license, network, hardware constraints, or the lack of available official code. Mitigation: separate first LLaMA-3.1-8B milestone from full paper reproduction claims.
- Benchmark comparability can be invalidated by prompt or tokenizer drift. Mitigation: Evaluation Reporter validates comparison grouping.
- Logging can distort latency. Mitigation: measure timing-sensitive regions separately from progress logging where practical.
- Batching adaptive precision can be complex. Mitigation: keep batching behavior open and allow per-query adaptive execution for initial correctness if approved.

## Assumptions

- The project remains planning-only; no implementation source tree exists yet.
- The first implementation target is LLaMA-3.1-8B.
- Official QAQ code will not be used as an implementation reference because it is private and unavailable.
- GPU selection is configurable and first milestone does not require all 8 RTX 3090 GPUs.
- MHA/FFN block granularity is the primary target because it matches the paper figure.
- Candidate precision must include 4-bit and 8-bit behavior; adding 6-bit or another mid precision remains open.
- CPU-only and fake-model tests are valid for correctness and schema checks, but not for GPU memory or on-demand loading claims.
- Static baselines are mandatory before QAQ results can be accepted.
- Any router-training efficiency regularizer is an implementation assumption unless confirmed later.
- Exact API names, package layout, config file format, result artifact format, dependency choices, and benchmark commands remain open.

## Open Questions

1. Which implementation language, package manager, and test runner should be used?
2. Which external libraries are approved for model loading, quantization, evaluation, datasets, and GPU memory measurement?
3. What canonical machine-readable result format should be used: JSON, JSONL, CSV, or a combination?
4. What exact quantization scheme, group size, scale shape, zero-point behavior, and artifact format should be used?
5. What exact router-training loss, training dataset, calibration data, optimizer, schedule, and hyperparameters should be used?
6. What exact router feature point should provide `h_j(x)`?
7. Should the first QAQ precision candidates be `{4, 8}` or a low/mid/high set such as `{4, 6, 8}`?
8. What deterministic router decision policy should be used for accepted runs?
9. If MHA/FFN block control is too costly for the first milestone, is whole-layer control acceptable as a temporary simplification?
10. What numeric tolerance should static-equivalent QAQ use against static 4-bit and static 8-bit baselines?
11. What exact accuracy, perplexity, memory, and latency thresholds supersede current assumed tolerances?
12. How should batching work when different queries in one batch select different precision profiles?
13. Are generation workloads in scope for first-milestone E2E tests, or should evaluation focus on benchmark scoring and perplexity?
14. What exact dependency install, lint, format, type-check, test, benchmark, and report commands should become accepted quality gates?
