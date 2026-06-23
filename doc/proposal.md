# Proposal: Rebuild QAQ Query-Adaptive Mixed-Precision Quantization

## Objective

Rebuild QAQ from `QAQ.pdf` as a faithful, testable research prototype for query-adaptive mixed-precision LLM inference.

The rebuild should demonstrate the paper's central behavior: for each input query, a lightweight router selects block-level precision, quantized weights are reconstructed from bit-plane representations, and optional on-demand CPU-to-GPU loading reduces GPU memory at the cost of measurable latency overhead.

The proposal target is not production serving. It is a staged implementation plan that can feed into a high-level design and later implementation work.

## Source Inputs

- `QAQ.pdf`: primary source for the QAQ architecture, equations, reported baselines, evaluation models, metrics, and limitations.
- Confirmed by user: official QAQ code is private and not available; `QAQ.pdf` is the only supplementary material available to this rebuild.
- `doc/problem-brief.md`: local problem framing, desired outcomes, constraints, risks, and open questions.
- Missing: `doc/repo-map.md`.
- Missing: `doc/quality-gates.md`.

## Current Project State

The available project source material is paper-first. `doc/problem-brief.md` states that the repository has no implementation beyond the paper and a minimal README. The standard repo map and quality gate documents are not present, so current module boundaries, test commands, and approved dependency choices are not yet defined.

Observed proposal inputs are `QAQ.pdf` and `doc/problem-brief.md`. `doc/proposal.md` did not exist before this proposal.

## Problem Summary

Static quantization uses one precision policy for all inputs and layers. The QAQ paper argues this is suboptimal because queries differ in difficulty and activation sensitivity. Harder queries or sensitive blocks may need higher precision, while simpler queries can tolerate more aggressive quantization.

QAQ proposes a co-designed algorithm and system:

- Bit-plane decomposition stores quantized weights in a representation that can reconstruct different effective bit-widths.
- A trainable router predicts query-dependent precision choices for each transformer block.
- Block-wise mixed-precision inference applies the selected precision profile across transformer layers, with MHA and FFN shown as the relevant block types in the figure.
- An optional dynamic loader transfers only router-selected bit-planes from CPU to GPU, reducing GPU memory while increasing latency due to synchronous transfers.

The rebuild must avoid becoming a generic quantization demo. It needs to preserve the paper's defining claims: query-conditioned routing, block-level mixed precision, teacher/student router training, static quantization baselines, and explicit accuracy-memory-latency measurement.

## Constraints

- The paper is a five-page workshop-style source and leaves several implementation details unspecified.
- The paper reports Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B, but the first local implementation target depends on available hardware, checkpoint access, licenses, and runtime support.
- Evaluation must include static quantization baselines, not only full precision.
- The main reported metrics are task accuracy, language-modeling perplexity, WikiText-2 latency, and peak GPU memory.
- Router training is source-supported as knowledge distillation from a full-precision teacher to a quantized student, but the exact loss composition and training data are not fully specified.
- Dynamic loading is source-supported but expected to introduce latency overhead unless prefetching or overlap is added later.
- External libraries, model choices, dataset access, and acceptable reproduction tolerance remain unresolved and should be decided before detailed design.

## Proposed Approach

Build the system in stages, validating each QAQ claim independently before combining them.

Stage 1 should establish a reproducible baseline harness. It should run one agreed model and benchmark path in full precision, static 8-bit, and static 4-bit modes. This gives a correctness oracle, baseline metrics, and a place to measure accuracy, perplexity, latency, and GPU memory consistently.

Stage 2 should implement bit-plane-backed quantized weights for a narrow target. Start with a single layer or block-level slice to verify that top-bit-plane reconstruction produces expected effective precision. Scale only after the isolated reconstruction test agrees with a conventional static quantized baseline within an agreed tolerance.

Stage 3 should add block-level mixed precision with routing disabled. This mode should execute a fixed precision profile per block, such as all 8-bit, all 4-bit, and hand-authored mixed profiles. It separates mixed-precision execution correctness from router quality.

Stage 4 should train and evaluate the router. The source-supported router consumes block/query-dependent hidden representations, uses a lightweight MLP scoring function, and produces probabilities over candidate bit-widths. Training should use a full-precision teacher and quantized student, with any added efficiency regularization documented as an assumption.

Stage 5 should enable query-adaptive inference with on-demand loading disabled. This reproduces the paper's `QAQ (on-demand off)` mode: adaptive precision decisions are active, but selected weights are already GPU-resident, so accuracy and routing behavior can be evaluated without CPU/GPU transfer effects.

Stage 6 should add the dynamic loader for `QAQ (on-demand on)`. Selected bit-planes should be materialized on GPU only when needed, while unselected or inactive bit-planes remain in CPU memory. This stage should explicitly report the expected trade-off: lower GPU memory and higher end-to-end latency.

Stage 7 should run the full comparison matrix and document deviations from the paper. The minimum comparison set should be full precision, static 8-bit, static 4-bit, QAQ with on-demand off, and QAQ with on-demand on.

## Algorithm Strategy

Baseline method:

- Full-precision inference establishes the teacher/reference behavior.
- Static 8-bit inference establishes the main quality target because the paper reports QAQ matching 8-bit accuracy.
- Static 4-bit inference establishes the aggressive compression baseline.
- Fixed mixed-precision profiles provide an intermediate baseline before enabling the learned router.

Intended optimized method:

- Precompute quantized weights as bit-planes up to a maximum bit-width, source-supported as `B`, with the paper giving 8 as an example maximum.
- For each query and transformer block, compute router scores from hidden representations and map them to candidate bit-width probabilities.
- Convert router decisions into block-level effective precision and reconstruct only the selected top bit-planes for the relevant block.
- Run adaptive inference first with all needed bit-planes GPU-resident, then with on-demand CPU-to-GPU transfer enabled.

Correctness strategy:

- Compare bit-plane reconstruction against static quantized weights for known bit-widths.
- Validate single-block behavior before whole-model execution.
- Check that all-8-bit router decisions match the static 8-bit path within tolerance.
- Check that all-4-bit router decisions match the static 4-bit path within tolerance.
- Compare QAQ logits, task predictions, and perplexity against static 8-bit and full-precision references.

Performance strategy:

- Measure peak GPU memory in each mode under the same model, prompt length, batch size, and evaluation harness.
- Measure end-to-end latency separately from model accuracy.
- Report router overhead separately when practical.
- Report dynamic-loader transfer overhead explicitly because the paper identifies synchronous loading as the reason for latency increases.

Candidate precision strategy:

- Source-supported baselines are 4-bit and 8-bit.
- The figure uses low, mid, and high precision categories, but the exact mid precision is not specified.
- A conservative first rebuild can route between 4-bit and 8-bit, then add a mid precision such as 6-bit only after the target precision set is approved.

## Alternatives Considered

Static-only quantization:

- Simpler and useful as a baseline.
- Rejected as the final target because it misses query-conditioned precision selection, which is the central QAQ contribution.

Multiple full quantized model copies:

- Easier to reason about than bit-plane decomposition.
- Rejected as the main approach because the paper's system explicitly relies on bit-plane decomposition and selective loading rather than maintaining independent model copies as the core mechanism.

Router-free heuristic routing:

- Useful as a diagnostic baseline, for example routing by prompt length or fixed block sensitivity.
- Not sufficient as the final QAQ rebuild because the paper specifies a trainable router based on hidden representations and knowledge distillation.

Production serving system:

- Out of scope for the first rebuild.
- The problem brief explicitly prioritizes a reproducible research prototype over production-grade serving, advanced scheduling, or UI work.

Asynchronous prefetching:

- Potentially important for reducing dynamic-loader latency.
- Deferred because the paper itself reports synchronous transfer overhead and identifies optimization of overlap/prefetching as future work.

## Module Candidates

- Experiment configuration: model, dataset, quantization mode, precision candidates, router checkpoint, seeds, prompt format, and hardware settings.
- Baseline inference harness: full precision, static 8-bit, static 4-bit, and fixed mixed-precision modes.
- Quantization and bit-plane storage: conversion, reconstruction, validation, and serialization of bit-plane weights.
- Block abstraction: mapping transformer layers into QAQ-controlled blocks, initially aligned with MHA and FFN as shown in the paper figure.
- Router training: teacher/student execution, hidden-feature capture, router scoring, distillation objective, and checkpointing.
- Adaptive inference runtime: query-level precision decisions, block-wise execution, and metric capture.
- Dynamic loader: CPU-resident bit-plane storage, GPU materialization, offload policy, and transfer timing.
- Evaluation harness: task accuracy, perplexity, latency, peak GPU memory, and comparison reports.
- Reproducibility utilities: checked-in configs, pinned environment documentation, run manifests, and result summaries.

## Milestones

1. Scope lock and environment decision:
   Confirm target model, allowed dependencies, GPU/CPU memory, datasets, benchmark subsets, precision candidates, and success thresholds.

2. Baseline harness:
   Run full precision, static 8-bit, and static 4-bit for the first agreed model/task pair with reproducible metric capture.

3. Bit-plane proof:
   Validate bit-plane decomposition and reconstruction for one representative block, then for the full model in static-equivalent modes.

4. Fixed mixed-precision inference:
   Support block-level precision profiles without learned routing and compare against static baselines.

5. Router training prototype:
   Train a lightweight router using full-precision teacher and quantized student signals, with all undocumented loss choices recorded.

6. QAQ on-demand off:
   Run query-adaptive block-wise mixed precision with all required bit-planes available on GPU and compare accuracy/latency/memory against static baselines.

7. QAQ on-demand on:
   Add CPU-to-GPU selected bit-plane loading and report memory reduction plus latency overhead.

8. Paper-aligned report:
   Produce a table matching the paper's comparison shape where feasible: full FP16, static 8-bit, static 4-bit, QAQ on-demand off, and QAQ on-demand on.

## Validation Plan

Minimum validation:

- A smoke test runs end-to-end inference for one prompt in full precision, static 8-bit, static 4-bit, fixed mixed precision, and QAQ adaptive mode.
- Static-equivalent QAQ profiles match their corresponding static quantized baselines within an agreed numerical tolerance.
- Router output is query-dependent and block-specific, not a constant global precision choice.
- Each evaluation run records model ID, dataset split, prompt format, batch size, context length, quantization mode, precision set, seed, hardware, latency, and peak GPU memory.

Accuracy validation:

- Classification benchmarks should target performance within 1 percentage point of static 8-bit unless hardware/model substitutions make this inappropriate.
- Language-modeling benchmarks should target perplexity within 5 percent relative of static 8-bit unless a different threshold is approved.
- Full precision, static 8-bit, static 4-bit, QAQ on-demand off, and QAQ on-demand on should be compared on the same benchmark settings.

Memory validation:

- QAQ on-demand on should reduce peak GPU memory relative to the static GPU-resident baseline, or the report should explain why the selected model/runtime prevents the result.
- QAQ on-demand off may match static memory if all relevant bit-planes are GPU-resident, consistent with the paper table.

Latency validation:

- End-to-end latency should be reported for all modes.
- Dynamic-loader latency should be expected to increase when synchronous CPU-to-GPU loading is enabled.
- Router and loader overhead should be measured separately when the runtime makes that practical.

Reproducibility validation:

- Runs should be controlled by checked-in configs.
- Dependency versions, model revisions, dataset revisions, and hardware details should be documented.
- Any departure from the paper's exact model list or benchmark list should be recorded as a reproduction limitation.

## Risks and Tradeoffs

- The paper under-specifies router loss, router features, precision candidates, and several storage/runtime details. The rebuild must document each decision as an implementation assumption.
- Exact reproduction may be blocked by model access, dataset access, hardware limits, or the lack of available official code.
- Bit-plane reconstruction may not integrate cleanly with existing quantized linear implementations, so isolated reconstruction tests should come before full-model work.
- Dynamic loading may reduce memory while harming latency substantially. This is not a failure if measured and reported, because the paper reports the same trade-off.
- A small stand-in model may validate system behavior but not reproduce paper-scale memory and accuracy claims.
- Training the router may become the highest-cost stage. A fixed or heuristic router can be used only as a baseline or interim diagnostic, not as the final QAQ result.
- Choosing an inference framework too early could constrain quantization and loading design. Framework selection should happen in high-level design after confirming approved dependencies and hardware.

## Assumptions

- The first deliverable is a research prototype, not a production inference service.
- QAQ should be rebuilt in stages so that static baselines, bit-plane reconstruction, router behavior, and dynamic loading can each be validated independently.
- The first target model may be smaller than Qwen3-4B if local hardware cannot support the paper models, but a paper-scale validation target should remain documented.
- Candidate precision should start from source-supported 4-bit and 8-bit behavior unless the project owner approves an explicit low/mid/high set.
- Hidden representations at or near each controlled block are the preferred initial router input because the paper defines router scoring from `hj(x)`.
- Any efficiency regularization added to router training must be marked as an implementation choice, because the paper only clearly establishes knowledge distillation at proposal level.

## Open Questions

- Should the rebuild prioritize faithful reproduction first, practical simplified implementation first, or a staged path with both?
- What is the first target model and checkpoint revision?
- What GPU model, GPU memory, CPU RAM, disk capacity, and driver/runtime environment are available?
- Which external libraries are allowed for model loading, quantization, evaluation, and memory measurement?
- What datasets and splits should train the router?
- What exact router loss should be used beyond the paper's knowledge distillation description?
- Should candidate precision be `{4, 8}` for the first rebuild or include a mid precision to match the figure's low/mid/high categories?
- Should block granularity be whole transformer layer, MHA/FFN block, or individual projection/MLP linear module?
- What accuracy, memory, and latency thresholds should define success for the first milestone?
