# Problem Brief

## Problem

The project is to rebuild QAQ, a query-adaptive mixed-precision quantization system for large language model inference, from the available paper and figure. QAQ aims to reduce GPU memory pressure while preserving accuracy close to static 8-bit quantization by adapting precision per query and per transformer block. The current repository has no implementation beyond the paper, so the immediate problem is to define a faithful, testable rebuild target before choosing architecture or writing code.

If this is not defined clearly, the rebuild could become a generic quantization demo that misses the paper's core claims: query-conditioned precision selection, block-level mixed precision, router training with a full-precision teacher, and CPU/GPU memory trade-offs during inference.

## Users / Stakeholders

- Project owner: needs a clear scope for rebuilding QAQ and a way to judge whether the result matches the paper's objectives.
- Researchers and students: need a reproducible prototype that exposes the paper's assumptions, trade-offs, and missing details.
- Future implementers and maintainers: need objective boundaries so implementation work does not overfit to vague paper language.
- Evaluators or reviewers: need measurable comparisons against full precision, static 8-bit, static 4-bit, and QAQ-style adaptive inference.
- Deployment users on constrained hardware: care about reduced GPU memory without unacceptable accuracy or latency degradation.

## Current Pain

- The repository currently contains only `README.md` and `QAQ.pdf`; there is no code, evaluation harness, configuration, or documented runtime path.
- The paper describes the framework at a high level but leaves important implementation details unresolved, including router inputs, router labels/loss, block granularity, bit-plane storage format, and dynamic loader scheduling.
- The reported benefits involve competing objectives: accuracy close to 8-bit baselines, lower GPU memory, and increased latency when on-demand loading is enabled.
- There is no confirmed target hardware, model checkpoint, dataset subset, or acceptable reproduction tolerance.
- Without a benchmark plan, success could be confused with simply running inference instead of validating the accuracy-memory-latency trade-off.

## Desired Outcome

The target outcome is a documented, reproducible QAQ rebuild objective that can guide implementation of a research prototype. The prototype should eventually demonstrate the core behavior described in the paper: a quantized LLM whose precision can vary by query and block, with measurable comparisons against full-precision and static quantization baselines.

The rebuild should make the paper's trade-offs visible: when adaptive/on-demand behavior is active, GPU memory should decrease or be better controlled, accuracy should remain close to static 8-bit, and any latency overhead should be measured rather than hidden.

## Success Metrics

- Correctness: the rebuilt system can run end-to-end inference for at least one agreed model and task using full precision, static quantization, and QAQ-style adaptive quantization modes.
- Accuracy: QAQ-style mode should target performance within 1 percentage point of static 8-bit on classification benchmarks or within 5 percent relative perplexity of static 8-bit on language modeling benchmarks, to be confirmed.
- Memory: QAQ-style on-demand mode should show lower peak GPU memory than a static 8-bit baseline or clearly explain why the selected hardware/model prevents that result.
- Latency: end-to-end latency must be reported for all modes, including any overhead from CPU-to-GPU loading.
- Reproducibility: runs should be controlled by checked-in configs, fixed seeds where applicable, and documented dependency/model versions.
- Evaluation coverage: baseline comparisons should include full precision, static 8-bit, static 4-bit, QAQ without on-demand loading, and QAQ with on-demand loading where feasible.

## Non-Goals

- Building a production-grade serving system.
- Training or fine-tuning a new base LLM from scratch.
- Guaranteeing exact reproduction of every paper number before the missing implementation details are resolved.
- Supporting every model family in the paper during the first milestone.
- Building a UI or dashboard.
- Inventing unrelated quantization methods beyond what is needed to rebuild and evaluate the QAQ objective.
- Optimizing asynchronous prefetching or advanced memory scheduling before the basic adaptive quantization behavior is validated.

## Constraints

- Available local context is limited to `QAQ.pdf`, the provided framework figure, and a minimal `README.md`.
- The paper evaluates Qwen3-4B, Qwen3-8B, and LLaMA-3.1-8B, but the first rebuild may need to start with a smaller or more accessible model depending on hardware.
- The system must compare against static quantization baselines, not only against full precision.
- The key operating dimensions are accuracy, latency, and GPU memory usage.
- Router training appears to require a full-precision teacher and a quantized student, so data availability and training cost may constrain fidelity.
- Dynamic loading requires CPU/GPU transfer support and may behave very differently across hardware, drivers, and inference frameworks.
- Model licenses, checkpoint access, benchmark dataset access, and network availability are not yet confirmed.
- Official QAQ code is private and not available for this rebuild; `QAQ.pdf` is the only supplementary material available to the project.

## Risks

- Misinterpreting the paper's unspecified details could produce a system that is not actually QAQ; mitigate by documenting every assumption and validating against the paper's stated baselines.
- Router training may be under-specified or expensive; mitigate by starting with a minimal, inspectable router objective and marking it as an approximation until confirmed.
- Bit-plane decomposition may not integrate cleanly with existing quantized linear layers; mitigate by validating one layer/block in isolation before scaling.
- On-demand CPU/GPU loading may reduce memory but create large latency stalls; mitigate by measuring latency separately from memory and treating the trade-off as a first-class result.
- Benchmark results may vary with model version, prompt formatting, dataset split, and evaluator implementation; mitigate with pinned configs and reproducible evaluation scripts.
- Hardware limitations may prevent using the exact paper models; mitigate by defining a smaller-model reproduction path and a separate full-scale validation target.
- Over-engineering the first version could delay validation of the central claim; mitigate by keeping the first milestone focused on measurable QAQ behavior.

## Unknowns

- Is the objective a faithful research reproduction of the paper, a practical simplified implementation, or both in stages?
- Which model should be the first target: Qwen3-4B, Qwen3-8B, LLaMA-3.1-8B, or a smaller stand-in model?
- What GPU, CPU RAM, disk, and runtime environment are available for development and evaluation?
- Are external libraries such as Hugging Face Transformers, bitsandbytes, GPTQ/AWQ tooling, or lm-evaluation-harness allowed?
- What data should train the router, and what exact knowledge distillation loss should be used?
- What features should the router consume: query embeddings, hidden states per block, calibration statistics, or another representation?
- What is the required precision set, such as 4/6/8-bit or low/mid/high categories from the figure?
- What block granularity is required: MHA and FFN separately, whole transformer layers, or individual linear projections?
- What accuracy, memory, and latency thresholds are acceptable for declaring the rebuild successful?
