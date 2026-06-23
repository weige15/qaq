# ADR 0003: Use MHA and FFN Blocks as Primary Precision Granularity

## Status
Accepted

## Context

The QAQ figure represents transformer layers as MHA and FFN blocks and describes block-wise quantization inference. The requirements infer that the block abstraction should map to MHA and FFN, while acknowledging that whole-layer granularity may need user confirmation if MHA/FFN control proves too costly. The HLD and detailed design set MHA/FFN as the primary target and reserve whole-layer control as a documented fallback.

This decision matters because block IDs join quantized artifacts, precision plans, router traces, loader events, logs, and result summaries. Changing granularity later would affect every artifact and comparison.

## Options Considered

- MHA and FFN block granularity. Considered because it is the closest match to the paper figure and supports block-wise adaptation without going down to individual projections. Pros: paper-aligned, explicit enough for routing traces, and practical for stable block IDs. Cons: may be harder to integrate with some model frameworks than whole-layer control. Accepted.
- Whole transformer layer granularity. Considered as a simpler fallback. Pros: fewer controlled units and easier model discovery. Cons: less faithful to the MHA/FFN block structure and may hide sensitivity differences between attention and feed-forward blocks. Rejected as the primary design, allowed only as a documented fallback if approved later.
- Individual projection or MLP linear granularity. Considered because it gives finer control. Pros: more detailed precision allocation. Cons: increases routing, artifact, and loader complexity beyond what the paper requires for the first rebuild. Rejected for the first target.
- Global per-query precision. Considered as a simple adaptive policy. Pros: easiest routing and execution. Cons: does not demonstrate block-wise adaptation. Rejected.

## Decision

The Block Registry will map supported transformer models into stable MHA and FFN controlled blocks as the primary precision granularity. Block IDs must be stable and must be used consistently by bit-plane artifacts, precision plans, router traces, loader events, and result artifacts.

Whole-layer control may be used only as an explicitly documented fallback if MHA/FFN control is proven infeasible for the first milestone. A fallback run must not be presented as a fully paper-aligned block-granularity reproduction without that limitation.

## Consequences

This decision keeps implementation aligned with the paper and provides enough structure to verify that routing varies by block. It also creates clear test targets for fake transformer discovery and stable ID generation.

The tradeoff is integration risk with real model implementations, especially if model internals do not expose clean MHA/FFN boundaries. Artifact and router checkpoints become tied to block naming and ordering, so block registry changes require migration or invalidation of old artifacts.

## Reversal Plan

Supersede this ADR if MHA/FFN control blocks the LLaMA-3.1-8B milestone or if official QAQ material defines a different granularity. Reversal requires updating Block Registry, Precision Plan, Bit-Plane Store metadata, router checkpoint metadata, loader event schemas, result schemas, tests, and benchmark limitation text. Old artifacts must be rejected or migrated because block IDs will no longer mean the same thing.
