# ADR 0005: Use Synchronous On-Demand Loading for the First Rebuild

## Status
Accepted

## Context

`QAQ.pdf` reports that on-demand loading reduces GPU memory but increases latency because precision-adaptive bit-plane loading occurs sequentially and computation stalls while higher-precision planes are fetched from CPU memory. The proposal, HLD, requirements, and detailed design all defer asynchronous prefetching or advanced memory scheduling until basic adaptive quantization behavior is validated.

The first rebuild needs to measure the paper's stated trade-off rather than hide it behind a more complex loader.

## Options Considered

- Synchronous on-demand loading. Considered because it matches the paper's reported limitation and is the simplest behavior to measure and explain. Pros: directly exposes CPU-to-GPU transfer overhead, simplifies correctness tests, and avoids premature scheduler complexity. Cons: expected to increase latency and may understate possible optimized performance. Accepted.
- Asynchronous prefetching or overlap. Considered because it could reduce transfer stalls. Pros: may improve latency if future scheduling is correct. Cons: not required for the first milestone and would make it harder to attribute measured latency and memory behavior. Rejected for the first rebuild.
- Keep all bit-planes GPU-resident. Considered because it improves access latency. Pros: simpler execution. Cons: does not demonstrate on-demand memory reduction. Rejected for `qaq_on_demand_on`, retained as the on-demand off comparison mode.
- Disk-backed loading. Considered as a possible extension of CPU-resident storage. Pros: could reduce CPU memory pressure. Cons: not specified in the local requirements and likely adds larger transfer overhead. Rejected for the first rebuild.

## Decision

The first on-demand loader will use synchronous CPU-to-GPU materialization of selected bit-planes or selected precision artifacts. It must record loader events, transfer timing where available, residency state, and failures. It must not claim asynchronous overlap or hidden prefetching behavior.

Asynchronous prefetching, overlap, and advanced release/cache policies are deferred until after the basic QAQ on-demand mode is validated and benchmarked.

## Consequences

This decision keeps loader behavior aligned with the paper's reported latency overhead and makes the memory/latency trade-off explicit. It reduces first-version complexity and supports deterministic loader tests.

The tradeoff is expected higher latency in `qaq_on_demand_on`. Performance impact is unverified until measured on the chosen model, dataset, GPU selection, and runtime. A synchronous loader may become a bottleneck, but that bottleneck is itself part of the first reproduction target.

## Reversal Plan

Supersede this ADR if synchronous loading prevents completing the first milestone, or after baseline measurements show that asynchronous overlap is necessary and can be evaluated without hiding correctness problems. Reversal requires a new loader ADR, updated loader event schemas to distinguish prefetched and demand-loaded transfers, benchmark changes that report overlap behavior, and regression tests proving memory accounting remains valid.
