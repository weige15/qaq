# ADR 0004: Separate Adaptive Routing from On-Demand Loading

## Status
Accepted

## Context

QAQ has two related but distinct behaviors: adaptive precision selection and optional on-demand CPU-to-GPU loading. The paper reports both `QAQ (on-demand off)` and `QAQ (on-demand on)`. The proposal stages adaptive inference with on-demand disabled before adding the dynamic loader, and the HLD separates Router Policy, Adaptive Inference Runtime, and Dynamic Loader modules.

This separation matters because router correctness and memory-transfer behavior need different tests and metrics. If they are coupled, a latency or memory failure could obscure whether precision routing works.

## Options Considered

- Separate routing, adaptive runtime, and dynamic loader modules. Considered because it mirrors the paper's two QAQ modes and allows adaptive behavior to be validated without CPU-to-GPU transfer overhead. Pros: isolates correctness, supports `qaq_on_demand_off` and `qaq_on_demand_on`, and makes loader overhead measurable. Cons: requires explicit contracts between precision plans, artifacts, and loader requests. Accepted.
- Couple routing and loading into one runtime path. Considered because it could reduce initial code paths. Pros: less module plumbing. Cons: makes on-demand off difficult to test and hides whether failures come from routing or loading. Rejected.
- Support only on-demand off first and omit loader architecture. Considered as a short-term simplification. Pros: validates routing sooner. Cons: misses one of QAQ's three core components and cannot evaluate the memory/latency trade-off. Rejected as the full target, although on-demand off remains an intermediate stage.
- Support only on-demand on. Considered because it is the memory-saving mode. Pros: focuses on the system contribution. Cons: prevents clean comparison against adaptive routing without transfer overhead. Rejected.

## Decision

The architecture will keep Router Policy, Adaptive Inference Runtime, and Dynamic Loader as separate responsibilities. Both QAQ runtime modes must use the same routing semantics and candidate precision set. The only intended difference between `qaq_on_demand_off` and `qaq_on_demand_on` is whether selected bit-planes or reconstructed weights are already GPU-resident or loaded from CPU on demand.

Accepted QAQ results must include routing summaries for both QAQ modes and loader summaries for `qaq_on_demand_on`.

## Consequences

This decision makes the evaluation more defensible: routing behavior can be tested without loader overhead, and loader overhead can be measured separately. It also enables smoke tests and integration tests that simulate loader behavior without full model runs.

The tradeoff is additional interface surface. Precision plans, block IDs, artifact metadata, and loader requests must stay compatible. Bugs in these contracts can cause failures before model execution, so schema and integration tests become mandatory.

## Reversal Plan

Supersede this ADR if a selected runtime framework cannot support separate on-demand off/on paths or if benchmark evidence shows the separation prevents correct execution. Reversal requires a new runtime ADR, updates to Adaptive Runtime, Dynamic Loader, Evaluation Reporter, and test-plan expectations, plus clear documentation of how routing and loading are still independently validated.
