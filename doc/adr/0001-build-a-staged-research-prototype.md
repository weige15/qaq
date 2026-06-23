# ADR 0001: Build QAQ as a Staged Research Prototype

## Status
Accepted

## Context

The repository is paper-first and has no implementation, package manifest, test harness, or CI configuration. The project goal is to rebuild QAQ from `QAQ.pdf` without turning it into a generic quantization demo. The paper leaves important implementation details unresolved, including router loss, router data, precision candidates, quantization metadata, and loader scheduling.

The proposal, HLD, detailed design, and test plan all prioritize a reproducible research prototype over production serving. They also require separate validation of static baselines, bit-plane reconstruction, fixed mixed precision, router behavior, adaptive inference, and on-demand loading before any paper-style claim is made.

## Options Considered

- Staged research prototype. Considered because QAQ combines algorithmic and runtime behavior, and each claim needs an independent validation point. Pros: preserves scientific traceability, exposes under-specified paper decisions, and supports small/fake-model tests before GPU runs. Cons: delays full end-to-end results and adds artifact/manifest/logging work. Accepted.
- Monolithic end-to-end rebuild. Considered because it could reach a runnable demo sooner. Pros: fewer initial module boundaries. Cons: makes it hard to isolate whether failures come from quantization, routing, loading, or evaluation; risks producing results that are not defensible. Rejected.
- Production serving system. Considered because QAQ is an inference system. Pros: could eventually serve real workloads. Cons: adds scheduling, API, deployment, and operational concerns outside the first rebuild scope. Rejected.
- Static-only quantization demo. Considered as a simpler baseline. Pros: easier to implement and useful for comparison. Cons: misses query-conditioned routing and on-demand loading, which are central QAQ claims. Rejected as the target system.

## Decision

The QAQ rebuild will be designed and implemented as a staged research prototype. The required stages are baseline validation, bit-plane proof, fixed mixed-precision execution, router training and validation, QAQ on-demand off, QAQ on-demand on, and paper-aligned reporting where feasible.

Implementation work must keep these stages independently testable and must not claim an accepted QAQ result unless the required static baselines, routing evidence, loader evidence, logs, and metrics are present.

## Consequences

This decision improves traceability from `QAQ.pdf` to implementation and test artifacts. It creates natural quality gates for bit-plane reconstruction, router behavior, and memory/latency measurement. It also makes unresolved paper details explicit instead of hiding them in code.

The tradeoff is extra upfront design and validation work. A staged prototype may be slower to reach a polished demo, and early runs may use fake models or small tensors that do not prove paper-scale behavior. Production serving concerns remain out of scope until a later decision supersedes this one.

## Reversal Plan

Supersede this ADR if official QAQ code, a production deployment requirement, or benchmark evidence shows that the staged structure blocks valid reproduction. A replacement ADR must identify the new target architecture, update `doc/proposal.md`, `doc/high-level-design.md`, `doc/detailed-design.md`, and `doc/test-plan.md`, and preserve migration paths for existing manifests, tests, and result artifacts.
