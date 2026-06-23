# ADR 0002: Use Bit-Plane Artifacts for Quantized Weights

## Status
Accepted

## Context

`QAQ.pdf` defines QAQ around bit-plane decomposition: weights are represented up to a maximum bit-width `B`, and lower effective precision is reconstructed from selected most significant bit-planes. The requirements confirm that bit-plane storage is mandatory and that static 4-bit and 8-bit baselines must be comparable to QAQ modes.

The paper also mentions multiple precision variants in CPU memory, but the equations and local requirements make bit-planes the primary source of truth. The detailed design records that real-valued signed weights need explicit quantization metadata because the paper does not specify scale, zero-point, group size, or artifact format.

## Options Considered

- Bit-plane artifacts as the primary quantized representation. Considered because it directly follows the paper equations and supports reconstructing multiple effective bit-widths from one maximum-bit representation. Pros: matches QAQ's core mechanism, supports dynamic precision, and enables static-equivalent tests. Cons: requires custom artifact metadata and careful validation against static quantized baselines. Accepted.
- Multiple independent quantized model copies. Considered because it is easier to reason about operationally. Pros: simple precision selection and possible reuse of existing quantized model loaders. Cons: conflicts with the paper's bit-plane mechanism and increases storage/memory complexity. Rejected as the core QAQ path.
- Static-only quantized weights. Considered as a baseline. Pros: required for 4-bit and 8-bit comparison. Cons: cannot support query-adaptive precision from one representation. Rejected as the QAQ storage design.
- Framework-native quantization artifacts only. Considered because existing frameworks may provide optimized kernels. Pros: may reduce implementation effort if compatible. Cons: may hide bit-plane structure and make on-demand selected-plane loading impossible to verify. Rejected as the sole artifact contract.

## Decision

The Quantization and Bit-Plane Store will own bit-plane artifacts as the primary quantized weight representation. Artifacts must record model identity, block ID, tensor metadata, maximum bit-width `B`, available bit-plane indices, quantization parameters, reconstruction policy, version, and validation status.

Static 4-bit, static 8-bit, fixed mixed-precision, QAQ on-demand off, and QAQ on-demand on must use this representation or a clearly documented derivative of it. Implementations must not treat independent full quantized model copies as a substitute for the bit-plane store in accepted QAQ results.

## Consequences

The design stays faithful to the paper and makes lower/higher precision reconstruction independently testable. It supports golden tests, property-based tests, and static-equivalent profile tests before adaptive routing is evaluated.

The tradeoff is that artifact compatibility becomes a project-level contract. The implementation must decide and document quantization scheme details not specified by the paper. Performance impact is unverified until benchmarked; custom reconstruction may not integrate cleanly with optimized quantized linear kernels.

## Reversal Plan

Supersede this ADR if implementation or benchmark evidence shows that bit-plane artifacts cannot support correct or measurable QAQ behavior on the chosen runtime. The reversal path is to introduce a new storage ADR, migrate artifact metadata and tests, update the Quantization Store, Dynamic Loader, Static Runtime, Adaptive Runtime, and Evaluation Reporter contracts, and clearly label any resulting system as a deviation from the paper's bit-plane mechanism.
