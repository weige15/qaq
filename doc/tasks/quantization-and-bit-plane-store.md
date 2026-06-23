# Quantization and Bit-Plane Store

## Goal

Implement bit-plane artifact creation, validation, loading, and reconstruction so static and adaptive modes can materialize requested effective bit-widths from one maximum-bit representation.

## Inputs

- `doc/proposal.md`: Stage 2 requires isolated bit-plane proof before whole-model adaptive inference.
- `doc/high-level-design.md`: The store maps artifacts to Block Registry IDs and enables static 4-bit, static 8-bit, and adaptive mixed-precision materialization.
- `doc/detailed-design.md`: Defines Bit-Plane Artifact Metadata, reconstruction request behavior, artifact validation, and unresolved quantization choices.
- `doc/test-plan.md`: Requires bit-plane unit tests, artifact roundtrip integration tests, golden bit-plane fixtures, property tests, and static-equivalent profile checks.

## Write Scope

Create or edit proposed paths: `qaq/quantization.py`, `qaq/bitplanes.py`, `qaq/artifacts.py`, `tests/unit/test_bitplanes.py`, `tests/integration/test_quantized_artifact_roundtrip.py`, `tests/golden/`, and `tests/fixtures/bitplanes/`.

## Read Scope

Inspect Block Descriptor and Precision Plan contracts, runtime reconstruction needs, loader request shape, and any approved tensor serialization or quantization library decisions.

## Dependencies

Experiment Configuration and Run Manifest. Block Registry and Precision Plan. External quantization/storage choices must be documented before artifact compatibility is treated as stable.

## Tasks

- [x] Define artifact metadata for model identity, block ID, tensor identity, original shape/dtype, quantization parameters, max bit-width, available planes, reconstruction policy, version, checksum, and validation status.
- [x] Implement small-tensor quantization and bit-plane decomposition for a known integer or quantized representation.
- [x] Implement reconstruction for requested bit-widths, including top-bit-plane selection, shape preservation, metadata validation, and invalid-request errors.
- [x] Implement artifact save/load roundtrip for tiny fixtures with model/block compatibility checks.
- [x] Add golden fixtures for known bit-plane tensors and expected 4-bit and 8-bit reconstructions.
- [x] Add property tests for valid precision sets and full-width reconstruction invariants if property testing is adopted. Property testing was not adopted in this dependency-free pass; unit and golden tests cover the invariants needed for this module checkpoint.

## Tests and Quality Gates

- [x] Run `pytest -q tests/unit/test_bitplanes.py tests/integration/test_quantized_artifact_roundtrip.py` when implemented.
- [x] Verify missing planes, invalid bit-widths, and model/block mismatches fail clearly.
- [x] Verify all-4-bit and all-8-bit reconstruction paths can feed static-equivalent profile tests.

## Done When

- [x] Tiny artifacts can be created, saved, loaded, validated, and reconstructed.
- [x] Known 4-bit and 8-bit reconstructions match golden expectations.
- [x] Bit-plane unit and artifact roundtrip tests pass.
