---
type: Experiment
title: 'Typed contraction groups G4P12-04b: retire the Q4_0 x8 digest text splice'
description: 'Commit fec173e rewires Metal.lower''s region_quant_digest fold and the test_epilogue_monoid digest helpers onto the schema-level structural emitter Metal_block_codegen.q4_0_x8_epilogue_kernel (proven byte-identical in 04a against frozen goldens) and deletes the legacy text-splice machinery — Metal.q4_0_x8_digest_store, q4_0_x8_digest_rewrite, emit_q4_0_x8_epilogue_kernel and its mli export. Metal.lower canonicalizes the tactic epilogue before emission to mirror the retired splice (which canonicalized internally) and keeps kernel_name derived from the tactic epilogue so registered ABI entry names still match the emitted kernel text. The test/golden/*.msl fixtures captured before retirement remain the byte-identity oracle. Admission stays Q4_0-x8-only; 04c widens it tile by tile.'
tags: [experiment, compiler, gemma4, contraction-group, epilogue, metal-block-codegen, digest, emission-widening, kernel-ir]
status: stable
generated: { by: poolside/1.0, at: '2026-09-07T11:45:00+07:00' }
sources:
  - id: loop
    resource: /LOOP-typed-contraction-groups.md
    title: 'Typed Contraction Groups G4P12 executor checklist (slice 4 sub-slice 04b)'
  - id: metal-lower
    resource: /lib/metal.ml
    title: 'lower fold over region_quant_digest_tactics now calls Metal_block_codegen.q4_0_x8_epilogue_kernel; q4_0 digest schema built once via Block_schema.of_storage'
  - id: metal-splice
    resource: /lib/metal.ml
    title: 'deleted q4_0_x8_digest_store / q4_0_x8_digest_rewrite / emit_q4_0_x8_epilogue_kernel (was :7034-7154)'
  - id: metal-mli
    resource: /lib/metal.mli
    title: 'deleted val emit_q4_0_x8_epilogue_kernel export'
  - id: test-epilogue-monoid
    resource: /test/test_epilogue_monoid.ml
    title: §5 emit_digest is now the structural emitter; golden fixtures are the sole byte-identity oracle
  - id: golden-dir
    resource: /test/golden/
    title: q4_0_x8_digest_gemma_chain.msl, q4_0_x8_digest_f32_expr.msl, q4_0_x8_digest_silu.msl (frozen in 04a)
  - id: exp-g4p1204a
    resource: /.okf/experiments/exp-0158-q4_0-x8-epilogue-tail-g4p12-04a-2026-09-07.md
    title: G4P12-04a record; 04b listed as the next sub-slice
---
# Result

G4P12-04b (commit `fec173e`) retires the legacy Q4_0 x8 digest text splice now that 04a proved the structural emitter byte-identical against it. `lib/metal.ml` no longer contains any of `q4_0_x8_digest_store`, `q4_0_x8_digest_rewrite`, or `emit_q4_0_x8_epilogue_kernel`, and the mli export is gone; grep across `lib/`, `test/`, `bin/`, `bench/` finds zero remaining references to any of those symbols.

1. **`Metal.lower` digest source repointed.** The fold over `region_quant_digest_tactics` (Q4_0 × x8-tile × epilogue triples) now emits through the schema-level structural emitter `Metal_block_codegen.q4_0_x8_epilogue_kernel ~name:kernel_name ~epilogue:(Epilogue.canonicalize epilogue) ~target q4_0_digest_schema`. The Q4_0 `Block_schema` is built once from `Block_schema.of_storage (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0)` before the fold. Two details preserve byte-identity with the pre-04b output: (a) the epilogue is canonicalized before emission because the structural emitter — like the other parametric emitters — expects an already-canonicalized epilogue, whereas the retired splice canonicalized internally; (b) `kernel_name` is still derived from the (unmodified) tactic epilogue via `Epilogue.algebraic_kernel_name`, so the ABI entry registered just below the fold (which also names from the tactic epilogue) continues to agree with the emitted kernel text. The loud `Error` for every other (quant × tile) combination is unchanged.
2. **Test call sites repointed.** In §5 of `test/test_epilogue_monoid.ml` the digest helpers collapse to one `emit_digest` = the structural emitter (canonicalizing first); the legacy-comparison leg of the byte-parity block is deleted and the `test/golden/*.msl` fixtures become the sole oracle, now guarded by a pin whose text records the splice retirement.
3. **Full-stack oracle untouched.** `test.ml`'s digest proofs, which route through `Metal.lower`, are unchanged and stay green — they assert the digest kernel binds its folded epilogue input buffers at buffer(2..), output/params after them, and the ABI entry properties. The first ocaml-test run after the repoint caught exactly the canonicalization requirement (buffer-2 binding broke before the fix) and is green again after it.

New test line: `[PASS] Structural Q4_0 x8 digest emission is byte-identical to the golden kernels (legacy splice retired)`, followed by the pre-existing §5 decode-base pins, unchanged.

## Status of the slice

- **Done.** Structural emitter is now the only digest emitter; splice + mli export deleted; admission unchanged (Q4_0 x8 only); gates green at `fec173e` (`Ran 79 tests ... OK`, `git diff --check` clean). Working tree fully committed before this docs slice.
- **Next**: G4P12-04c widens admission tile by tile — epilogue tails on `_x4`/`_m16`/`_m32` (and a decision on the reachable-but-unused x8 split_k:2 branch), then `Metal.lower` digest admission and `metal_runtime.ml` digest dispatch keys widen per tile with loud errors for every still-unservable combination. G4P12-04d then turns `_add` into Residual_add epilogue emission and deletes `quant_linear_add_source`. After 04: G4P12-05 remaining quant families, G4P12-06 multi-slot groups, G4P12-07 fan-out scheduling, G4P12-08 reduction phases + round close-out.
- Device/bundle benchmarks remain user-run (standing constraint).

## Supersedes

No earlier experiment (04a `b5ee732` recorded in exp-0158). This record is the 04b checkpoint of the G4P12-04 emission-widening program (04a..04d).

# Static Validation

`ninja -f ninja.build ocaml-test` and `ninja -f ninja.build test` both passed at `fec173e` (79 Python tests OK), `git diff --check` clean.
