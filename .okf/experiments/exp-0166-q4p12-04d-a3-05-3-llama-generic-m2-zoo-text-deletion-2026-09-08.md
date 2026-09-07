---
type: Experiment
title: 'Typed contraction groups G4P12-04d-A3 + 05-3: retired llama generic/m2 zoo text deleted from all seven family strings'
description: 'Cleanup slices landed b543932 (04d-A3: Q4_0) and 342f43b (05-3: the six non-Q4_0 families): the llama-style generic and _m2 kernel bodies in q{8_0,5_0,4_0,4_k,5_k,6_k,iq4_xs}_source are dead since 04d-A2/05-1a/05-1b (quant_source strips them from every Linear program; parametric generic/m2 flats + structural _add twins ship instead; the quant-add fold dispatches all seven families'' generic/_m2 bases before the shim). Pin-audit: add_block32_kernels is an mli export with zero in-repo callers; emit_dequant_* surfaces keep their MSL-compile pins; an odd-shape serving probe confirmed the K families serve llama m2_x4 at every m=2 n (even n not divisible by 4, via the col>=n guard) so plain llama m2 is never selected, and IQ4_XS/Q8_0/Q5_0 m2 and generic bases are all structural. 26.7KB removed; program byte-diffs pre/post show only one inert blank line per family (runtime-strip artifact) and zero kernel-text change. Kept the live llama tiles (dequant, llama_mv, splitk/x1/x2/x4_l32, gated, down_add, dual_swiglu) for 05-2. test.ml gains structural MSL assertions for q4_0; both gates green at 342f43b.'
tags: [experiment, compiler, gemma4, contraction-group, quant-families, cleanup, llama-zoo, quant-add]
status: stable
generated: { by: poolside/1.0, at: '2026-09-08T01:10:00+07:00' }
sources:
  - id: loop
    resource: /LOOP-typed-contraction-groups.md
    title: 'G4P12 executor checklist items 4 (04d-A3 complete) and 5 (05-1 complete, 05-3 complete, 05-2 entry condition)'
  - id: exp-0163
    resource: /.okf/experiments/exp-0163-q4p12-05-seam-audit-remaining-quant-families-2026-09-07.md
    title: G4P12-05 seam audit (per-family llama zoo inventories + sub-slice decomposition)
  - id: exp-0164
    resource: /.okf/experiments/exp-0164-q4p12-05-1a-block32-add-twins-q8_0-q5_0-2026-09-07.md
    title: 05-1a parent record
  - id: exp-0165
    resource: /.okf/experiments/exp-0165-q4p12-05-1b-superblock-add-twins-q4_k-q5_k-q6_k-iq4_xs-2026-09-07.md
    title: 05-1b parent record
  - id: metal-ml
    resource: /lib/metal.ml
    title: 'q{8_0,5_0,4_0,4_k,5_k,6_k,iq4_xs}_source after deletion (6/4/6/10/7/5/1 kernels), quant_source, add_block32_kernels, emit_dequant_*'
  - id: test-ml
    resource: /test/test.ml
    title: 'structural MSL assertions after the q4_0 emit_dequant compile pin; the six emit_dequant_* compile pins'
---
# Result

Two cleanup slices landed, completing both deferred text-deletion items in the G4P12-04d/05 area:

1. **`b543932` (04d-A3).** The llama `llmopt_q4_0_linear_f16` and `_m2` bodies are deleted from `q4_0_source` (66 lines). Audit before deletion: nothing extracts them (quant_source strips them from Q4_0 programs; the quant-add fold retires the Q4_0 generic/m2 bases structurally; the m4/m16/m32 `_add` twins come from the parametric `q4_0_linear` family emission in metal_block_codegen, not the llama string). Whole-string consumers re-verified: `add_block32_kernels` (mli export, zero in-repo callers) and `emit_dequant_q4_0` (test.ml:5447) now emit the six live kernels only. test.ml gains structural assertions next to the MSL-compile pin.
2. **`342f43b` (05-3).** The same deletion across the six non-Q4_0 family strings: `q{8_0,5_0,4_k,5_k,6_k,iq4_xs}_source` lose their llama generic + `_m2` bodies (591 lines / 26.7KB). Serving probe first (m=2/3 × n=61/62/63): the K families serve llama `m2_x4` at *every* m=2 n (its `col >= params.n` guard handles non-multiples of 4), so plain llama `m2` is never selected; Q8_0/Q5_0/IQ4_XS generic/m2 bases are structural. Kept the live llama tiles for 05-2: dequant, `llama_mv`, `splitk`/`splitk_m2`, the `x1_l32`/`x2_l32`/`n1_l32`/`n2_l32` shapes, `gated` variants, `down_add`, `dual_swiglu`.

**Proof of no serving change**: superblock program captures before/after the deletion diff in exactly one inert blank line per family (the runtime `remove_kernel` strip already removed the bodies from program text; deleting them from the source strings removes the strip's blank-line residue) — zero kernel-text difference. All MSL compile pins (six `emit_dequant_*` surfaces) and both gates green.

## Status of the slices

- **Done.** G4P12-04d complete (A1 `fc3fa2e`, A2 `1651ec8`, A3 `b543932`). G4P12-05-1 complete (`159a676`), 05-3 complete (`342f43b`).
- **Next.** G4P12-05-2: port the K families' `m2_x4` four-column m=2 tile parametrically (the only remaining shim-served `_add` surface), with the numerics-equivalence audit between the llama `m2_x4` decode and the parametric superblock value decode as the entry condition (llama m2_x4 serves m=2 plain decode today while m≥3 serves the parametric generic). Then the remaining llama tiles per family. 06/07/08 remain on the horizon.
- Device/bundle benchmarks remain user-run (standing constraint).

## Supersedes

exp-0163's 05-3 sub-slice and the LOOP's 04d-A3 pending item (both complete now).
