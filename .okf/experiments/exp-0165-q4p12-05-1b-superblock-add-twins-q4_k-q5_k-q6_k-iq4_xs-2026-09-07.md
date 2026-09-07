---
type: Experiment
title: 'Typed contraction groups G4P12-05-1b: superblock family _add twins re-homed onto Metal_block_codegen (Q4_K/Q5_K/Q6_K/IQ4_XS)'
description: 'Second code wedge of G4P12-05 (after 05-1a Q8_0/Q5_0), committed 159a676. The superblock families'' served _add twins come from Metal_block_codegen through the emit_superblock twin templates (emit_superblock_linear_add / emit_superblock_m2_add, subblock schema-value decode over 256-element superblocks, double-rounding store tails, flat ABI) plus per-family geometry/admissions (q4_k_linear_add_kernel, q5_k_linear_add_kernel, q6_k_linear_add_kernel, iq4_xs_linear_add_kernel, iq4_xs_m2_add_kernel), and Metal.lower''s quant-add fold dispatches them before the shim. Probed serving surface: every m<>2 Linear-add shape selects the generic tile for all four families; Q4_K/Q5_K/Q6_K additionally serve the llama m2_x4 four-column tile at m=2 (kept text-patched here - 05-2 ports that llama-only tile); IQ4_XS (whose llama zoo has no m2_x4) serves plain _m2 at m=2 and is fully structural. The recorded Q4_K indexing-bug note does not gate this: the bug lives in the dead G4P11-era emit_parametric_q4_k_linear export (metal.ml:11156, no consumers), not in the current superblock family emitters, and the twin shares its decode with the parametric flat that already serves plain Q4_K decode (llama generic/m2 text is removed from programs; only the patched twins carried llama decode semantics before, so the re-homing removed a latent decode inconsistency). Program byte diffs: plain Linear programs byte-identical for all four families; only the twin bodies changed (llama -> parametric schema-value decode); K-family m=2 programs unchanged (m2_x4 still llama-patched). Pins: five emission-oracle fixtures + structural checks + K-family m2_x4-stays-llama assertions. Both gates green at 159a676.'
tags: [experiment, compiler, gemma4, contraction-group, metal-block-codegen, emission-widening, quant-families, quant-add, twins, superblock]
status: stable
generated: { by: poolside/1.0, at: '2026-09-07T22:50:00+07:00' }
sources:
  - id: loop
    resource: /LOOP-typed-contraction-groups.md
    title: 'G4P12 executor checklist item 5 (05-1 complete; m2_x4 port moved to 05-2)'
  - id: exp-audit
    resource: /.okf/experiments/exp-0163-q4p12-05-seam-audit-remaining-quant-families-2026-09-07.md
    title: G4P12-05 seam audit (per-family llama zoo inventories + sub-slice decomposition)
  - id: exp-05-1a
    resource: /.okf/experiments/exp-0164-q4p12-05-1a-block32-add-twins-q8_0-q5_0-2026-09-07.md
    title: G4P12-05-1a record (block32 twins; parent of this wedge)
  - id: block-codegen
    resource: /lib/metal_block_codegen.ml
    title: 'emit_superblock_linear_add / emit_superblock_m2_add after the block32 twins; superblock_twin_geometry + per-family twin_schema_geometry + the five superblock schema-level admissions'
  - id: metal-ml
    resource: /lib/metal.ml
    title: 'quant-add Result fold dispatches the superblock families'' generic/_m2 bases to Metal_block_codegen before the quant_linear_add_source shim (which now only serves the K m2_x4 m=2 tiles)'
  - id: test-epilogue-monoid
    resource: /test/test_epilogue_monoid.ml
    title: '05-1b emission-oracle pins + structural checks + K-family m2_x4-stays-llama assertions'
  - id: test-ml
    resource: /test/test.ml
    title: 'all-quants graph-level Linear_add (m=2, keeps serving every family through the m2/m2_x4 twin)'
---
# Result

G4P12-05-1b (commit `159a676`) re-homes the superblock families' served `_add` twins onto `Metal_block_codegen` structural emission, completing 05-1 (the four block32/superblock twin families are now structural for their reachable generic/`_m2` surfaces).

1. **Superblock twin templates.** `emit_superblock_linear_add` / `emit_superblock_m2_add` mirror `emit_superblock_linear_pair`'s generic and paired-row bodies (one schema-value call per 32-element subblock over 256-element superblocks) with the twin writeback tails — element-wise double-rounding `half(half(acc) + bias[idx])`, flat ABI, no `has_bias`/`bias_value` store block. The schema value function is in scope because `quant_source` emits it before the twin text.
2. **Schema admissions.** `superblock_twin_geometry` validates the superblock shape (block=256/subblock=32/SIMD=32), quant type and code width; per-family wrappers additionally assert the distinguishing physical fields; the five exports (`q4_k_linear_add_kernel`, `q5_k_linear_add_kernel`, `q6_k_linear_add_kernel`, `iq4_xs_linear_add_kernel`, `iq4_xs_m2_add_kernel`) emit the twins.
3. **Reachable serving surface (probed, m=1/2/4/16/22 × n=32/64/4096).** Every m≠2 shape selects the generic tile for all four families; m=2 selects the llama `m2_x4` four-column tile for Q4_K/Q5_K/Q6_K (kept text-patched here — 05-2 ports that llama-only tile) and plain `_m2` for IQ4_XS (whose llama zoo has no `m2_x4`), so **IQ4_XS's reachable Linear-add surface is now 100% structural** and the K families' m≠2 surface is structural.
4. **Q4_K gate resolved (recorded indexing-bug note).** The recorded bug lives in the dead G4P11-era `emit_parametric_q4_k_linear` export (metal.ml:11156 — no consumers anywhere), not in the current superblock family emitters used here. The new twins share their decode with the parametric flat kernels that already serve plain Q4_K decode (llama generic/m2 text is removed from every program via `quant_source`); before 05-1b the patched `_add` twins were the only llama-semantics Q4_K decode left in a Q4_K program, so the re-homing removed a latent decode inconsistency instead of introducing one.
5. **Pins.** Five emission-oracle fixtures (`test/golden/q{4_k,5_k,6_k}_linear_f16_add.msl`, `iq4_xs_linear_f16_add.msl`, `iq4_xs_linear_f16_m2_add.msl`) + structural checks (uchar weight, `llmopt_schema_q{...}_value(packed, lane, subblock)` decode, double-rounding tails, no llama `block_q{4_K,5_K,6_K}` structs, no `has_bias`/`bias_value`), plus assertions that the K families' m=2 programs still carry the llama `m2_x4` `_add` twin and no parametric `_m2` twin.

## Status of the slice

- **Done.** G4P12-05-1 complete at `159a676` (05-1a block32 + 05-1b superblock); both gates green. Working tree fully committed before this docs slice.
- **Next.** G4P12-05-2: port the K families' `m2_x4` four-column m=2 tile (the only remaining shim-served `_add` surface) parametrically, then the other llama tiles (splitk/splitk_m2, x1_l32/x2_l32/n1_l32/n2_l32, llama_mv/m1_r2_sg4, gated, down_add/dual_swiglu). 05-3 (legacy string deletion) follows behind the `add_block32_kernels`/`emit_dequant_*` pin-audit; 04d-A3 remains deferred.
- Device/bundle benchmarks remain user-run (standing constraint).

## Supersedes

exp-0164 (05-1a) is the parent record; exp-0163 is the seam audit. This is the 05-1b checkpoint.
