---
type: Experiment
title: 'Typed contraction groups G4P12-05 seam audit: remaining quant families parametric emission'
description: 'Seam audit + sub-slice decomposition for G4P12-05 (Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/IQ4_XS into the parametric emitters) recorded at HEAD b1e582f, immediately after G4P12-04d-A2 (1651ec8) retired the Q4_0 arms of Metal.quant_linear_add_source. Facts: every non-Q4_0 family already has parametric value-function + generic + _m2 flat kernels (emit_linear_pair for the block32 families Q8_0/Q5_0, emit_superblock_linear_pair for Q4_K/Q5_K/Q6_K/IQ4_XS) and nothing else; each family''s llama legacy zoo (dequant + the real serving shapes splitk/x2/x4/x_l32/llama_mv/m1_r2_sg4/gated/down_add) lives in metal.ml strings (Q8_0 :2574 9 kernels, Q5_0 :2906 6, Q4_K :4530 12, Q5_K :5531 9, Q6_K :6330 7, IQ4_XS :6751 3); quant_source removes each family''s llama generic/m2 from program text while the shim still patches them into _add twins. The served non-Q4_0 _add surface is exactly generic + _m2 (the only llama tiles present); bases the uniform costed catalog names but the llama zoo lacks crash extract_kernel with Invalid_argument today (the same latent crash 04d-A2 turned into a loud Error for Q4_0 x8/x4). Sub-slices: 05-1 re-homes the six families'' served generic/m2 _add twins onto codegen parametric emission (the 04d-A2 analogue; biggest risk-reduction wedge); 05-2 ports each family''s remaining served llama shapes per (quant, tile) with the epilogue tail parameter (byte-identical runtime serving, new pins first; do not inherit the recorded Q4_K parametric-emitter indexing bugs); 05-3 deletes llama zoo entries only when unreferenced (add_block32_kernels metal.ml:4518 + emit_dequant_* :4525-4529 consume the block32/K llama strings whole; emit_dequant_* exercised at test.ml:5421-5447).'
tags: [experiment, compiler, gemma4, contraction-group, kernel-ir, metal-block-codegen, emission-widening, quant-families, seam-audit]
status: stable
generated: { by: poolside/1.0, at: '2026-09-07T21:45:00+07:00' }
sources:
  - id: loop
    resource: /LOOP-typed-contraction-groups.md
    title: 'G4P12 executor checklist item 5 (rewritten with the seam audit + 05-1/05-2/05-3 decomposition)'
  - id: metal-block-codegen
    resource: /lib/metal_block_codegen.ml
    title: 'emit_linear_pair :45, emit_superblock_linear_pair :82, per-family q8_0_linear :323, q5_0_linear :354, q4_k_linear :185, q5_k_linear :218, q6_k_linear :254, iq4_xs_linear :287, linear_kernels dispatch'
  - id: metal-ml
    resource: /lib/metal.ml
    title: 'llama zoo strings q8_0 :2574, q5_0 :2906, q4_0 :3188, q4_k :4530, q5_k :5531, q6_k :6330, iq4_xs :6751; quant_source :6975; quant_linear_add_source (non-Q4_0 shim) :6976; add_block32_kernels :4518; emit_dequant_* :4525-4529; quant-add Result fold :10857-10920'
  - id: test-ml
    resource: /test/test.ml
    title: 'all-quants graph-level Linear_add :1828-1858; emit_dequant_* exercises :5421-5447'
  - id: exp-g4p12-04d-a2
    resource: /.okf/experiments/exp-0162-q4_0-m32-digest-emission-g4p12-04c-3-2026-09-07.md
    title: 'prior 04c-3 record naming G4P12-05 as the next program item after 04'
---
# Result

G4P12-05 is a six-family port with no prior sub-slice plan. Following the campaign's audit-then-split discipline (the same seam audit that split 04d into A1/A2/A3), this record fixes the seams at HEAD `b1e582f` and decomposes 05 into three commit-sized sub-slices. No code changed in this slice.

1. **Parametric state per family (the starting point).** `linear_kernels` already emits, for every non-Q4_0 family, a schema value function + the generic + `_m2` flat kernels only — the block32 families (Q8_0 `q8_0_linear` :323, Q5_0 `q5_0_linear` :354) through `emit_linear_pair` (:45, one `value(block, lane)` call per 32-element block, signed Q8_0 / offset Q5_0 decode), the superblock families (Q4_K :185, Q5_K :218, Q6_K :254, IQ4_XS :287) through `emit_superblock`/`emit_superblock_linear_pair` (:82, `value(block, lane, subblock)` over 256-element superblocks with 32-element subblocks). There are **no** non-Q4_0 parametric m4/m16/m32/x4/x8/splitk emitters. Q4_0's full parametric zoo (04a-04d) is the outlier, not the norm.
2. **Llama legacy zoo per family (what actually serves today).** metal.ml holds one string per family with dequant plus the real serving shapes: Q8_0 (:2574, 9 kernels: generic, m1_r2_sg4, llama_mv, m2, splitk, splitk_m2, dual_swiglu, down_add + dequant), Q5_0 (:2906, 6: generic, m2, gated, llama_mv_add, gated_llama_mv + dequant), Q4_K (:4530, 12: generic, m2, splitk, splitk_m2, m2_x2_l32, m2_n2_l32, m2_x4, gated, gated_m2_x1_l32, down_add, llama_mv_add + dequant), Q5_K (:5531, 9), Q6_K (:6330, 7), IQ4_XS (:6751, 3). `quant_source` (:6975) removes each family's llama generic/m2 from program text (the parametric generic/m2 replace them) but the llama generic/m2 stay in the strings because the `quant_linear_add_source` shim (non-Q4_0 since 04d-A2) patches them into `_add` twins.
3. **The served `_add` surface is exactly generic + `_m2`.** Graph-level and region `Linear_add` on the six non-Q4_0 families are served only for tiles the llama zoo contains — generic and `_m2` — via the shim's llama-text patch (the all-quants test at test.ml:1828-1858 exercises m=2 → `_m2`). Tiles the uniform costed catalog can name but a family's llama zoo lacks (`_m4`/`_m16`/`_m32` for the block32 families; `_x4`/`_x8` everywhere non-Q4_0) crash `extract_kernel` with `Invalid_argument` today — the same latent crash that 04d-A2 turned into a loud digest-route Error for Q4_0 x8/x4. Runtime `_add` selection keys are name-based and need no change for 05.
4. **Sub-slice decomposition.**
   - **05-1 — re-home the six families' served generic/m2 `_add` twins** (the 04d-A2 analogue; the first and highest-value wedge): parameterized twin emitters beside `emit_linear_pair`/`emit_superblock_linear_pair` (one block32 twin-pair function serving Q8_0/Q5_0, one superblock twin-pair serving Q4_K/Q5_K/Q6_K/IQ4_XS — same decode body, twin store tail `half(half(acc) + bias[idx])`, flat ABI, no has_bias block), per-family schema-level `q{...}_linear_add_kernel`/`q{...}_m2_add_kernel` admissions, per-family dispatch arms in `Metal.lower`'s quant-add Result fold (:10857-10920), llama-zoo-missing bases converted from `Invalid_argument` crashes to loud digest-route Errors, shim arms retired as they go dead. Pins: per-family emission-oracle fixtures + structural checks + loud-error pins; the all-quants test keeps serving every family's `_m2` Linear-add.
   - **05-2 — per-family llama tile porting** (only after 05-1): port each family's remaining *served* llama shapes (splitk, m2_x4/x2/x1_l32, llama_mv, m1_r2_sg4, gated, down_add/dual_swiglu) into parametric emitters per (quant, tile) with the epilogue tail parameter, family by family, runtime serving byte-identical (new pins before each text change). Do **not** ship a parametric Q4_K tile that inherits the recorded parametric-emitter indexing bugs (out-of-scope note in the LOOP).
   - **05-3 — legacy string deletion**: delete llama zoo entries per family only when the parametric substitute is shipped and unreferenced. `add_block32_kernels` (metal.ml:4518, concatenates the block32 llama strings whole) and `emit_dequant_*` (:4525-4529, exercised at test.ml:5421-5447) are the pin-audit surfaces — the same constraint 04d-A3 recorded for `q4_0_source`.

## Status of the slice

- **Done.** The audit is recorded in LOOP item 5 (rewritten) and this experiment. HEAD `b1e582f` is unchanged by this docs slice (LOOP-only edit, `git diff --check` clean).
- **Next.** G4P12-05-1 (the 04d-A2 analogue for the six non-Q4_0 families' served `_add` twins) is the actionable executor slice. After 05: G4P12-06 multi-slot groups, 07 fan-out/export scheduling, 08 reduction phases + round close-out. 04d-A3 (dead Q4_0 llama text deletion) remains deferred behind its `add_block32_kernels`/`emit_dequant_q4_0` pin-audit.
- Device/bundle benchmarks remain user-run (standing constraint).

## Supersedes

No earlier experiment covered the non-Q4_0 families (04a-04d were Q4_0-only emission widening; exp-0158..0162). This record is the G4P12-05 seam audit.
