---
type: Experiment
title: 'Typed contraction groups G4P12-05-2: K-family m2_x4 m=2 tile ported to Metal_block_codegen (decode-equivalence audit, exp-0167)'
description: 'First llama-tile port of G4P12-05-2, committed c40cc10. The Q4_K/Q5_K/Q6_K m2_x4 four-column m=2 tile (the only remaining shim-served _add surface after 05-1) leaves the llama zoo: emit_superblock gains ~m2_x4:true for the K families and appends the flat m2_x4 kernel through the shared emit_superblock_m2_x4_flat template (uchar weight + per-element schema-value decode over 256-element superblocks; same ABI, lane semantics, col>=n guard and (n+3)/4 short-row grid as the llama tile), the on-demand _add twins come from emit_superblock_m2_x4_add through the exported q{4_k,5_k,6_k}_m2_x4_add_kernel admissions (double-rounding element-wise tails, no has_bias/bias_value store), Metal.lower''s quant-add fold dispatches the K m2_x4 bases before the quant_linear_add_source shim (now a defensive fallback only), and the llama m2_x4 bodies are deleted from q{4_k,5_k,6_k}_source. Numerics audit (the 05-2 entry condition): per-element decode identity holds for all three families - the llama m2_x4 pair-wise uchar4/float4-dot decode and the parametric schema value functions index the same bytes/nibbles/scales/mins/high-bits/sign per element (Q4_K code byte (s>>1)*32+i with subblock-parity nibble and 6-bit scale/min table decode; Q5_K same plus qh[i]>>s high bit; Q6_K ql/qh 6-bit packing with scale_index half*8+(i>>4)+within*2), so the port swaps an identical weight per element. Accumulation differs in FP order (llama: per-pair d*s*dot(x,q) - dmin*m*sum(x) over float4; parametric: per-subblock scalar x*(d*s*q - dmin*m) sums), so m=2 outputs may differ in low FP bits from the retired llama tile and are now FP-uniform with the parametric family arithmetic serving m>=3 (option A: decode-consistent scalar switch, adopted - decode identity was the audit''s contract, not bit-identical accumulation). Serving is unchanged: name-based runtime dispatch (kernel_abi m2_x4 tile, grid (n+3)/4 short row). Pins: three emission-oracle fixtures + structural/flat checks + MSL-compile pins of the full K-family m=2 Linear-add program sources; both gates green at c40cc10.'
tags: [experiment, compiler, gemma4, contraction-group, metal-block-codegen, emission-widening, quant-families, quant-add, twins, m2_x4, llama-tile-port, numerics-audit]
status: stable
generated: { by: poolside/1.0, at: '2026-09-08T00:20:00+07:00' }
sources:
  - id: loop
    resource: /LOOP-typed-contraction-groups.md
    title: 'G4P12 executor checklist item 5 (05-2: m2_x4 port first; entry condition = parametric vs llama m2_x4 decode-equivalence audit)'
  - id: exp-0165
    resource: /.okf/experiments/exp-0165-q4p12-05-1b-superblock-add-twins-q4_k-q5_k-q6_k-iq4_xs-2026-09-07.md
    title: '05-1b parent record (m2_x4 stayed llama-patched here; probed serving surface: m=2 -> m2_x4 for every n)'
  - id: exp-0166
    resource: /.okf/experiments/exp-0166-q4p12-04d-a3-05-3-llama-generic-m2-zoo-text-deletion-2026-09-08.md
    title: '05-3 record (llama generic/m2 bodies deleted; live llama tiles incl. m2_x4 kept for 05-2)'
  - id: llama-m2x4-baseline
    resource: /lib/metal.ml
    title: 'llama m2_x4 bodies at the pre-slice commit 104dd2b (q4_k/q5_k/q6_k_source), deleted by this slice - the decode baseline of the audit'
  - id: block-codegen
    resource: /lib/metal_block_codegen.ml
    title: 'emit_superblock_m2_x4_flat (:344) / emit_superblock_m2_x4_add (:398), emit_superblock ~m2_x4 (:450), q{4_k,5_k,6_k}_m2_x4_add_kernel admissions (:3054+); the schema value functions q4_k (:470)/q5_k (:512)/q6_k (:548)'
  - id: metal-ml
    resource: /lib/metal.ml
    title: 'quant-add fold arms for the K m2_x4 bases (:10119-10137) dispatch before the quant_linear_add_source shim (defensive fallback only); llama m2_x4 bodies removed from the K family zoo strings'
  - id: test-epilogue-monoid
    resource: /test/test_epilogue_monoid.ml
    title: 'K-family m2_x4 emission-oracle fixtures + structural/flat checks + MSL-compile pins (section 5)'
  - id: fixtures
    resource: /test/golden
    title: 'q{4_k,5_k,6_k}_linear_f16_m2_x4_add.msl emission oracles'
---
# Result

G4P12-05-2's first tile (the K families' `m2_x4` four-column m=2 tile, the
only remaining shim-served `_add` surface) is parametric at `c40cc10`,
closing the last llama-style serving gap on the `_add` surface.

1. **Parametric `m2_x4` emission.** `emit_superblock` gains `~m2_x4:true`
   for `q4_k`/`q5_k`/`q6_k` (false for `iq4_xs`, whose llama zoo never had
   the tile) and appends the flat kernel through
   `emit_superblock_m2_x4_flat` (uchar weight buffer(1), per-element
   `llmopt_schema_q{4_k,5_k,6_k}_value(packed, lane, subblock)` decode over
   8x32-element subblocks per 256-element superblock, same buffer ABI,
   `column_group = gid >> 5u`/`lane >> 3u` layout, `column >= params.n`
   guard and `(n + 3) / 4` short-row grid as the llama tile, `has_bias`
   store tail). The on-demand `_add` twin is
   `emit_superblock_m2_x4_add` through the exported
   `q{4_k,5_k,6_k}_m2_x4_add_kernel` admissions (element-wise
   double-rounding `half(half(acc) + bias[idx])` per row, no
   `has_bias`/`bias_value` store block), keeping the legacy twin numerics
   contract distinct from the single-rounding digest tail.
2. **Fold dispatch + zoo deletion.** `Metal.lower`'s quant-add fold matches
   the three `m2_x4` bases to the codegen twins before the
   `quant_linear_add_source` shim (re-scoped to a defensive fallback; the
   K families' m2_x4 twins no longer reach it). The llama `m2_x4` bodies
   are deleted from `q{4_k,5_k,6_k}_source`; the family text keeps serving
   m=2 plain decode from the parametric flat and m=2 Linear-add from the
   codegen twin, byte-identical serving names/grids (runtime dispatch
   unchanged).
3. **Numerics audit (entry condition).** The parametric and llama `m2_x4`
   decodes were compared per element against the llama bodies preserved at
   `104dd2b`. Per family the byte/nibble/scale/min/high-bit/sign indexing
   is identical:
   - Q4_K: code byte `(subblock >> 1) * 32 + element_in_subblock` with the
     subblock-parity nibble, and the 6-bit scale/min decode from the 12-byte
     scale-min table at `j = subblock & 3` (both the `j < 4` direct arms and
     the `j >= 4` `table[j+8]` + carry-bit arms match the llama pair-wise
     `low_group`/`high_group` formulas exactly).
   - Q5_K: same layout plus the fifth code bit `(qh[element] >> subblock)
     & 1` - the llama `m2_x4` reads `uchar4(qh + 4*local_lane)` and shifts
     by the group index, which is the same `(qh[i] >> s) & 1` per element.
   - Q6_K: 6-bit packing over `ql`/`qh` (`half_block*64 + (within&1)*32 +
     i` byte, `(within>>1)*4` nibble shift, `half_block*32 + i` high byte,
     `within*2` high shift) and scale index `half_block*8 + (i >> 4) +
     within*2` - element-for-element the llama `m2_x4` loop's indices.
   The **accumulation order differs**: the llama tile processes pairs of
   32-element subblocks per iteration with `d*s*dot(x,q) - dmin*m*sum(x)`
   float4 math; the parametric tile accumulates per-element scalar
   `x * (d*s*q - dmin*m)` products subblock by subblock (same
   4/2/1 shuffle-down butterfly across the column's 8 lanes in both).
   Weights per element are identical; m=2 results may differ from the
   retired llama tile in low FP bits. **Decision (option A, adopted):
   decode-consistent scalar switch** - m=2 now uses the parametric family's
   arithmetic, FP-uniform with the m >= 3 generic that already served
   through the same value functions. Decode identity (not bit-identical
   accumulation) was the LOOP entry-condition contract; serving, ABI and
   grids are unchanged.
4. **Pins.** Three emission-oracle fixtures
   (`test/golden/q{4_k,5_k,6_k}_linear_f16_m2_x4_add.msl`) + structural
   checks on the twin (schema-value decode, per-row double-rounding tails,
   no llama `block_q{4_K,5_K,6_K}` structs, no `has_bias`/`bias_value`, no
   parametric `_m2` twin at m=2) and on the flat tile (schema decode +
   `has_bias` tail preserved), plus MSL-compile pins compiling the full
   K-family m=2 Linear-add program sources (the new parametric text is no
   longer covered by the `emit_dequant_*` zoo compile pins). Both gates
   green at `c40cc10`; `git diff --check` clean.
5. **One defect caught by the pins.** The first draft of the `_add`
   template used the short-name header style (`llmopt_%s_linear_f16_m2_x4_add`)
   while the schema admissions (like every twin export) pass the full kernel
   name - the emitted text name was mangled
   (`llmopt_llmopt_q4_k_linear_f16_m2_x4_add_linear_f16_m2_x4_add`), which
   would have desynced the ABI entry from the kernel text at runtime. Fixed
   to the sibling-twin convention (`kernel void %s(`) before the slice
   landed; the emission-oracle fixtures pin it.

## Status of the slice

- **Done.** G4P12-05-2 m2_x4 tile complete at `c40cc10` (all gates green);
   the `_add` serving surface is now 100% structural across all seven
   quant families (Q4_0 digest-served regions, Q4_0 m4/m16/m32 zoo statics,
   Q8_0/Q5_0/Q4_K/Q5_K/Q6_K/IQ4_XS generic+_m2 codegen twins, K-family
   m2_x4 codegen twins).
- **Next.** G4P12-05-2 later tiles: the K families' `splitk`/`splitk_m2`,
   the `x1_l32`/`x2_l32`/`n1_l32`/`n2_l32` shapes, `llama_mv`/`m1_r2_sg4`,
   `gated`, `down_add`/`dual_swiglu` (each with a numerics-equivalence audit
   like this one before its text leaves the zoo). G4P12-06 (multi-slot
   groups through the stack) and G4P12-07 (fan-out/export scheduling)
   follow.
- Device-level validation of the m2_x4 numerics remains user-run (standing
   constraint); this record covers the per-element decode audit only.

## Supersedes

exp-0166 is the 05-3 record (llama generic/m2 zoo deletion) that preceded
this port; exp-0165 is the 05-1b parent. This is the 05-2 m2_x4 checkpoint.
