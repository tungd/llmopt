---
type: Experiment
title: 'Typed contraction groups G4P12-04c-1: parametric Q4_0 x4 digest emission + admission'
description: 'Commit c179cb7 widens Q4_0 digest admission to the x4 decode tile (m=1, output width a multiple of 4 but not 8). The digest-of-flat machinery in lib/metal_block_codegen.ml is deduplicated into one shared single-row form (q4_0_single_row_epilogue_store / q4_0_single_row_digest_of_flat) used by both the x8 and x4 tiles, and the x4 emitter routes canonicalized non-None epilogues through the exported schema-level q4_0_x4_epilogue_kernel while the None default keeps the flat llmopt_q4_0_linear_f16_x4 text byte-identical. Metal.lower''s digest fold and metal_runtime.ml''s digest dispatch widen to serve both tiles with per-tile grids (x4 (n + 3) / 4 vs x8 (n + 7) / 8); every other (quant x tile) shape stays loud. New §5 x4 structural pins and a generalized full-stack lower proof over [("x8", 4096); ("x4", 4092)]. Both gates green; G4P12-04c-2 (m16) and 04c-3 (m32) remain.'
tags: [experiment, compiler, gemma4, contraction-group, epilogue, metal-block-codegen, digest, emission-widening, kernel-ir]
status: stable
generated: { by: poolside/1.0, at: '2026-09-07T19:10:00+07:00' }
sources:
  - id: loop
    resource: /LOOP-typed-contraction-groups.md
    title: 'Typed Contraction Groups G4P12 executor checklist (slice 4 sub-slice 04c-1)'
  - id: block-codegen
    resource: /lib/metal_block_codegen.ml
    title: 'shared single-row digest machinery q4_0_single_row_epilogue_store / q4_0_single_row_digest_of_flat (:434-506); x8 and x4 wrappers; schema-level q4_0_x4_epilogue_kernel (:2261)'
  - id: block-codegen-mli
    resource: /lib/metal_block_codegen.mli
    title: 'new val q4_0_x4_epilogue_kernel export after q4_0_x8_epilogue_kernel'
  - id: metal-lower
    resource: /lib/metal.ml
    title: 'digest fold (:10847-10891) selects q4_0_x8_epilogue_kernel / q4_0_x4_epilogue_kernel per tactic base over one shared Q4_0 Block_schema; loud error names x8/x4 regions'
  - id: metal-runtime
    resource: /lib/metal_runtime.ml
    title: 'digest dispatch (:3026-3067) computes per-tile digest_columns ((n+7)/8 x8, (n+3)/4 x4) and grids linear_f16_grid over both tiles'
  - id: test-epilogue-monoid
    resource: /test/test_epilogue_monoid.ml
    title: '§5 x4 pins: digest rename, exactly four column-guarded stores over accumulator0..3 with base_col + 3u guard, flat default without epilogue buffers, deterministic emission'
  - id: test-ml
    resource: /test/test.ml
    title: 'full-stack lower digest proof generalized over [("x8", 4096); ("x4", 4092)]; marker "llmopt Q4_0 x4 digest emission and lower admission passed"'
  - id: exp-g4p1204b
    resource: /.okf/experiments/exp-0159-q4_0-x8-digest-emitter-rewire-g4p12-04b-2026-09-07.md
    title: G4P12-04b record; 04c-1 (x4) named as the first tile-widening sub-slice
---
# Result

G4P12-04c-1 (commit `c179cb7`) opens the 04c tile-by-tile admission widening with the Q4_0 x4 decode tile (m=1, output width a multiple of 4 but not 8), extending the parametric digest path proven for x8 in 04a/04b to a second real tile while keeping the flat default and every unservable shape byte-stable or loud.

1. **Shared single-row digest machinery.** `lib/metal_block_codegen.ml` consolidates the digest-of-flat machinery into one shared form used by both tiles: `q4_0_single_row_epilogue_store ~columns epilogue` and `q4_0_single_row_digest_of_flat ~columns ~label ~name epilogue kernel` (:434-506). The stale duplicated x8-only block (previously ~:612-710) is deleted; the x8 wrapper (:503-504) and the new x4 wrapper (:506-507) are thin `~columns:8 ~label:"x8"` / `~columns:4 ~label:"x4"` calls over the same code path, so both tiles emit structurally identical kernel shapes that differ only in the column count and rename suffix.
2. **x4 emitter + schema-level entry.** `emit_q4_0_linear_f16_x4 ?(name = "llmopt_q4_0_linear_f16_x4") ?(epilogue = Epilogue.none) ... ()` keeps the None default byte-identical to the flat `llmopt_q4_0_linear_f16_x4` kernel; a canonicalized non-None epilogue routes into the digest variant. The exported `Metal_block_codegen.q4_0_x4_epilogue_kernel ~name ~epilogue ~target schema` (mli + :2261) validates the Q4_0/4-bit schema and is the lower-level admission point, alongside the 04b-era `q4_0_x8_epilogue_kernel` (one label-order fix landed this slice: all labeled args precede the positional `schema`).
3. **Lower admission widened.** `Metal.lower`'s digest fold over `region_quant_digest_tactics` (lib/metal.ml :10847-10891) builds the Q4_0 `Block_schema` once and selects `q4_0_x8_epilogue_kernel` vs `q4_0_x4_epilogue_kernel` per tactic base (`_x8` / `_x4`), canonicalizing the epilogue before emission and keeping `kernel_name` derived from the tactic epilogue so registered ABI entry names agree with the emitted text. The loud `Error` for every other (quant × tile) combination now names the x8/x4 regions explicitly.
4. **Runtime keys + grids widened.** `metal_runtime.ml`'s digest dispatch (Assignment_linear_epilogue arm, :3026-3067) computes `digest_columns` per tile — `(n+7)/8` for `_x8`, `(n+3)/4` for `_x4` — with the digest grid `linear_f16_grid columns`, folded input buffers via `find_values state (Epilogue.inputs epilogue)`, params `[m; n; k; 0]`, and name `Epilogue.algebraic_kernel_name ~base:base_name epilogue`. Non-Q4_0 or non-x8/x4 shapes still fail loudly with the exact-message rule. Stale comments in that arm now reference both decode tiles.
5. **Pins.** §5 of `test/test_epilogue_monoid.ml` proves x4 structural emission: the digest kernel is renamed to the `_x4` digest name, contains exactly four column-guarded `output[...] = half(...)` stores over `accumulator0..3` with the `base_col + 3u < params.n` tail guard, the flat default contains `kernel void llmopt_q4_0_linear_f16_x4(` and no epilogue buffer declarations, and emission is deterministic. `test.ml`'s full-stack lower digest proof is generalized over `[("x8", 4096); ("x4", 4092)]`: plain kernel registered, no `base^"_add"` kernel/source, digest-keyed kernel present with the folded input at buffer(2), output buffer(3)/params buffer(4), the Linear/Q4_0/Float16 ABI entry, and threadgroup (256,1,1). Marker: `llmopt Q4_0 x4 digest emission and lower admission passed`.

## Status of the slice

- **Done.** x4 admission open through lower and runtime; flat x4 default byte-identical; machinery deduplicated; gates green at `c179cb7` (`Ran 79 tests ... OK`, `git diff --check` clean). Working tree fully committed before this docs slice.
- **Next**: G4P12-04c-2 gives the m16 tile a digest tail (multi-row writeback loop applying the epilogue per element at index `(row_base + r) * params.n + column`, runtime grid `((m+15)/16)*n`, new m=16 full-stack pins); G4P12-04c-3 does m32 (grid `((m+31)/32)*n*2`). The x8 split_k:2 branch stays flat-only — dead code with no caller — and its digest tail keeps the loud `invalid_arg` for split_k≠1. G4P12-04d then turns `_add` into Residual_add epilogue emission inside the parametric emitters and deletes `quant_linear_add_source`; after 04: G4P12-05 remaining quant families, G4P12-06 multi-slot groups, G4P12-07 fan-out scheduling, G4P12-08 reduction phases + round close-out.
- Device/bundle benchmarks remain user-run (standing constraint).

## Supersedes

No earlier experiment (04b `fec173e` recorded in exp-0159). This record is the 04c-1 checkpoint of the G4P12-04 emission-widening program (04a..04d).

# Static Validation

`ninja -f ninja.build ocaml-test` and `ninja -f ninja.build test` both passed at `c179cb7` (79 Python tests OK, marker `llmopt Q4_0 x4 digest emission and lower admission passed`), `git diff --check` clean.
