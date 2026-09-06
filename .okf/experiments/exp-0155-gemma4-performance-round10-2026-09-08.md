---
type: Experiment
title: 'Round 10: retire the bespoke W4A16 route and generalize fused linear epilogues over the Epilogue monoid'
description: 'Round 10 completed mechanism: the Round 9 bespoke W4A16 seam is retired and fused linear epilogues are generalized end to end over the Epilogue monoid. G4P10-01 (c07ba78) deletes the bespoke route (Epilogue.w4a16_algebraic_route, Metal.emit_w4a16_algebraic_kernels, W4A16EpiParams, the lower/runtime Some branches); G4P10-02 (6f07846) generalizes Kernel_ir.Fused_tactic.Linear_epilogue region folding with chain_of / fold_linear_epilogue; G4P10-03 (43d4b6e) makes Epilogue.msl_buffer_decls the single dynamic buffer-declaration contract and adds fail-stop runtime dispatch; G4P10-04b (998906e + 10ef998) lands digest-keyed Q4_0 x8 epilogue emission + Metal.lower/Metal_runtime coverage; G4P10-04c (a366791) un-gates Kernel_ir selection to every fully-folded epilogue; G4P10-04d (12ac1d7) widens constructive_lowering so silu/mul/add/bias chains form single regions/events. The user-run device handoff (G4P10-05) remains open.'
tags: [experiment, compiler, gemma4, epilogue, fusion, kernel-ir, routing, metal, metal-runtime, algebraic-ir]
status: stable
generated: { by: antigravity/2.0, at: '2026-09-08T03:45:00+07:00' }
sources:
  - id: decision-round8
    resource: /.okf/decisions/applicative-ir-coordinate-category-and-epilogue-monoid.md
    title: Applicative IR, Coordinate Category, and Contraction Epilogue Monoid architecture (round 8)
  - id: epilogue
    resource: /lib/epilogue.ml
    title: Epilogue monoid core, generalized msl_buffer_decls, algebraic_kernel_name, mul ~factor, inputs
  - id: kernel-ir
    resource: /lib/kernel_ir.ml
    title: Fused_tactic.Linear_epilogue, chain_of, fold_linear_epilogue, ungated select admission (04c)
  - id: metal-lowering
    resource: /lib/metal.ml
    title: digest-keyed Q4_0 x8 epilogue emission, region_tactics quant_digest routing, lower component assembly
  - id: metal-runtime
    resource: /lib/metal_runtime.ml
    title: Assignment_linear_epilogue digest dispatch with dynamic Epilogue.inputs binding; legacy _add path kept
  - id: constructive-lowering
    resource: /lib/constructive_lowering.ml
    title: folded_candidates chain absorption (04d) into one Linear_add event per block-quantized chain region
  - id: test-suite
    resource: /test/test.ml
    title: Rewritten section 12 legacy pins, Fold_catalog, digest ABI suite, chain-absorption catalog test
  - id: exp-round9
    resource: /.okf/experiments/exp-0154-gemma4-performance-round9-2026-09-07.md
    title: Round 9 record of the bespoke W4A16 seam retired by this round
  - id: loop-round10
    resource: /LOOP-gemma4-performance-round10.md
    title: Gemma 4 performance round-10 loop checklist (completed mechanism; G4P10-05 handoff open)
---
# Result

Round 10 replaces the review-rejected Round 9 bespoke seam ("Kernel Zoo Trap": one hardcoded `m = 1` groupwise-4/64 flat-residual-add route, a duplicated `W4A16EpiParams` emitter family, a fixed 5-buffer runtime branch, no decode-visible work) with an integrated epilogue generalization that is committed and green on `feat/unified-optimizing-compiler` at HEAD `12ac1d7`:

1. **G4P10-01 (commit `c07ba78`) — bespoke seam retired.** Deleted `Epilogue.w4a16_algebraic_route`, `Epilogue.w4a16_algebraic_base`, `Epilogue.algebraic_residual_add`, `Metal.emit_w4a16_algebraic_kernels`, `W4A16EpiParams`, the routed branch in `Metal.lower`, and the runtime `Some` branch; legacy dispatch restored byte-identical and test.ml section 12 rewritten to pin it. `lib/` contains no `W4A16EpiParams`, no `llmopt_w4a16_linear_f16_g64_epi`, no `w4a16_algebraic_route`, no `emit_w4a16_algebraic_kernels` (the only remaining mentions are test.ml's negative pins). KEPT intact: `Epilogue.Monoid`, `compose`, `canonicalize`, `msl_buffer_decls`, `algebraic_kernel_name`, `inputs`, `equal`, `map_reads`, `of_expr`/`to_expr`, all legacy constructors, and the three parametric emitters.
2. **G4P10-02 (commit `6f07846`) — region folding generalized over the monoid.** `Kernel_ir.Fused_tactic` carries `Linear_epilogue { linear; epilogue; output }` (was `Linear_additive`), with `type chain = Folded of Epilogue.t * int | External of Ir.Value.t | Nonfoldable`, `chain_of`, and the exported ungated algebra core `fold_linear_epilogue` (bare linear → `Some Epilogue.none`, full downstream chain otherwise). Select-time `linear_epilogue` was admission-gated until 04c so legacy dispatch stayed byte-identical. `Epilogue.mul ~factor` added.
3. **G4P10-03 (commit `43d4b6e`) — one dynamic buffer-declaration contract + fail-stop dispatch.** `Epilogue.msl_buffer_decls ~start_index` is the single source of buffer-declaration layout for EVERY variant: typed variants bind role-named `bias`/`up`/`residual` buffers in `Epilogue.inputs` order, `Expr` binds `buf_<Value_id>`, decl dtypes derived (`Float16 → half`, else `float`). The three parametric emitters all consume it. Runtime assignment type is `Assignment_linear_epilogue { linear; epilogue; output }`; non-flat epilogues failed loudly until 04b wired the digest route.
4. **G4P10-04b (slices `998906e` + `10ef998`) — decode-visible digest service.** Provenance REFUTED: no `x8_add`/`x4_add` twins can exist because the static q4_0 source has no plain x8/x4 body and `quant_linear_add_source` only renames STATIC kernels. `Metal.emit_q4_0_x8_epilogue_kernel ~name ~epilogue` derives the real `llmopt_q4_0_linear_f16_x8` kernel text, canonicalizes the epilogue, rebuilds the header to the positional layout `input [[buffer(0)]]`, `weight [[buffer(1)]]`, `Epilogue.msl_buffer_decls ~start_index:2`, `output [[buffer(2+p)]]`, `params [[buffer(3+p)]]`, rewrites the single-store tail into eight column-guarded stores applying `emit_msl_writeback`, and drops the legacy `params.has_bias` block. `Metal.lower` `region_tactics` carries `quant_digest` triples and registers digest `Kernel_abi` entries; `Metal_runtime` resolves a legacy `_add` twin first (byte-identical flat path) and otherwise dispatches the digest kernel with contraction buffers followed by `Epilogue.inputs` in stable order, failing loudly for unservable tactics.
5. **G4P10-04c (commit `a366791`) — selection ungated.** `Fused_tactic.linear_epilogue` admits ANY `Folded (epilogue, depth)` with `depth >= 1` and `depth + 1 = List.length region.bindings` (full-chain coverage). Fold catalog extended: Silu-residual, external-scale, full Silu-mul-residual, and stacked residual-add regions now select carrying the exact folded `Epilogue.t`; Sigmoid/cast/bare/multi-linear/fan-out `None` pins remain as grammar cuts.
6. **G4P10-04d (commit `12ac1d7`) — constructive chains absorbed into one region/event.** `tactic_boundary_values` gained a `Linear_epilogue` arm (`linear.input :: linear.weight :: linear.parameters @ Epilogue.inputs epilogue`); `rms_candidates` generalised to `folded_candidates` admitting single-output `Block_quantized` `Linear_epilogue` regions whose tactic slice covers the whole downstream silu/mul/add/bias chain (Dense/Groupwise_packed stay out of scope); chain events reuse `Lowering.Kind.Linear_add` (no eighth constructor; fallback matchers and coverage keys stay exhaustive); `unsupported_regions` skips regions whose assignments are all claimed so absorbed members never leak to legacy gated fallbacks. `Metal_runtime` gained a shared `digest_epilogue ()` closure serving richer folded epilogues (Silu/Gelu/Silu_mul/Gelu_mul/Bias_residual_add/Silu_mul_residual_add/Gelu_mul_residual_add/Expr) through the digest-keyed Q4_0 x8 kernel with fp16 + same-value-metadata validation; flat `Residual_add` regions keep the byte-identical legacy `_add` + `epilogue_mode:2` path.

## Status of the round

- **Open (G4P10-05)**: user-run bundle rebuild + device benchmark handoff (text only). The ADR targets become measurable only after the user rebuilds the real decode bundle and benchmarks on-device.
- ADR targets remain unmeasured by this mechanism slice: schedule **4,617 → <500 commands**, **2,797 → <150 barriers**, ~762 dispatches incl ~385 standalone elementwise → toward **~140 dispatches/token**, latency toward **~35 ms**, bit-exact 128-token parity with golden tokens `100, 45518, 107, 14947, 135383, 568, 11124, 568, 147505, 236789, ...`.
- Known pre-existing device blockers recorded for the handoff: two parametric-emitter indexing bugs in `emit_parametric_w4a16_linear` (element vs simdgroup space mix) and `emit_parametric_q4_k_linear` (`row = simd_idx / total_elements` wrong axis) must be fixed or quarantined before any device use.

## Supersedes

exp-0154 (Round 9, commits `46416c3` + `fde9817`) documented the bespoke seam as implemented; that seam is now retired and its symbols no longer exist in `lib/` after `c07ba78`. exp-0154 remains in the register as the historical record of the approach the user review rejected; this round supersedes it with the monoid-general mechanism above.

# Static Validation

`ninja -f ninja.build ocaml-test` passes with every canonical OCaml suite PASS line green — the tail now reads `llmopt chain-absorption fold catalog passed` (G4P10-04d test), `llmopt fused Linear_epilogue fold catalog passed` (Fold_catalog), and `llmopt canonical W4A16/KVQ8 tests passed` as the final line. `ninja -f ninja.build test` passes — 79 Python unit tests OK. `git diff --check` is clean; the working tree was committed at `12ac1d7` with no residual changes before this docs slice.

Negative source assertions holding after G4P10-01 (test.ml section 12 pins, all green): decode/prefill lowered programs contain no `kernel void llmopt_w4a16_linear_f16_g64_epi`, no `W4A16EpiParams`, and no `llmopt_w4a16_linear_f16_g64_epi` ABI entry; the legacy `has_bias`/`bias_row_stride` writeback text is present on the byte-identical static path. Legacy static Q4_0/Q4_K/W4A16 kernels and their pins are intact.

Commits on `feat/unified-optimizing-compiler`: `c07ba78` (G4P10-01 retire seam), `6f07846` (G4P10-02 fold generalization), `43d4b6e` (G4P10-03 buffer contract + fail-stop), docs `c7c7215` (interim checkpoint), `998906e` (G4P10-04b slice 1 digest emission), `10ef998` (G4P10-04b slice 2 lower/runtime wiring), docs `f4d28d9`, `a366791` (G4P10-04c selection un-gate), docs `cb65f43`, `12ac1d7` (G4P10-04d chain absorption). Round-end docs update pending in this docs slice.
