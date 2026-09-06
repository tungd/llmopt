---
type: Experiment
title: 'Round 10: retire the bespoke W4A16 route and generalize fused linear epilogues over the Epilogue monoid'
description: 'Mid-round checkpoint for the Gemma 4 decode compiler slice on Apple Silicon Metal: Round 9 seam retired and the epilogue mechanism generalized end to end. G4P10-01 (c07ba78) deletes the bespoke W4A16 route (Epilogue.w4a16_algebraic_route, Metal.emit_w4a16_algebraic_kernels, W4A16EpiParams, the lower/runtime Some branches) and restores byte-identical legacy dispatch; G4P10-02 (6f07846) generalizes linear region folding in Kernel_ir.Fused_tactic.Linear_epilogue with chain_of / fold_linear_epilogue over the Epilogue monoid (select admission kept flat to Residual_add); G4P10-03 (43d4b6e) makes Epilogue.msl_buffer_decls the single dynamic buffer-declaration contract for every variant, consolidates the three parametric emitters on it, and carries the full epilogue through Metal_runtime assignment dispatch with a fail-stop Error for any epilogue the legacy families cannot yet serve. Decode-visible fusion (G4P10-04) and the user-run device handoff (G4P10-05) remain open.'
tags: [experiment, compiler, gemma4, epilogue, fusion, kernel-ir, routing, metal, metal-runtime, algebraic-ir]
status: stable
generated: { by: antigravity/2.0, at: '2026-09-08T02:30:00+07:00' }
sources:
  - id: decision-round8
    resource: /.okf/decisions/applicative-ir-coordinate-category-and-epilogue-monoid.md
    title: Applicative IR, Coordinate Category, and Contraction Epilogue Monoid architecture (round 8)
  - id: epilogue
    resource: /lib/epilogue.ml
    title: Epilogue monoid core, generalized msl_buffer_decls, algebraic_kernel_name, mul ~factor, inputs
  - id: kernel-ir
    resource: /lib/kernel_ir.ml
    title: Fused_tactic.Linear_epilogue, chain_of, fold_linear_epilogue region epilogue generalization
  - id: metal-lowering
    resource: /lib/metal.ml
    title: Parametric emitter consolidation onto msl_buffer_decls, region_tactics, lower component assembly
  - id: metal-runtime
    resource: /lib/metal_runtime.ml
    title: Assignment_linear_epilogue full-epilogue carry and fail-stop dispatch
  - id: test-suite
    resource: /test/test.ml
    title: Rewritten section 12 legacy pins, Fold_catalog linear epilogue fold catalog, select pins
  - id: exp-round9
    resource: /.okf/experiments/exp-0154-gemma4-performance-round9-2026-09-07.md
    title: Round 9 record of the bespoke W4A16 seam retired by this round
  - id: loop-round10
    resource: /LOOP-gemma4-performance-round10.md
    title: Gemma 4 performance round-10 loop checklist (checkpoint)
---

# Result

Round 10 replaces the review-rejected Round 9 bespoke seam ("Kernel Zoo Trap": one hardcoded `m = 1` groupwise-4/64 flat-residual-add route, a duplicated `W4A16EpiParams` emitter family, a fixed 5-buffer runtime branch, no decode-visible work) with an integrated epilogue generalization whose mechanism slice is now committed and green on `feat/unified-optimizing-compiler` at HEAD `43d4b6e`:

1. **G4P10-01 (commit `c07ba78`) — bespoke seam retired.** Deleted `Epilogue.w4a16_algebraic_route`, `Epilogue.w4a16_algebraic_base`, `Epilogue.algebraic_residual_add`, `Metal.emit_w4a16_algebraic_kernels`, `W4A16EpiParams`, the routed branch in `Metal.lower`, and the runtime `Some` branch; legacy dispatch restored byte-identical and test.ml section 12 rewritten to pin it. `lib/` now contains no `W4A16EpiParams`, no `llmopt_w4a16_linear_f16_g64_epi`, no `w4a16_algebraic_route`, no `emit_w4a16_algebraic_kernels` (the only remaining mentions are test.ml's negative pins). KEPT intact: `Epilogue.Monoid`, `compose`, `canonicalize`, `msl_buffer_decls`, `algebraic_kernel_name`, `inputs`, `equal`, `map_reads`, `of_expr`/`to_expr`, all legacy constructors, and the three parametric emitters. 7 files, +82/−305.
2. **G4P10-02 (commit `6f07846`) — region folding generalized over the monoid.** `Kernel_ir.Fused_tactic` carries `Linear_epilogue { linear; epilogue; output }` (was `Linear_additive`), with `type chain = Folded of Epilogue.t * int | External of Ir.Value.t | Nonfoldable`, `chain_of`, and the exported ungated algebra core `fold_linear_epilogue` (bare linear → `Some Epilogue.none`, full downstream chain otherwise). Select-time `linear_epilogue` stays admission-gated to flat `Residual_add` regions whose `depth + 1 = List.length region.bindings`, so legacy dispatch is byte-identical until emission + runtime can serve richer epilogues. `Epilogue.mul ~factor` added. The test `Fold_catalog` (top-level module, ~test.ml:115, invoked in the main tail) pins: flat linear+residual folds to `Epilogue.Residual_add`, bare linear folds `Some Epilogue.none`, fanout → `None`. 9 files, +514/−31.
3. **G4P10-03 (commit `43d4b6e`) — one dynamic buffer-declaration contract + fail-stop dispatch.** `Epilogue.msl_buffer_decls ~start_index` generalized from `Expr`-only to the single source of buffer-declaration layout for EVERY variant: typed variants bind role-named `bias`/`up`/`residual` buffers (per-variant binding list `[(value, name)]`) in `Epilogue.inputs` order, `Expr` binds `buf_<Value_id>`, decl dtypes derived (`Float16 → half`, else `float`) — all matching `emit_msl_writeback`'s reads. The three parametric emitters (`emit_parametric_w4a16_linear ~start_index:6`, `emit_parametric_q4_k_linear ~start_index:6`, `emit_simdgroup_matrix_gemm ~start_index:4`) all consume it; the bespoke `epilogue_expr_buffer_decls` helper and per-variant hardcoded buffer matches were deleted. Runtime: the assignment type is `Assignment_linear_epilogue { linear; epilogue; output }`; flat `Residual_add` regions keep the byte-identical legacy path (`dispatch_quant_linear ~epilogue_mode:2` with `find_values [linear.input; linear.weight; residual]`; W4A16 g64/g64_m4 unchanged), and every non-flat epilogue returns `Error (Printf.sprintf "assignment Linear-epilogue %s is not servable by the legacy %s kernel family yet" ...)` via `Ir.Dtype.quant_to_string` — a selected-but-unserved region can no longer degrade silently (the `encode_schedule` `None` branch now means only "no region"). 4 files, +148/−150.

## Status of the round

- **Open (G4P10-04)**: decode-visible service — digest-keyed epilogue kernels for the real Q4_0 `Block_quantized` decode families (`metal_block_codegen` + `Metal.lower` region components), then un-gating `Kernel_ir` selection to fully-folded epilogues, then widening the `constructive_lowering` matcher so silu/mul/add/bias consumer chains form single regions. The real decode bundle changes (and the ADR targets become measurable) only after this slice.
- **Open (G4P10-05)**: user-run bundle rebuild + device benchmark handoff (text only).
- ADR targets remain unmeasured by this mechanism slice: schedule **4,617 → <500 commands**, **2,797 → <150 barriers**, ~762 dispatches incl ~385 standalone elementwise → toward **~140 dispatches/token**, latency toward **~35 ms**, bit-exact 128-token parity with golden tokens `100, 45518, 107, 14947, 135383, 568, 11124, 568, 147505, 236789, ...`.

## Supersedes

exp-0154 (Round 9, commits `46416c3` + `fde9817`) documented the bespoke seam as implemented; that seam is now retired and its symbols no longer exist in `lib/` after `c07ba78`. exp-0154 remains in the register as the historical record of the approach the user review rejected; this round supersedes it with the monoid-general mechanism above.

# Static Validation

`ninja -f ninja.build ocaml-test` passes with all 15 canonical OCaml suite PASS lines green (including `llmopt fused Linear_epilogue fold catalog passed` at the Fold_catalog tail and `llmopt canonical W4A16/KVQ8 tests passed` as the final line); `ninja -f ninja.build test` passes — 79 Python unit tests OK; `git diff --check` is clean; the working tree was committed at `43d4b6e` with no residual changes before this docs slice.

Negative source assertions holding after G4P10-01 (test.ml section 12 pins, all green): decode/prefill lowered programs contain no `kernel void llmopt_w4a16_linear_f16_g64_epi`, no `W4A16EpiParams`, and no `llmopt_w4a16_linear_f16_g64_epi` ABI entry; the legacy `has_bias`/`bias_row_stride` writeback text is present on the byte-identical static path.

Commits on `feat/unified-optimizing-compiler`: `c07ba78` (G4P10-01 retire seam), `6f07846` (G4P10-02 fold generalization), `43d4b6e` (G4P10-03 dynamic buffer contract + fail-stop dispatch).
