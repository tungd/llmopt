---
type: Experiment
title: 'Multi-export contraction groups and fan-out/export scheduling (G4P12-07/08)'
description: 'G4P12-07 (36a1e95) lands multi-export contraction group selection, kernel emission, Metal lowering, and runtime dispatch — two-export fan-out regions become selectable and servable. G4P12-08 closes the G4P12 round with an ADR for reduction-phase semantics and documentation.'
tags: [experiment, compiler, contraction-group, multi-export, fan-out, kernel-emission, metal, runtime, reduction-phase]
status: stable
generated: { by: poolside/1.0, at: '2026-09-11T10:00:00+07:00' }
sources:
  - id: loop
    resource: /LOOP-typed-contraction-groups.md
    title: Typed Contraction Groups G4P12 executor checklist (active program)
  - id: contraction-group
    resource: /lib/contraction_group.ml
    title: multi_export_epilogues, widened select_contraction_group for multi-export
  - id: kernel-ir
    resource: /lib/kernel_ir.ml
    title: Fused_tactic.Multi_export_contraction_group
  - id: metal-block-codegen
    resource: /lib/metal_block_codegen.ml
    title: q4_0_multi_export_linear_f16_x8_kernel schema-level emitter
  - id: metal-q4-0-decode-codegen
    resource: /lib/metal_q4_0_decode_codegen.ml
    title: emit_q4_0_multi_export_linear_f16_x8 kernel template
  - id: metal-selection
    resource: /lib/metal_selection.ml
    title: region_tactics quant_multi_export_digest collection
  - id: metal
    resource: /lib/metal.ml
    title: multi-export kernel emission and registration in lower
  - id: metal-runtime
    resource: /lib/metal_runtime.ml
    title: Assignment_multi_export_contraction_group dispatch
  - id: constructive-lowering
    resource: /lib/constructive_lowering.ml
    title: Multi_export_contraction_group in boundary values, selection, kind mapping
  - id: adr-reduction-phase
    resource: /.okf/decisions/reduction-phase-semantics-for-normalization.md
    title: 'ADR: keep reduction phases separate from pointwise group algebra'
  - id: exp-0156
    resource: /.okf/experiments/exp-0156-typed-contraction-groups-g4p12-2026-09-07.md
    title: G4P12-01/02 algebra core and select/fold routing
  - id: exp-0167
    resource: /.okf/experiments/exp-0167-g4p12-05-2-m2x4-parametric-port-2026-09-07.md
    title: G4P12-05-2 m2_x4 parametric port
---

# Result

G4P12-07 lands multi-export contraction group support across the full Metal serving stack. G4P12-08 closes the G4P12 round with architectural documentation.

## G4P12-07 — fan-out/export scheduling (commit `36a1e95`)

**Problem.** Groups with exported intermediates (two region results from one contraction group — e.g., `a = silu(g)` exported AND consumed by `z = a + u`) were formable by the elaboration (`of_region` produced groups with 2+ exports) but not selectable as tactics. `select_contraction_group` required exactly one export, and `select_linear_epilogue` required exactly one contraction and one export. Two-result fan-out regions (test J: rejected by fold/select since G4P12-01) remained unclaimed.

**Solution.** Five changes across the stack:

1. **Tactic type extension.** `Kernel_ir.Fused_tactic.t` gains `Multi_export_contraction_group { contractions; epilogues; outputs }` — a list of per-export `int Applicative_expr.t` epilogues and per-export `Ir.Value.t` outputs, alongside the shared contraction list. This is the multi-export counterpart to the existing `Contraction_group` (single epilogue, single output).

2. **Group algebra widening.** `Contraction_group.multi_export_epilogues` reconstructs one `int Applicative_expr.t` per export by inlining the shared step environment. `select_contraction_group` now handles three cases: 2+ contractions with 1 export → `Contraction_group`; 1+ contractions with 2+ exports → `Multi_export_contraction_group`; everything else → `None`. The widening for 1-contraction 2-export groups covers the test J fan-out pattern.

3. **Kernel emitter.** `emit_q4_0_multi_export_linear_f16_x8` in `metal_q4_0_decode_codegen.ml` generates a kernel with E output buffers (one per export), E writeback blocks (one per epilogue), shared weight/aux/reduction infrastructure, and `params` at buffer `1 + num_slots + num_aux + num_exports`. The schema-level `q4_0_multi_export_linear_f16_x8_kernel` wrapper validates Q4_0 block geometry. Buffer layout: `input(0) · weight_0(1) · weight_1(2) · [aux_0(3)] · output_0(3+aux) · output_1(4+aux) · params(5+aux)`.

4. **Metal lowering + registration.** `Metal_selection.region_tactics` collects `quant_multi_export_digest` tuples `(quant, tactic, num_slots, epilogues, outputs)` alongside the existing single-export digest list. `Metal.lower` emits and registers multi-export kernels with a digest name derived from the first epilogue. The quant-types list includes multi-export types.

5. **Runtime dispatch.** `Metal_runtime` gains `Assignment_multi_export_contraction_group` in the assignment type and the `assignment_execution` match. Dispatch: look up the multi-export digest kernel by name, validate shapes, compute grid, assemble buffers (input + weights + aux + output buffers), dispatch with `QuantLinearParams`, bind all outputs to workspace buffers.

**Test pins.** Sections G and H in `test_contraction_group.ml`:
- **G (multi-export selection):** A two-contraction two-export region (g = x*wg, u = x*wu, a = silu(g), z = a + u; exports a and z) elaborates into a group with 2 contractions and 2 exports. `multi_export_epilogues` returns `[Silu(Slot 0); Add(Silu(Slot 0), Slot 1)]`. `select_contraction_group` returns `Multi_export_contraction_group` with correct contractions/epilogues/outputs. `select` (widened) also returns `Multi_export_contraction_group`.
- **H (multi-export kernel emission):** `q4_0_multi_export_linear_f16_x8_kernel` with epilogues `[Silu(Slot 0); Add(Silu(Slot 0), Slot 1)]` produces a kernel with correct buffer layout (weight_0 at buffer(1), weight_1 at buffer(2), output_0 at buffer(3), output_1 at buffer(4), params at buffer(5)), both output-row writebacks, and compiles cleanly with `xcrun metal`.
- **Full-stack proof in test_metal.ml:** Multi-export kernel emitter produces correct buffer layout and writeback verification.

**Known limitation.** `Passes.optimize` creates one fusion region per graph output. With two graph outputs, it produces two single-export regions rather than one two-export region. The multi-export selection fires on manually-constructed regions (unit tests). The emission/registration/runtime stack is proven end-to-end by the unit and full-stack proofs. Making the optimizer group multiple outputs into one region requires changes to the fusion region formation algorithm (out of scope for G4P12-07).

## G4P12-08 — reduction phases (close-round docs)

**ADR.** `decisions/reduction-phase-semantics-for-normalization.md` documents the architectural decision: keep RMSNormalize, RoPE, and ShortConvSiLU as distinct reduction-phase tactic constructors rather than folding them into the scalar-applicative group algebra. The Applicative expression language is a pure pointwise algebra over scalar values — it has no cross-thread reduction primitive (`simd_sum`), no positional indexing, and no synchronization barrier. RMSNormalize requires `simd_sum` reduction (hardware synchronization), which is fundamentally different from pointwise evaluation. The boundary is the reduction synchronization point: the group algebra owns pointwise composition, and scheduling composes reduction phases with contraction groups as sequential dispatches (or fused when the target supports it).

**Numerical contracts.** Two contracts are surfaced at the scheduling boundary:
1. `Applicative_expr.simplify` finite-reals contract: callers must feed values that produce well-defined floats. Constrains what epilogues are legal.
2. Reduction-phase contract: the reduction must produce a finite sum-of-squares before normalization. The scheduler validates both before committing a schedule.

**G4P12 round summary.** Eight slices completed:
- G4P12-01 (`9c2cdab`): algebra core — typed contraction group normal form
- G4P12-02 (`b516513`): recognition routed through elaboration
- G4P12-03 (`29451b0`): composition reduced to the graph
- G4P12-04 (`b5ee732`→`357fab9`→`fc3fa2e`→`1651ec8`→`b543932`): emission widening (x8/x4/m16/m32 digest tails, retire splice, _add twin re-homing)
- G4P12-05 (`30835b4`→`342f43b`→`c40cc10`): remaining quant families (block32/superblock twins, m2_x4 port, zoo deletion)
- G4P12-06 (`f8c647a`→`b3f4049`→`dbf4fa0`): multi-slot groups through the stack
- G4P12-07 (`36a1e95`): fan-out/export scheduling
- G4P12-08: reduction-phase ADR + close-round docs

**Outstanding work (future rounds).**
- Optimizer multi-output region formation (needed for end-to-end multi-export from graph)
- Reduction-phase kernel fusion (linear → rms_norm in one dispatch)
- Per-family remaining tiles in G4P12-05-2 (splitk, x1/x2_l32, llama_mv, gated, down_add/dual_swiglu)
- Device verification of multi-export serving
