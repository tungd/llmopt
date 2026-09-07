---
type: Experiment
title: 'Typed contraction groups (G4P12-01/02): shared-assignment-graph algebra core, then select/fold routed through the elaboration'
description: 'Round G4P12-01/02 establish the typed contraction-group program: G4P12-01 (9c2cdab) adds the additive scalar-DAG elaboration core lib/contraction_group.ml/.mli with unit tests proving y*y folds at the algebra level; G4P12-02 (b516513) routes consumer-facing recognition through the elaboration — select/fold_linear_epilogue legacy-first with byte-identical tag reconstruction, widened to single-slot branch shapes — cascades all four consumers (metal region_tactics, serving_schedule validation, metal_runtime assignment execution, constructive_lowering folded-candidate claims) to Contraction_group.select, and pins the y*y square end to end through the Q4_0-x8 digest path (one two-binding Linear_add event, zero-buffer digest kernel, output at buffer(2)/params at buffer(3)).'
tags: [experiment, compiler, gemma4, contraction-group, elaboration, kernel-ir, fusion, epilogue, metal, routing]
status: stable
generated: { by: poolside/1.0, at: '2026-09-07T10:00:00+07:00' }
sources:
  - id: loop
    resource: /LOOP-typed-contraction-groups.md
    title: Typed Contraction Groups G4P12 executor checklist (active program, supersedes G4P11)
  - id: contraction-group
    resource: /lib/contraction_group.ml
    title: of_region/classify grammar, single_acc_epilogue reconstruct, G4P12-02 select/fold_linear_epilogue widening section
  - id: kernel-ir
    resource: /lib/kernel_ir.ml
    title: Fused_tactic (select/select_all/fold_linear_epilogue legacy oracle), region/assignment model
  - id: constructive-lowering
    resource: /lib/constructive_lowering.ml
    title: folded_candidates claims through select_linear_epilogue filter; legacy-first candidates then elaborated fallback
  - id: metal
    resource: /lib/metal.ml
    title: region_tactics on Contraction_group.select; quant_digest/quant_add buckets; Q4_0-x8 digest emitter
  - id: metal-runtime
    resource: /lib/metal_runtime.ml
    title: assignment_execution dispatch through Contraction_group.select on fusion regions
  - id: serving-schedule
    resource: /lib/serving_schedule.ml
    title: validate_fusion_command admission through Contraction_group.select
  - id: test-suite
    resource: /test/test.ml
    title: Fold_catalog section K (y*y unit pins), full-stack Q4_0 y*y digest proof
  - id: exp-round10
    resource: /.okf/experiments/exp-0155-gemma4-performance-round10-2026-09-08.md
    title: Round 10 record (monoid-general digest service) that this program extends and supersedes
---
# Result

The typed contraction-group program (G4P12, LOOP-typed-contraction-groups.md) lands its first two slices, making the shared SSA assignment graph the algebraic object recognition runs over instead of single-slot tag shapes:

1. **G4P12-01 (commit `9c2cdab`) — algebra core, additive, zero behavior change.** `lib/contraction_group.ml/.mli` introduce the typed normal form: a `group` of one-or-more bias-free Linear `contraction`s plus SSA `step` bindings (`expr = Slot | Const | Read | Unop | Binop` in the Applicative grammar; Cast classified as `Unop (Cast …)`, Add/Mul canonicalized external-first to internal-first), exports equal to the region results, `of_region` requiring every binding to feed a result with no Opaque on result paths. `single_acc_epilogue` inlines steps into one monoid expression and `reconstruct` maps step-shape normal forms back to legacy tags (`none`/`silu`/`gelu`/`residual_add`) so the algebraic tree and the role metadata both survive. `test/test_contraction_group.ml` proves equivalence and the three slice exercises at the algebra level (y*y single-slot, two-projection two-slot, exported intermediate). Legacy `Kernel_ir` recognizers stayed the untouched pinned oracle this slice.
2. **G4P12-02 (commit `b516513`) — recognition routed through the elaboration (single-acc proof).** `Contraction_group.select`/`fold_linear_epilogue`/`select_linear_epilogue` are exported from the algebra module (NOT added to `kernel_ir`, which cannot reference `Contraction_group` — OCaml cmi self-shadowing makes that compile path impossible) and cascade at every consumer call site: `metal.ml` `region_tactics`, `serving_schedule.ml` `validate_fusion_command`, `metal_runtime.ml` `assignment_execution`, and `constructive_lowering.ml` `folded_candidates` (legacy candidate list first, elaborated `select_linear_epilogue` fallback only when the legacy fold finds nothing). `select` runs the legacy `Kernel_ir.Fused_tactic.select` first — every already-folded tag shape stays byte-identical (tags reconstructed, incl. the one semantic spot `is_residual_add_epilogue`) — and falls back to the elaboration widened to single-slot branch shapes: a `y*y` square admits as `Linear_epilogue` with `Epilogue.Expr (Binop (Mul, Slot Acc, Slot Acc))` (m=1, n=64, k=64 in the pin), where the legacy fold and select both return `None`. Casts and Sigmoid cuts still cut both paths. `select_linear_epilogue` additionally requires a non-empty step list (bare identity stays unclaimed) and `group.steps <> []`, keeping test H/G/I/E/F semantics unchanged. Non-Q4_0 quants keep failing loudly at `lower` with the exact existing message.

New pins added in the same commit (Fold_catalog section K + contraction-group section B + full-stack proof):

- **Unit pins.** The `y*y` region (`Linear → Mul(acc, acc)`, one result) yields `fold`/`select` = `None` on the legacy recognizers but `Some` from `Contraction_group.select` with `Epilogue.equal` to `of_expr (Binop (Mul, Slot Acc, Slot Acc))`, and `Contraction_group.fold_linear_epilogue` returns the same epilogue.
- **Full-stack Q4_0-x8 digest proof.** A real `Ir.Graph` (`Linear [1;3840] × Q4_0[4096;3840]` then `Pointwise.Binary Mul` over the projection duplicated as both inputs, f16 `[1;4096]` output) optimizes to ONE `Linear_add` event with two member node ids, inputs `{input, weight}`, outputs `{output}`, zero legacy fallbacks, and fusion-region binding primitives `[linear Block_quantized Q4_0; binary Mul]`. The emitted source contains `llmopt_q4_0_linear_f16_x8`, the `_x8_` digest kernel, and the zero-epilogue-input header `device half* output [[buffer(2)]], constant QuantLinearParams& params [[buffer(3)]]` — a `y*y` digest kernel has zero epilogue buffers, so the accumulator-square writeback rides the existing digest emitter with no new emission code.

The OCaml self-type constraint is the reason the widening lives in `contraction_group.ml` helpers consumed at call sites: `kernel_ir.ml` cannot call `Contraction_group` (its own in-progress cmi shadows `Kernel_ir.region`, verified with a minimal repro). Dependency direction stays contraction_group → kernel_ir and the interface cmi graph stays acyclic.

## Status of the slice

- **Done through G4P12-02.** All legacy pins (Fold_catalog A–J incl. bare-identity, stacked residual-add, cast cuts, two-result fan-out, Sigmoid cuts) unchanged and green; new y*y pins green; full `ninja -f ninja.build test` ends `Ran 79 tests ... OK`; `git diff --check` clean; HEAD `b516513`.
- **Next**: G4P12-03 strips the typed cases from `Epilogue.compose` (graph-level composition; tags = normal-form reconstruction with explicit roles; byte-identical pins), then emission widening (G4P12-04/05), multi-slot groups through the stack (G4P12-06), fan-out/export scheduling (G4P12-07), reduction phases + numerics + round close-out (G4P12-08).
- Device/bundle benchmarks remain user-run (standing constraint).

## Supersedes

exp-0155 (Round 10) recorded the monoid-general digest service whose recognizers this program generalizes; G4P11 (LOOP-quant-family-parametric.md) is superseded by G4P12 as the active program. exp-0155/exp-0154 stay in the register as the historical records of the paths they document.

# Static Validation

`ninja -f ninja.build ocaml-test` passes with the canonical suite tail now reading: `All typed contraction-group algebra tests passed!` (G4P12-01 suite), `llmopt chain-absorption fold catalog passed`, `llmopt single-accumulator y*y digest proof passed` (new full-stack proof), `Running fused Linear_epilogue fold catalog tests...`, `llmopt fused Linear_epilogue fold catalog passed` (Fold_catalog incl. new section K), and `llmopt canonical W4A16/KVQ8 tests passed` final. `ninja -f ninja.build test` passes — 79 Python unit tests OK. `git diff --check` clean; working tree committed at `b516513` with no residual changes before this docs slice.
