---
type: Decision
title: 'Reduction-Phase Semantics for RMSNorm/RoPE Within Typed Contraction Groups'
description: 'Keep RMSNormalize and RoPE as distinct reduction-phase tactic constructors rather than folding them into the scalar-applicative group algebra; represent reduction phases explicitly in the scheduling model with domain, synchronization, and retained-value metadata.'
tags: [decision, compiler, contraction-group, reduction, normalization, rmsnorm, rope, scheduling, metal]
status: stable
generated: { by: poolside/1.0, at: '2026-09-11T10:00:00+07:00' }
sources:
  - id: loop
    resource: /LOOP-typed-contraction-groups.md
    title: Typed Contraction Groups G4P12 executor checklist
  - id: contraction-group
    resource: /lib/contraction_group.ml
    title: Typed group algebra core — of_region, multi_acc_epilogue, select
  - id: kernel-ir
    resource: /lib/kernel_ir.ml
    title: Fused_tactic with Rms_normalize, Rms_rope, Contraction_group, Multi_export_contraction_group
  - id: applicative-expr
    resource: /lib/applicative_expr.mli
    title: Polymorphic applicative expression tree with finite-reals soundness contract
  - id: metal-runtime
    resource: /lib/metal_runtime.ml
    title: Runtime dispatch for RMS normalize, RoPE, and contraction group tactics
  - id: exp-0156
    resource: /.okf/experiments/exp-0156-typed-contraction-groups-g4p12-2026-09-07.md
    title: G4P12-01/02 algebra core and select/fold routing
---

# Decision

`llmopt` keeps RMSNormalize, RoPE, and ShortConvSiLU as **distinct reduction-phase tactic constructors** (`Kernel_ir.Fused_tactic.Rms_normalize`, `Rms_rope`, `Short_conv_silu`) rather than folding them into the scalar-applicative group algebra (`Applicative_expr.t` over contraction-group slots).

## Context

The typed contraction-group program (G4P12) made the shared SSA assignment graph the algebraic object for contraction fusion. `Contraction_group.of_region` classifies region bindings as contractions (Linear) or foldable scalar steps (pointwise over the Applicative grammar), and `select_contraction_group` produces `Contraction_group` / `Multi_export_contraction_group` tactics with algebraic epilogues.

RMSNormalize is a three-phase operation: (1) **reduction** — compute sum-of-squares over the row dimension via `simd_sum`, requiring cross-thread synchronization; (2) **normalization** — divide by `sqrt(mean + epsilon)`, a per-element pointwise op; (3) **optional scaling** — multiply by a weight tensor, another per-element pointwise op.

RoPE adds position-dependent rotation: apply `cos`/`sin` rotary embeddings to pairs of adjacent elements, interleaving query/key tensor halves. The position indexing (pair stride, half-width) and the `cos`/`sin` lookups are domain-specific.

The Applicative expression language (`'var t`) is a pure pointwise algebra over scalar values — it has no cross-thread reduction primitive, no positional indexing, and no synchronization barrier. Its soundness contract assumes finite reals: every expression evaluates to a well-defined float at each element position independently.

## Decision

**Keep reduction-phase tactics separate from the pointwise group algebra.** Concretely:

1. **Rms_normalize and Rms_rope stay as `Fused_tactic` constructors** with their own fields (input, weight, residual, epsilon, config). They are not expressible as `Applicative_expr.t` epilogues because they require cross-thread `simd_sum` reduction.

2. **The group algebra owns pointwise composition only.** `Contraction_group` / `Multi_export_contraction_group` handle Linear contractions + pointwise scalar steps. The boundary is the reduction synchronization point.

3. **Scheduling composes reduction phases with contraction groups.** A future scheduler can chain a contraction group (fused linear + pointwise epilogue) with a normalization step (reduction + rescale) as sequential dispatches, or fuse them into a single kernel when the reduction can be inlined (e.g., for small `n` fitting one simdgroup). The scheduling decision is target-aware and cost-model-scored.

4. **Numerical contracts are surfaced at the scheduling boundary.** The `Applicative_expr.simplify` contract (finite reals: callers must feed values that produce well-defined floats) constrains what epilogues are legal. Reduction phases add a second contract: the reduction must produce a finite sum-of-squares before normalization. The scheduler validates both contracts before committing a schedule.

## Rationale

- **Expressiveness boundary is real.** `simd_sum` is a hardware synchronization primitive (simdgroup barrier). The Applicative algebra cannot represent it — adding a `Reduce` constructor would break the pointwise assumption and require every consumer (simplify, canonicalize, emit_msl_expr, algebraic_kernel_name) to handle a fundamentally different evaluation model.

- **Existing emitters already separate the phases.** The Metal RMS normalize kernel (`rms_norm_source` in `metal.ml`) runs the reduction in a separate kernel dispatch from the contraction group kernel. The runtime dispatches them as independent commands. Folding normalization into the group would require a new kernel emitter that interleaves reduction with contraction — a complexity not justified by current serving shapes.

- **Future fusion is compatible.** A scheduler that wants to fuse "linear → silu → add → rms_norm" into one kernel can express the contraction group (linear + silu + add) as a `Multi_export_contraction_group` with the normalization as a post-group reduction phase. The reduction phase would carry metadata (domain = row, synchronization = simdgroup_sum, retained = weight, epsilon) that the kernel emitter consumes.

- **`Applicative_expr.simplify` contract stays clean.** The finite-reals contract applies only to pointwise expressions. Reduction phases have their own contract (finite sum-of-squares). Keeping them separate avoids conflating two different soundness requirements.

## Consequences

- RMSNormalize/RoPE/ShortConvSiLU continue to be discovered by their own recognizers in `Kernel_ir.Fused_tactic.select` (topology-based pattern matching over reduction primitives). They are not discovered by `Contraction_group.select` (which operates over pointwise scalar steps).

- The `region_tactics` function in `Metal_selection` collects reduction-phase tactics (`has_rms`, `has_rms_rope`, `has_short_conv_silu`) separately from contraction-group tactics (`quant_multi_digest`, `quant_multi_export_digest`). The Metal program includes both kernel families.

- A future "fused reduction" kernel (linear → rms_norm in one dispatch) would add a new `Fused_tactic` variant combining a contraction group with reduction-phase metadata, not by extending `Applicative_expr`.

- The cost model scores reduction-phase dispatches independently. The "keep local" vs "materialize" scheduling choice from G4P12-07 applies to contraction-group exports; reduction phases are always separate dispatches (or fused with the preceding group when the target supports it).
