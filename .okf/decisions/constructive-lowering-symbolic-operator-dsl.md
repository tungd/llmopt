---
type: Decision
title: 'Constructive Lowering via Algebraic Inlining and Symbolic Operator DSL'
description: 'Replace fragile AST subgraph pattern matching with constructive lowering in OCaml, using overloaded symbolic operators and arrayjit algebraic virtualization to fuse elementwise, normalization, and activation chains by construction.'
tags: [decision, compiler, lowering, symbolic-dsl, inlining, virtualization, fusion, fx]
status: approved
generated: { by: human:tung, at: '2026-08-31T14:05:00+07:00' }
sources:
  - id: fusion-query
    resource: /lib/fusion_query.ml
    title: AST subgraph query engine
  - id: pass-fuse-rms
    resource: /lib/pass_fuse_rms_norm.ml
    title: RMSNorm AST pattern matching pass
  - id: pass-fuse-swiglu
    resource: /lib/pass_fuse_swiglu_ffn.ml
    title: SwiGLU AST pattern matching pass
  - id: arrayjit-lowering
    resource: /.okf/prior-art/ocannl-arrayjit.md
    title: OCANNL arrayjit loop-nest compiler analysis
---

# Decision

`llmopt` adopts **constructive algebraic lowering with an OCaml symbolic operator DSL**, replacing fragile AST subgraph pattern matching (`fusion_query.ml` and bespoke `pass_fuse_*.ml` passes). 

Instead of searching for rigid graph shapes across PyTorch FX decompositions, `llmopt` ingests FX operations directly into symbolic Assignment trees (`Assignments.t` / `Low_level.scalar_t`), relying on `arrayjit`'s **Virtualization (`virtual_llc`)** and **Scalar Inlining (`simplify_llc`)** to fuse multi-op chains automatically by construction.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. PyTorch FX Ingestion (Direct translation to symbolic expressions)        │
│    aten.pow(x, 2) ──► sqr x                                                 │
│    aten.mean(..., -1) ──► mean (sqr x)                                      │
│    aten.rsqrt(x + eps) ──► rsqrt (var +! eps)                               │
│    aten.mul(x, inv_std) ──► x *! inv_std *! weight                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Symbolic OCaml DSL (Math_dsl)                                            │
│    Constructs typed Low_level.scalar_t trees using overloaded operators:   │
│    (+!), (*!), (-!), (/!), (**!), rsqrt, silu, sigmoid, exp, tanh          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Algebraic Virtualization (virtual_llc)                                   │
│    Liveness analysis detects single-use intermediate activations and marks   │
│    them Virtual (zero DRAM / buffer memory allocation).                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. Constructive Inlining & CSE (simplify_llc & hoist_cross_statement_cse)   │
│    Substitutes scalar trees directly into consumer loop registers:          │
│    out[i] = x[i] * rsqrt(mean_var + eps) * weight[i]                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. Target Code Generation (Metal MSL / CUDA PTX / AMD HIP)                  │
│    Emits single-pass fused kernel loops with zero intermediate memory round-│
│    trips, without requiring a specialized fusion pass.                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

# Context & Problem

The current `llmopt` fusion pipeline relies on **AST subgraph pattern matching**:
- [`pass_fuse_rms_norm.ml`](file:///Users/tung/Projects/std23/llmopt/lib/pass_fuse_rms_norm.ml) is 686 lines long.
- [`pass_fuse_swiglu_ffn.ml`](file:///Users/tung/Projects/std23/llmopt/lib/pass_fuse_swiglu_ffn.ml) is 500 lines long.
- [`pass_fuse_linear_bias.ml`](file:///Users/tung/Projects/std23/llmopt/lib/pass_fuse_linear_bias.ml) is 600 lines long.
- [`fusion_query.ml`](file:///Users/tung/Projects/std23/llmopt/lib/fusion_query.ml) maintains a complex query engine over AST nodes.

### Failure Modes of Pattern Matching:
1. **Decomposition Brittleness**: If PyTorch Dynamo emits `pow(x, -0.5)` instead of `rsqrt(x)`, or inserts an explicit cast or view node, the AST pattern query fails to match.
2. **High Maintenance Debt**: Adding a new model architecture or activation variation (e.g. Squared ReLU, GeGLU, RMSNorm scaling tweaks) requires writing and debugging a new 500+ line pattern matcher.
3. **All-or-Nothing Degradation**: When a query fails to match, execution falls back to un-fused multi-kernel dispatches with heavy DRAM activation traffic.

# Architecture & Implementation

### 1. Symbolic Operator DSL (`Math_dsl`)
A clean, strongly-typed OCaml module exposing overloaded operators that construct `Low_level.scalar_t` expressions:

```ocaml
module Math_dsl = struct
  type t = Low_level.scalar_t

  let ( +! ) a b = Low_level.Binop (Ops.Add, (a, Ops.single), (b, Ops.single))
  let ( *! ) a b = Low_level.Binop (Ops.Mul, (a, Ops.single), (b, Ops.single))
  let ( -! ) a b = Low_level.Binop (Ops.Sub, (a, Ops.single), (b, Ops.single))
  let ( /! ) a b = Low_level.Binop (Ops.Div, (a, Ops.single), (b, Ops.single))
  let ( **! ) a p = Low_level.Binop (Ops.Pow, (a, Ops.single), (Low_level.Constant p, Ops.single))

  let rsqrt x = Low_level.Unop (Ops.Rsqrt, (x, Ops.single))
  let silu x  = x *! Low_level.Unop (Ops.Sigmoid, (x, Ops.single))
  let get tn idx = Low_level.Get (tn, idx)
end
```

### 2. Ingest and Algebraic Inlining
When an FX node is processed:
1. Its mathematical semantics are emitted directly via `Math_dsl`.
2. `virtual_llc` tags intermediate nodes as `Local_scope` (virtual registers).
3. `simplify_llc` inlines the scalar expression into the enclosing loop nest.
4. `hoist_cross_statement_cse` hoists common sub-terms into local variables (`Declare_local`).

### 3. Progressive Deprecation
- **Phase 1**: Introduce `Math_dsl` and constructive lowering alongside existing passes.
- **Phase 2**: Route RMSNorm, SwiGLU, and activation-product fusions through constructive lowering; verify bit-exact parity.
- **Phase 3**: Retire `pass_fuse_rms_norm.ml`, `pass_fuse_swiglu_ffn.ml`, and `fusion_query.ml`.
