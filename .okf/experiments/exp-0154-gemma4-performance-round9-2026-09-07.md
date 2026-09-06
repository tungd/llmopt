---
type: Experiment
title: 'Fused W4A16 Linear-add epilogue schedule routing to parametric Metal kernels'
description: 'Wires the Round 8 contraction-epilogue monoid through real fused-region schedule dispatch: a shared pure route Epilogue.w4a16_algebraic_route admits single-row (m = 1) Groupwise_packed {bits=4; group_elements=64} W4A16 Linear-add regions, Metal.lower emits one deterministic W4A16EpiParams-parameterized kernel (llmopt_w4a16_linear_f16_g64_epi_<digest8>) carrying the residual as an Epilogue.Expr read, and Metal_runtime recomputes the same route and digest to dispatch that kernel; every other layout/geometry keeps today''s byte-identical legacy static kernels.'
tags: [experiment, compiler, gemma4, epilogue, fusion, scheduling, routing, metal, w4a16, algebraic-ir]
status: stable
generated: { by: antigravity/2.0, at: '2026-09-07T04:20:00+07:00' }
sources:
  - id: decision-round8
    resource: /.okf/decisions/applicative-ir-coordinate-category-and-epilogue-monoid.md
    title: Applicative IR, Coordinate Category, and Contraction Epilogue Monoid architecture (round 8)
  - id: epilogue-route
    resource: /lib/epilogue.ml
    title: Shared schedule-routing key (algebraic_residual_add, canonicalize, msl_buffer_decls, algebraic_kernel_name, w4a16_algebraic_route)
  - id: metal-lowering
    resource: /lib/metal.ml
    title: emit_w4a16_algebraic_kernels parametric emitter and lower/region_tactics routed registration
  - id: metal-runtime
    resource: /lib/metal_runtime.ml
    title: encode_schedule Linear-add dispatch branch recomputing the routing key
  - id: test-suite
    resource: /test/test.ml
    title: Section 12 routing assertions (route policy, canonical naming, lowering, ABI registration, legacy negative)
  - id: loop-round9
    resource: /LOOP-gemma4-performance-round9.md
    title: Gemma 4 performance round-9 loop checklist
---

# Result

Round 9 carries the fused epilogue from algebraic representation to executed schedule for Gemma 4 W4A16 decode linear-adds, plus the review-hardened IR foundation it rests on:

1. **Review hardening (G4P9-01, commit `46416c3`)**: `Coordinate_map` sealed + validated with the `Permute` fold direction corrected; `Applicative_expr.equal` and `required_values` contracts made total; `Epilogue.equal` made a congruence; a dead result computation dropped; tests and docs updated.
2. **One shared routing policy**: `Epilogue.w4a16_algebraic_route ~storage ~m ~residual` admits exactly `Groupwise_packed { bits = 4; group_elements = 64 }` at `m = 1` and returns `Some (canonicalize (algebraic_residual_add ~residual))`, otherwise `None`. Both the lowering site (`Metal.region_tactics`) and the runtime dispatch site (`Metal_runtime.encode_schedule`) recompute this same pure function, so emission and dispatch cannot drift.
3. **Canonical parametric emission**: `Metal.emit_w4a16_algebraic_kernels` emits one `kernel void llmopt_w4a16_linear_f16_g64_epi_<digest8>` per routed region. Slot convention follows the legacy kernels so the existing runtime binder applies unchanged in list order: `input [[buffer(0)]]`, `packed_weight [[buffer(1)]]`, `scale [[buffer(2)]]`, writable `output [[buffer(3)]]`, the canonical residual as `device const half* buf_0 [[buffer(4)]]` (via `Epilogue.msl_buffer_decls ~start_index:4`), and `constant W4A16EpiParams& params [[buffer(5)]]` (`struct W4A16EpiParams { uint m; uint n; uint k; };`). The streaming body is the legacy g64 algorithm with the bias add replaced by the register writeback `output[row * params.n + col] = half((acc + ((float)buf_0[col])));` — provably equal to the legacy `bias_or_scale[row * bias_row_stride + col]` flat-add contract at `row = 0`.
4. **Deterministic runtime routing**: `canonicalize` re-numbers epilogue tensor inputs by stable first-occurrence ordinal, and `Epilogue.algebraic_kernel_name ~base` digests the canonical writeback plus every required tensor's dtype. Structurally equal residuals therefore collide on one name no matter their graph `Value_id` (verified with a `Value_id` 777 alias against the emitted id-0 epilogue). The runtime `Some` branch resolves that name through `kernel_entry`, binds `[input; weight; scale_buffer; output_buffer; additive_buffer]`, sends parameters `[m; n; k]` (3 u32s) and grid `linear_f16_grid (m * n)`, and returns the bound output buffer — the same `Kernel_abi` entry `Metal.lower` registered with `threadgroup (256,1,1)`, `operation W4a16_linear`, `Float16` in/out.
5. **Legacy path untouched**: `None` routing (multi-row W4A16, `group_elements ≠ 64`, dense/non-packed layouts, any non-Linear-add region) keeps today's byte-identical static `llmopt_w4a16_linear_f16_g64`/`g64_m4` dispatch, so the behavioral invariant holds by construction.

Design constraint honored during the slice: the residual operand of a fused region must be a graph input (`Ir.Graph.tensor_input`) — a dangling fresh value produced by nothing fails compute-plan validation with `Failure("node ... reads value ... with no producer")`; the working end-to-end test idiom lowers `Passes.Optimization.execution_graph` of the fused optimization.

# Static Validation

`ninja -f ninja.build test` passes cleanly — 79 Python unit tests OK and every canonical OCaml suite PASS line green. Section 12 routing assertions (all green):

- `routing admits single-row groupwise-4/64 Linear-add residuals`
- `routing declines multi-row W4A16 Linear-add regions`
- `routing keeps non-groupwise-4/64 layouts on the legacy kernel path`
- `routing canonicalizes the residual read to buffer ordinal 0`
- `routed kernel names are deterministic digests under the epi base`
- `msl_buffer_decls binds the canonical residual at buffer(4)`
- `Metal lowering emits the routed parametric W4A16 Linear-add kernel` (source contains `kernel void <name>`, `W4A16EpiParams`, `device const half* buf_0 [[buffer(4)]]`, and `(acc + ((float)buf_0[col]))`)
- `Metal lowering registers the routed kernel ABI entry`
- `routed kernel names survive structural renaming of the residual tensor`
- `multi-row regions keep today's static legacy W4A16 kernel path` (m = 4 program contains no `kernel void llmopt_w4a16_linear_f16_g64_epi` and no `W4A16EpiParams`)

`git diff --check` is clean. Commits on `feat/unified-optimizing-compiler`: `46416c3` (G4P9-01 refactor) and `fde9817` (G4P9-02 routing).
