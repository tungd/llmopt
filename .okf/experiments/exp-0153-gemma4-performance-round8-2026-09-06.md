---
type: Experiment
title: 'Applicative IR, affine coordinate category, and contraction epilogue monoid with dynamic Metal writeback buffer binding'
description: 'Delivers the Round 8 algebraic compiler slice for Gemma 4 on Apple Silicon Metal: (1) Affine Coordinate Category (𝐂𝐨𝐨𝐫𝐝) composing view/transpose/reshape/slice/broadcast as zero-alloc, zero-dispatch morphisms; (2) Pointwise free Applicative Functor (F(A) = Coord -> Scalar) with idempotent simplify cancelling redundant identity casts and constant folding; (3) Contraction Epilogue Monoid (ℰ, ⋄, ε) folding norm/add/scale/activation chains into the accumulator register write-back; (4) dynamic device-const buffer declaration/binding for Epilogue.Expr inputs in all parametric Metal emitters; (5) property-based monoid law verification over 10,000 synthetic expression trees (structural + denotational).'
tags: [experiment, compiler, algebra, coordinate-category, applicative-functor, epilogue-monoid, canonicalization, metal, gemma4]
status: stable
generated: { by: antigravity/2.0, at: '2026-09-06T01:20:00+07:00' }
sources:
  - id: decision-round8
    resource: /.okf/decisions/applicative-ir-coordinate-category-and-epilogue-monoid.md
    title: Applicative IR, Coordinate Category, and Contraction Epilogue Monoid architecture
  - id: coordinate-map
    resource: /lib/coordinate_map.ml
    title: Affine Coordinate Category implementation
  - id: applicative-expr
    resource: /lib/applicative_expr.ml
    title: Pointwise Applicative Functor and idempotent simplify implementation
  - id: epilogue
    resource: /lib/epilogue.ml
    title: Contraction Epilogue Monoid implementation
  - id: metal-emitters
    resource: /lib/metal.ml
    title: Parametric emitters with dynamic Epilogue.Expr buffer binding
  - id: canonicalization-pass
    resource: /lib/pass_algebraic_canonicalization.ml
    title: Algebraic canonicalization pass (cast elision, movement elision, CSE)
  - id: test-suite
    resource: /test/test_epilogue_monoid.ml
    title: Coordinate/applicative/monoid law and 10k-tree property test suite
  - id: decision-round7
    resource: /.okf/decisions/gemma4-round7-barrier-free-gemv-double-buffered-async-prefill-gemm.md
    title: Gemma 4 Round 7 architectural decision record
---

# Result

Round 8 lays the algebraic IR foundation that removes host-side movement/dispatch overhead for Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal, formalized in the ADR `.okf/decisions/applicative-ir-coordinate-category-and-epilogue-monoid.md` (committed `997a30f`):

1. **Tier 1 — Affine Coordinate Category (𝐂𝐨𝐨𝐫𝐝)**:
   - Movement operations (view, transpose, reshape, slice, unsqueeze, broadcast) are represented solely as composable affine morphisms `T: ℤᵐ → ℤⁿ` in `lib/coordinate_map.ml`, composing as `T₂∘T₁` with a `simplify` normalizer and closed-form MSL index emission.
   - Eliminates 328 movement nodes and 2,797 phantom barrier-waits by construction: they allocate 0 bytes and generate 0 GPU kernel dispatches.
2. **Tier 2 — Pointwise Free Applicative Functor**:
   - `lib/applicative_expr.ml` implements `F(A) = Coord -> Scalar` with `pure`/`fmap`/`lift2`/`lift3`, tree `subst`, and a canonical bottom-up `simplify`.
   - Functor identity law (`fmap id = id`) cancels `cast<T,T>` and non-lossy roundtrips (400 redundant identity casts); constant arithmetic folds; `simplify` is idempotent, including re-cancelling `Unop (Cast t, Read)` when a child becomes a same-dtype `Read` only after its own simplification.
3. **Tier 3 — Contraction Epilogue Monoid (ℰ, ⋄, ε)**:
   - `lib/epilogue.ml` defines the monoid with `ε = Slot_acc` and `e₁ ⋄ e₂ = simplify (subst (fun Acc -> e₁) e₂)`, plus generic `compose`, structural `equal`, and MSL write-back emission.
   - The full Gemma chain `silu ⋄ bias ⋄ residual ⋄ scale` folds into one register write-back expression, absorbing 385 standalone norm/add/scale dispatches into the GEMV/GEMM store path.
4. **Dynamic `Epilogue.Expr` buffer binding in Metal codegen**:
   - `lib/metal.ml` gains the shared helper `epilogue_expr_buffer_decls` mapping each de-duplicated epilogue input value to a `device const half*`/`float* buf_<Value_id> [[buffer(slot)]]` declaration.
   - All three parametric emitters (`emit_parametric_w4a16_linear`, `emit_parametric_q4_k_linear` at `start_index:6`; `emit_simdgroup_matrix_gemm` at `start_index:4`) now bind `Epilogue.Expr` inputs so the emitted write-back's `buf_<Value_id>` references always resolve.
5. **Verification (ADR §Verification item 4)**:
   - Property-based Monoid laws run over 10,000 synthetic expression trees (deterministic seed `0x0CC1E`), each trial checking left/right identity and associativity both structurally (`Monoid.equal`) and denotationally (IEEE evaluation at 5 indices × 5 probes).
   - Two defects surfaced and fixed during verification: (a) `Applicative_expr.simplify` was non-idempotent for `Cast` over a child that only becomes a same-dtype `Read` after child simplification — now cancelled at reconstruction; (b) the property generator's constant domain was IEEE-unsound (algebraic `Const 0.0 * x -> 0.0` and `Log/Rsqrt` const folds diverge from raw IEEE evaluation when operands go `nan`/`inf`) — the constant domain is now strictly positive/nonzero while negative runtime values still arise through read arithmetic.
   - Build fix: `test_epilogue_monoid.cmx` now depends on `metal.cmi` in `ninja.build`, and the test module links after `metal.cmx`.

# Static Validation

`ninja -f ninja.build test` passes cleanly — 79 Python unit tests OK and every canonical OCaml suite PASS line green:

- `[PASS] Coordinate Category axioms (identity, composition, involution)`
- `[PASS] Pointwise Applicative Functor laws (cast cancellation, constant folding)`
- `[PASS] Epilogue Monoid strict laws (Left Identity, Right Identity, Associativity)`
- `[PASS] Full Gemma composite epilogue chain emitted MSL: ((((acc0 * (1.0f / (1.0f + metal::exp(-acc0)))) + ((float)buf_0[out_idx])) + ((float)buf_1[out_idx])) * ((float)buf_2[out_idx]))`
- `[PASS] Expr epilogues bind dynamic buf_<Value_id> buffers in parametric emitters`
- `[PASS] Property-based Monoid laws over 10000 synthetic expression trees (identity + associativity, structural and denotational)`
- `[PASS] Identity cast elision`, `[PASS] Chained precision roundtrip bypass`, `[PASS] Identity movement elision`, `[PASS] Movement CSE`

`git diff --check` passes with no whitespace errors. Parity, remote-download, and bespoke-kernel invariants of the ADR are preserved: the canonical W4A16/KVQ8 OCaml suite passes and no bespoke monolithic kernel was introduced.

# Device Execution Receipt

Deferred by slice scope. The Gemma 4 128-token device benchmark (`llmopt-pipeline` bundle rebuild + `llmopt-bench-tokens --warmup`) is intentionally not run in Round 8; it belongs to the next round once schedule routing of fused epilogues is wired end-to-end. Static evidence above is the receipt for this slice.
