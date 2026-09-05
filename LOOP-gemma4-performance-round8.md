# Gemma 4 Performance & Throughput Parity Loop - Round 8

## Goal

Lay the algebraic compiler foundation that removes the remaining *host-side dispatch* overhead between `llmopt` and `llama.cpp` for Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal (M4 Pro), through three tiered architectural pillars formalized in the Round 8 ADR:

1. **Tier 1 — Affine Coordinate Category (𝐂𝐨𝐨𝐫𝐝 / Index Monoid)**: Represent all movement operations (view, transpose, reshape, slice, unsqueeze, broadcast) as composable affine morphisms `T: ℤᵐ → ℤⁿ` that allocate 0 bytes and generate 0 GPU kernel dispatches, eliminating 328 movement nodes and 2,797 phantom barrier-waits by construction.
2. **Tier 2 — Pointwise Applicative Functor**: Represent pointwise chains as `F(A) = C → A` with identity-law cast cancellation (`cast<T,T>` and non-lossy roundtrips simplify to identity at construction time), eliminating 400 dead identity casts, and folding arbitrary pointwise DAGs into single register ASTs.
3. **Tier 3 — Contraction Epilogue Monoid (ℰ, ⋄, ε)**: Fold downstream pointwise chains (norm/add/scale/activation) directly into the GEMV/GEMM accumulator register write-back via the monoid `ε = Slot_acc`, `e₁ ⋄ e₂ = e₂[Slot_acc ↦ e₁]`, absorbing 385 standalone dispatches into the parametric emitter store path, and declare/bind each `Epilogue.Expr` input value as a dynamic `device const` buffer in the Metal codegen.

Target invariants:
- **Parity Invariant**: Maintain 100% bit-exact token parity on prompt `2,105,9731,...` generating `100, 45518, 107, 14947, 135383,...` (device re-run deferred to the next round per slice scope).
- **Architectural Invariant**: Zero bespoke monolithic kernels and zero remote downloads; all quantized linear operators remain parameterized emissions from schema definitions in `lib/metal_block_codegen.ml` / `lib/metal.ml`.
- **Verification Invariant (ADR §Verification)**: (1) 128-token parity invariant; (2) zero remote downloads; (3) zero bespoke monolithic kernels; (4) property-based identity + associativity tests over 10,000 synthetic expression trees in `test/test_epilogue_monoid.ml`.

## Out of scope

- Running the Gemma 4 128-token device benchmark or `llmopt-pipeline` bundle rebuild in this slice (GPU run deferred by decision; next round after schedule routing of fused epilogues is wired end-to-end).
- Writing handwritten monolithic kernels in `lib/metal.ml`.
- Downloading remote weights or GGUF files.
- Multi-token speculative decoding / MTP assistant serving in this milestone.

## Relevant files

- `lib/coordinate_map.ml` / `lib/coordinate_map.mli`: Affine Coordinate Category (morphisms, composition, simplify, `Permute`/`Identity` emission).
- `lib/applicative_expr.ml` / `lib/applicative_expr.mli`: Pointwise free Applicative Functor AST, `fmap`, `lift2`/`lift3`, `subst`, idempotent `simplify`, MSL emission, `required_values`.
- `lib/epilogue.ml` / `lib/epilogue.mli`: Contraction Epilogue Monoid (`Expr`, `Monoid`, `empty`, `append`, `compose`, `equal`, `to_string`, `inputs`).
- `lib/pass_algebraic_canonicalization.ml` / `.mli`, `lib/passes.ml`, `lib/pass_co_schedule.ml`, `lib/serving_schedule.ml`: canonicalization pass wiring and schedule integration.
- `lib/metal.ml`: Parametric emitters `emit_parametric_w4a16_linear`, `emit_parametric_q4_k_linear`, `emit_simdgroup_matrix_gemm` and dynamic `Epilogue.Expr` buffer declaration/binding helper.
- `lib/metal_runtime.ml`: `epilogue_mode` u32 parameter plumbing for `_add`-suffixed kernels (reserved path).
- `test/test_epilogue_monoid.ml`, `test/test_algebraic_canonicalization.ml`, `test/test.ml`: law and codegen test suites.
- `.okf/decisions/applicative-ir-coordinate-category-and-epilogue-monoid.md`: Governing OKF decision record.

## Supporting documents

- `.okf/decisions/applicative-ir-coordinate-category-and-epilogue-monoid.md`: `EXISTING` - Architectural decision record for Round 8 (committed `997a30f`).
- `.okf/decisions/gemma4-round7-barrier-free-gemv-double-buffered-async-prefill-gemm.md`: `EXISTING` - Round 7 architectural decision record.
- `.okf/experiments/exp-0152-gemma4-performance-round7-2026-09-05.md`: `EXISTING` - Round 7 experiment tracking document.
- `.okf/experiments/exp-0153-gemma4-performance-round8-2026-09-06.md`: `PENDING` - Round 8 experiment tracking document.

## Complete when

- All checklist items below are executed and checked.
- All unit and integration test suites pass (`ninja -f ninja.build test`), including 79 Python unit tests and the full canonical OCaml suite.
- `git diff --check` passes with no whitespace errors.
- Property-based Monoid laws verified over 10,000 synthetic expression trees (structural identity/associativity + denotational equality at 5 indices × 5 probes each).
- An immutable experiment receipt is recorded in `.okf/experiments/` and the research register `.okf/tracking.md` is updated.
- All changes committed on `feat/unified-optimizing-compiler`.

---

## Checklist

### [x] G4P8-01: Affine Coordinate Category (𝐂𝐨𝐨𝐫𝐝 / Index Monoid) in `lib/coordinate_map.ml`
- **REPO**: `llmopt`
- **WHERE**: `lib/coordinate_map.ml`, `lib/coordinate_map.mli`
- **IMPORTANT FILES**:
  - `lib/coordinate_map.ml`: `MODIFY` - Implement coordinate morphisms `Identity` / `Permute [p1; ...; pn]` with `compose`, `simplify`, and `emit_msl_index` so movement ops allocate zero bytes and generate zero dispatches.
  - `test/test_epilogue_monoid.ml`: `MODIFY` - Assert Coordinate Category axioms (identity, composition, involution) and exact index-expression emission (e.g. transpose `[1;0]` over `[4;4]` emits `((col % 64u) * 32u + (col / 64u))`).
- **IMPORTANT SYMBOLS**: `Coordinate_map.Identity`, `Coordinate_map.Permute`, `Coordinate_map.simplify`, `Coordinate_map.emit_msl_index`.
- **WHY**: 328 movement nodes and 2,797 phantom barrier-waits today cost host dispatch + barrier overhead per token; as pure coordinate morphisms they cost nothing.
- **FIX**: Eliminate movement nodes by construction through categorical composition $T_2 \circ T_1$.
- **QUALITY**: Keep morphisms closed under composition and idempotent under `simplify`.
- **DO NOT**: Allocate buffers or generate kernels for coordinate mappings.
- **VERIFY**: `ninja -f ninja.build test` passes.
- **DONE WHEN**: Coordinate axioms PASS line green and transpose emission string verified.
- **ESCALATE IF**: Composition of >2 movement ops diverges from a single closed-form mapping.

### [x] G4P8-02: Pointwise Free Applicative Functor with idempotent `simplify` in `lib/applicative_expr.ml`
- **REPO**: `llmopt`
- **WHERE**: `lib/applicative_expr.ml`, `lib/applicative_expr.mli`
- **IMPORTANT FILES**:
  - `lib/applicative_expr.ml`: `MODIFY` - Implement the `F(A) = Coord -> Scalar` AST with `pure`/`fmap`/`lift2`/`lift3`, `subst`, and a canonical `simplify`; fold `cast<T,T>` and non-lossy roundtrip chains to identity (functor identity law, `fmap id = id`) and fold constant unop/binop arithmetic.
  - `test/test_epilogue_monoid.ml`: `MODIFY` - Assert cast cancellation and constant folding laws.
- **IMPORTANT SYMBOLS**: `Applicative_expr.simplify`, `Applicative_expr.fmap`, `Applicative_expr.required_values`, `Applicative_expr.emit_msl_expr`.
- **WHY**: 400 redundant identity casts and non-canonical pointwise chains reach the Metal emitters today.
- **FIX**: Cancel same-dtype `Cast` at construction/simplification; guarantee `simplify` is idempotent so `Monoid.equal` normal-form comparisons are stable.
- **QUALITY**: `simplify` must be bottom-up idempotent — including re-cancelling `Unop (Cast t, Read)` when a child *becomes* a same-dtype `Read` after simplification.
- **DO NOT**: Change emission semantics of surviving nodes.
- **VERIFY**: `ninja -f ninja.build test` passes.
- **DONE WHEN**: Applicative Functor laws PASS line green and the 10,000-tree property suite completes without structural mismatch.
- **ESCALATE IF**: A simplifier rewrite is not IEEE-exact on finite operands (see G4P8-05 constant-domain note).

### [x] G4P8-03: Contraction Epilogue Monoid (ℰ, ⋄, ε) in `lib/epilogue.ml` + canonicalization pass
- **REPO**: `llmopt`
- **WHERE**: `lib/epilogue.ml`, `lib/epilogue.mli`, `lib/pass_algebraic_canonicalization.ml`, `lib/passes.ml`, `lib/pass_co_schedule.ml`, `lib/serving_schedule.ml`
- **IMPORTANT FILES**:
  - `lib/epilogue.ml`: `MODIFY` - Add `Expr` variant and `Monoid` module with `empty = Slot Acc`, `append p c = simplify (subst (fun Acc -> p) c)`, generic `compose`, structural `equal`, and MSL write-back emission for the Gemma chain `silu ⋄ bias ⋄ residual ⋄ scale`.
  - `test/test_epilogue_monoid.ml`: `MODIFY` - Assert strict monoid laws (left/right identity, associativity) and the full Gemma composite emitted MSL.
- **IMPORTANT SYMBOLS**: `Epilogue.Monoid.empty`, `Epilogue.Monoid.append`, `Epilogue.Monoid.equal`, `Epilogue.inputs`, `Epilogue.emit_msl_writeback`.
- **WHY**: 385 standalone norm/add/scale dispatches after contraction run as separate kernels; folding them into the accumulator register store removes them.
- **FIX**: Represent downstream pointwise chains as monoid elements over `Slot_acc` and fold them into `out[j] = emit_msl(E_final, acc)`.
- **QUALITY**: Substitution-based `mappend` must be associative and identity-respecting by construction; pass wiring replaces ad-hoc matchers.
- **DO NOT**: Emit a kernel per epilogue element.
- **VERIFY**: `ninja -f ninja.build test` passes.
- **DONE WHEN**: Strict monoid laws PASS line green; Gemma chain MSL fold string verified.
- **ESCALATE IF**: A fused chain needs a value not reachable from the contraction's declared buffers.

### [x] G4P8-04: Dynamic `Epilogue.Expr` buffer declaration/binding in parametric Metal emitters
- **REPO**: `llmopt`
- **WHERE**: `lib/metal.ml`, `lib/metal_runtime.ml`
- **IMPORTANT FILES**:
  - `lib/metal.ml`: `MODIFY` - Add shared helper `epilogue_expr_buffer_decls ~start_index epilogue` that maps `Epilogue.inputs (Expr e)` (de-duplicated `required_values`) to `device const half* buf_<Value_id> [[buffer(slot)]]` (f16 → `half`, else `float`); route `Epilogue.Expr _` through it in `emit_parametric_w4a16_linear` and `emit_parametric_q4_k_linear` at `~start_index:6` and in `emit_simdgroup_matrix_gemm` at `~start_index:4`.
  - `lib/metal_runtime.ml`: `MODIFY` (reserved) - `epilogue_mode` enum `0/1/2` in `Parameters.u32s [m;n;k;epilogue_mode]` for `_add`-suffixed kernels.
  - `test/test_epilogue_monoid.ml`: `MODIFY` - Assert every `buf_<Value_id>` referenced by emitted write-back MSL is declared as a bound buffer with the right storage type in all three emitters.
- **IMPORTANT SYMBOLS**: `Metal.emit_parametric_w4a16_linear`, `Metal.emit_parametric_q4_k_linear`, `Metal.emit_simdgroup_matrix_gemm`, `epilogue_expr_buffer_decls`, `Epilogue.inputs`.
- **WHY**: Emitted write-back expressions reference `buf_<Value_id>` but legacy emitters only declared hard-coded legacy buffers (bias/up/residual), so an `Expr` epilogue would fail to bind its inputs.
- **FIX**: Declare one dynamic buffer per required epilogue input value, contiguous from the emitter's legacy start index.
- **QUALITY**: Buffer ordering matches `Epilogue.inputs` so declaration list and expression names always agree.
- **DO NOT**: Hand-edit the epilogue AST inside the emitters.
- **VERIFY**: `ninja -f ninja.build test` passes; `test_epilogue_monoid.cmx` depends on `metal.cmi` in `ninja.build`.
- **DONE WHEN**: `[PASS] Expr epilogues bind dynamic buf_<Value_id> buffers in parametric emitters` green.
- **ESCALATE IF**: A kernel's slot range collides with runtime-bound legacy buffers.

### [x] G4P8-05: Property-based Monoid laws over 10,000 synthetic expression trees
- **REPO**: `llmopt`
- **WHERE**: `test/test_epilogue_monoid.ml`
- **IMPORTANT FILES**:
  - `test/test_epilogue_monoid.ml`: `MODIFY` - Deterministic generator (`Random.init 0x0CC1E`) building trees from `Slot Acc` / `Const` / f16 `Read` atoms, depth-capped unops/binops/triops; 10,000 trials each checking left identity, right identity, and associativity both structurally (`Monoid.equal`) and denotationally (IEEE eval at 5 indices × 5 probes with `close` tolerance and nan-equality).
- **IMPORTANT SYMBOLS**: `Epilogue.Monoid.append`, `Epilogue.Monoid.empty`, `Epilogue.Monoid.equal`.
- **WHY**: ADR §Verification requires identity + associativity property tests over 10,000 synthetic trees to pin the monoid laws.
- **FIX**: Run both structural and denotational checks per trial; use an IEEE-sound constant domain (strictly positive, no `0.0`) so algebraic `Const` folds are exact against the raw IEEE evaluation even when read arithmetic drives subtrees to `nan`/`inf`.
- **QUALITY**: Deterministic seed so any regression reproduces at the same trial.
- **DO NOT**: Weaken `close` to swallow genuine divergences.
- **VERIFY**: `ninja -f ninja.build test` passes.
- **DONE WHEN**: `[PASS] Property-based Monoid laws over 10000 synthetic expression trees (identity + associativity, structural and denotational)` green.
- **ESCALATE IF**: A structural counterexample appears — first re-check `simplify` idempotence before questioning the monoid itself.

### [x] G4P8-06: Record Round 8 evidence and OKF tracking
- **REPO**: `llmopt`
- **WHERE**: `LOOP-gemma4-performance-round8.md`, `.okf/experiments/exp-0153-gemma4-performance-round8-2026-09-06.md`, `.okf/experiments/index.md`, `.okf/tracking.md`
- **IMPORTANT FILES**:
  - `LOOP-gemma4-performance-round8.md`: `CREATE` - This round document (all items checked).
  - `.okf/experiments/exp-0153-gemma4-performance-round8-2026-09-06.md`: `CREATE` - Experiment tracking document with PASS evidence.
  - `.okf/experiments/index.md`: `MODIFY` - Register exp-0153.
  - `.okf/tracking.md`: `MODIFY` - Advance the top slice row from "proposed architecture & formalized in ADR" to implemented-with-evidence.
- **VERIFY**: `ninja -f ninja.build test` passes (79 Python unit tests + all canonical OCaml suites), `git diff --check` clean, working tree committed on `feat/unified-optimizing-compiler`.
- **DONE WHEN**: Evidence recorded and all changes committed in one slice commit.
