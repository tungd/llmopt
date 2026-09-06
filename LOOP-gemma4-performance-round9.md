# Gemma 4 Performance & Throughput Parity Loop - Round 9

## Goal

Wire the fused **Linear-add epilogue schedule routing** end to end for Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal, on top of the Round 8 algebraic IR (affine Coordinate Category, Pointwise Applicative Functor, Contraction Epilogue Monoid). A fused W4A16 `Groupwise_packed {bits=4; group_elements=64}` `Linear_additive` region at decode shape (`m = 1`) now:

1. lowers through one deterministic **parametric** kernel family (`llmopt_w4a16_linear_f16_g64_epi_<digest8>`) whose residual add is carried as an `Epilogue.Expr` read (`(acc + ((float)buf_0[col]))`), and
2. dispatches at runtime to the same kernel because `Metal_runtime` recomputes the exact same pure routing key (`Epilogue.w4a16_algebraic_route`) that `Metal.lower` used when emitting it.

Every other geometry and layout is untouched: non-`{4,64}` packed layouts and multi-row (`m ≠ 1`) regions keep today's byte-identical legacy static `llmopt_w4a16_linear_f16_g64*` kernels and the legacy runtime branch.

This round also lands the review hardening of the Round 8 algebraic IR that the end-to-end routing depends on (G4P9-01, committed `46416c3`), followed by the routing implementation (G4P9-02, committed `fde9817`).

Target invariants:
- **Parity Invariant**: Maintain 100% bit-exact token parity on prompt `2,105,9731,...` generating `100, 45518, 107, 14947, 135383,...` (device re-run deferred beyond this round per slice scope).
- **Behavioral Invariant**: Non-routed regions (W4A16 but `m ≠ 1`, or non-`{4,64}` layouts) keep today's byte-identical legacy static `llmopt_w4a16_linear_f16_g64*` kernels and legacy runtime branch.
- **Architectural Invariant**: No bespoke monolithic kernels; every emitted routed kernel is parameterized codegen (`emit_w4a16_algebraic_kernels`) driven by the canonical algebraic epilogue, and the kernel-name digest makes emission and dispatch agree without stringly dispatch.
- **Verification Invariant**: The routing key is exercised at the shared pure function, at `Metal.lower` source/`Program.kernels` ABI registration, and at string-level emitter output in `test/test.ml` section 12; the full `ninja -f ninja.build test` suite (79 Python + all canonical OCaml suites) stays green with `git diff --check` clean.

## Out of scope

- Running the Gemma 4 128-token device benchmark or `llmopt-pipeline` bundle rebuild in this slice (GPU run deferred by decision; round-8 completion notes name schedule routing of fused epilogues as the prerequisite, now satisfied — the device benchmark is the next round's slice).
- Writing handwritten monolithic kernels in `lib/metal.ml`.
- Multi-op silu/residual chains or new graph ops; the routing seam covers the flat W4A16 Linear-add residual contract that the legacy static kernels already implemented with `bias_or_scale[row * bias_row_stride + col]`.
- Runtime dispatch verification on a live Metal device (not unit-testable without a device; pinned instead at the shared route + lowering + ABI + string seams).

## Relevant files

- `lib/epilogue.ml` / `lib/epilogue.mli`: shared schedule-routing key — `algebraic_residual_add`, `canonicalize` (stable tensor-id renumbering by first occurrence), `msl_buffer_decls ~start_index`, `algebraic_kernel_name ~base` (digest over canonical writeback + required tensor dtypes), `w4a16_algebraic_base`, and the single policy `w4a16_algebraic_route ~storage ~m ~residual`.
- `lib/metal.ml`: `emit_w4a16_algebraic_kernels` (parametric `W4A16EpiParams` emitter family), `region_tactics ~target` threading the per-region routed-epilogue list, and `lower` registering each routed kernel as a `Kernel_abi` entry (threadgroup 256).
- `lib/metal_runtime.ml`: `encode_schedule` Linear-add dispatch — recomputes the route; `Some` branch dispatches the parametric kernel with `[input; weight; scale; output; residual]` buffers and `[m; n; k]` params; `None` branch is the untouched legacy dispatch.
- `test/test.ml`: section 12 (round-9 routing assertions).
- `lib/coordinate_map.ml` / `.mli`, `lib/applicative_expr.ml` / `.mli`: review-hardened algebraic IR foundation (G4P9-01).
- `ninja.build`: `metal_runtime.cmx`/link edges extended with the `coordinate_map`/`applicative_expr`/`epilogue` dependencies.

## Supporting documents

- `.okf/decisions/applicative-ir-coordinate-category-and-epilogue-monoid.md`: `EXISTING` - Round 8 ADR; the routing seam is the decision's Tier-3 carry into real fused-region dispatch.
- `.okf/experiments/exp-0153-gemma4-performance-round8-2026-09-06.md`: `EXISTING` - Round 8 experiment tracking document.
- `.okf/experiments/exp-0154-gemma4-performance-round9-2026-09-07.md`: `CREATED` - Round 9 experiment tracking document.

## Complete when

- All checklist items below are executed and checked.
- `ninja -f ninja.build ocaml-test` and `ninja -f ninja.build test` pass (79 Python unit tests OK and every canonical OCaml suite PASS line green).
- `git diff --check` passes with no whitespace errors.
- Routing behavior pinned in `test/test.ml` section 12: admission/decline policy of `w4a16_algebraic_route`, canonical `buf_0` binding, deterministic digest naming surviving structural residual renames, emitted source containing the parametric kernel + `W4A16EpiParams` + `device const half* buf_0 [[buffer(4)]]` + `(acc + ((float)buf_0[col]))`, ABI entry registration, and the multi-row legacy-path negative.
- An immutable experiment receipt is recorded in `.okf/experiments/` and the research register `.okf/tracking.md` is updated.
- All changes committed on `feat/unified-optimizing-compiler`.

---

## Checklist

### [x] G4P9-01: Review-hardening of the Round 8 algebraic IR
- **REPO**: `llmopt`
- **WHERE**: `lib/coordinate_map.ml`, `lib/coordinate_map.mli`, `lib/applicative_expr.ml`, `lib/applicative_expr.mli`, `lib/epilogue.ml`, `lib/epilogue.mli`, `test/test_epilogue_monoid.ml`, `test/test.ml`, `ninja.build`
- **COMMIT**: `46416c3` `refactor(compiler): seal coordinate maps, make algebraic equality/order contracts total and pinned (G4P9-01)`
- **IMPORTANT SYMBOLS**: `Coordinate_map.seal`/validation + corrected `Permute` fold direction, `Applicative_expr.equal`, `Applicative_expr.required_values`, `Epilogue.equal` congruence, dropped dead result values.
- **FIX**: Review findings before routing could depend on canonical equality: seal + validate `Coordinate_map`, make equality/order contracts total, remove a dead computation.
- **VERIFY**: `ninja -f ninja.build ocaml-test` and `ninja -f ninja.build test` pass.
- **DONE WHEN**: Commit `46416c3` present and full suites green.

### [x] G4P9-02: Fused W4A16 Linear-add epilogue schedule routing, end to end
- **REPO**: `llmopt`
- **WHERE**: `lib/epilogue.ml`, `lib/epilogue.mli`, `lib/metal.ml`, `lib/metal.mli`, `lib/metal_runtime.ml`, `ninja.build`, `test/test.ml`
- **COMMIT**: `fde9817` `feat(compiler): route fused linear-add epilogues to parametric Metal kernels (G4P9-02)`
- **IMPORTANT FILES**:
  - `lib/epilogue.ml`: `MODIFY` - Add `algebraic_residual_add`, total idempotent `canonicalize` (re-number tensor ids by stable first-occurrence ordinal over `required_values`), `msl_buffer_decls ~start_index`, `algebraic_kernel_name ~base` (digest over canonical writeback + required tensor dtypes), `w4a16_algebraic_base = "llmopt_w4a16_linear_f16_g64_epi"`, and `w4a16_algebraic_route` (admits `m = 1` `Groupwise_packed {4,64}` only).
  - `lib/metal.ml`: `MODIFY` - `emit_w4a16_algebraic_kernels` emits one `kernel void <name>` per routed epilogue: `input 0` / `packed_weight 1` / `scale 2` / `output 3` / epilogue buffers from 4 (`device const half* buf_0 [[buffer(4)]]`) / `constant W4A16EpiParams& params [[buffer(4 + buffer_count)]]`, body streaming-identical to the legacy g64 kernel minus the bias add, final store `output[row * params.n + col] = half(<writeback>)`. `region_tactics` threads a routed-epilogue list; `lower` appends a component emitting the kernel family and registering `kernel_entry_with_threadgroup ~threadgroup:(256,1,1) ~operation:W4a16_linear ~input_dtype:Float16 ~output_dtype:Float16`.
  - `lib/metal_runtime.ml`: `MODIFY` - `encode_schedule` Linear-add branch recomputes `Epilogue.w4a16_algebraic_route`; `Some epilogue` resolves `Epilogue.algebraic_kernel_name` (same digest as lowering), binds `[input; weight; scale_buffer; output_buffer; additive_buffer]`, parameters `[m; n; k]`, grid `linear_f16_grid (m*n)`; `None` keeps the byte-identical legacy `g64`/`g64_m4` dispatch.
  - `test/test.ml`: `MODIFY` - Section 12 routing suite: route admission (m=1, {4,64}), decline (m=4; group_elements=32; Dense), canonical `buf_0` ordinal, digest naming + structural-rename stability (`Value_id 777` alias), emitted source strings, ABI entry registration, and the multi-row legacy negative.
- **IMPORTANT SYMBOLS**: `Epilogue.w4a16_algebraic_route`, `Epilogue.algebraic_kernel_name`, `Epilogue.canonicalize`, `Epilogue.msl_buffer_decls`, `Metal.emit_w4a16_algebraic_kernels`.
- **WHY**: Round 8 built the epilogue monoid and dynamic writeback buffer binding but did not carry it through real fused-region dispatch; a fused decode Linear-add region either fused to nothing on device or stayed a candidate for yet another bespoke kernel. The route is the seam that turns the algebraic epilogue into a schedule decision shared byte-for-byte between lowering and runtime.
- **FIX**: One pure routing policy recomputed by both sides; deterministic content-digest names; parametric emission reusing the legacy slot convention so the existing runtime binder applies.
- **QUALITY**: `canonicalize` is total and idempotent; structural equality of epilogues ⇒ equal names; runtime dispatch needs no Metal symbols.
- **DO NOT**: Add an `Epilogue` field to `Kernel_ir`, drive parametric emitters whose epilogue buffers follow constant params, or introduce `*_for_test` mutators.
- **VERIFY**: `ninja -f ninja.build ocaml-test` then `ninja -f ninja.build test`, `git diff --check`, `_build/bin/llmopt-test` section 12 PASS lines.
- **DONE WHEN**: `fde9817` present; decode-shaped region lowers to one deterministic parametric kernel and registers its ABI entry; prefill-shaped (m=4) program contains no `llmopt_w4a16_linear_f16_g64_epi` kernel and no `W4A16EpiParams`.
- **ESCALATE IF**: A routed name differs between `Metal.lower` and `Metal_runtime` (digest inputs diverge), or a graph shape change breaks the no-dangling-producer rule (residual must be a graph input, never a fresh dangling value).

### [x] G4P9-03: Record Round 9 evidence and OKF tracking
- **REPO**: `llmopt`
- **WHERE**: `LOOP-gemma4-performance-round9.md`, `.okf/experiments/exp-0154-gemma4-performance-round9-2026-09-07.md`, `.okf/experiments/index.md`, `.okf/tracking.md`
- **IMPORTANT FILES**:
  - `LOOP-gemma4-performance-round9.md`: `CREATE` - This round document (all items checked).
  - `.okf/experiments/exp-0154-gemma4-performance-round9-2026-09-07.md`: `CREATE` - Experiment tracking document with PASS evidence.
  - `.okf/experiments/index.md`: `MODIFY` - Register exp-0154.
  - `.okf/tracking.md`: `MODIFY` - Advance the top slice row from round-8 foundation to fused-epilogue schedule routing implemented with evidence.
- **VERIFY**: `ninja -f ninja.build test` passes (79 Python unit tests + all canonical OCaml suites), `git diff --check` clean, working tree committed on `feat/unified-optimizing-compiler`.
- **DONE WHEN**: Evidence recorded and all changes committed in one slice commit.
