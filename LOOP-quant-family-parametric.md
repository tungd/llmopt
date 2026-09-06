# Quant-Family Parametric Emission Loop (G4P11)

## Goal

Replace the remaining **text-splicing codegen** — `Metal.quant_linear_add_source` (the `_add` rename/addition replacement list), `Metal.q4_0_x8_digest_rewrite` (kernel-text splice on magic string markers, reviewed as "Kernel Zoo 2.0"), and the legacy static quant family strings kept alive in `Metal` (`q4_0_source`, `q4_k_source`, …) — with **one parametric emission architecture**: every quantized-linear kernel (every quant family × tile shape × writeback policy) is printed by `Metal_block_codegen` from schema + tile-topology data, with the **writeback tail a parameter** (`epilogue : Epilogue.t`), so any (quant, tile, epilogue) lowers without new per-shape code paths and without per-epilogue kernel-text surgery.

Hard invariants:
- **Flat decode emission stays byte-identical.** The plain kernels printed today (`llmopt_q4_0_linear_f16*`, `_x4`, `_x8`, `_m2/_m4/_m16/_m32`, `_add` variants, and the Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/IQ4_XS families) must not change a single byte until a byte-identical parametric substitute is in place (legacy pins in test.ml §12/§13 and the epilogue/chain-absorption suites guard this).
- No new bespoke monolithic kernels; no new per-tactic stringly dispatch; no silent compute loss (any combination the emitters cannot yet serve must fail loudly at `lower` with an exact message).
- Gated linear regions keep routing to `Gated_linear`; flat `Residual_add` keeps the legacy `_add`/flat path until the parametric layer replaces it byte-identically.

## Verified architecture (read this before coding — all line numbers at HEAD `049b0c7`)

- `lib/metal_block_codegen.ml` **is already the parametric emitter** for flat families: schema-driven per (quant, tile) printers (`emit_linear_pair`, `emit_superblock_linear_pair`, Q4_0 `_x4/_x8/_m16/_m32`(+`_add`)/`_gated_x8`/`_qkv_x8`/`_qkv_m32`, plus the literal block at :1761-2044 for `f16/_m2/_m4`(+`_add`)), assembled by `q4_0_linear` (:1736) / `linear_kernels` (:2069) / `qkv_kernels` (:2098). Every emitter hardcodes its writeback tail; **`epilogue` is the missing parameter**.
- `lib/metal.ml` `quant_source` (:6957) = `Metal_block_codegen.linear_kernels` output ^ legacy static source with only `llmopt_<q>_linear_f16` and `_m2` duplicates removed (:6968-6974). Other legacy kernels still shadow/supplement the generator.
- `Metal.q4_0_x8_digest_rewrite` (:7083-7130) slices the **parametric** x8 kernel text between `"    uint gid [[thread_position_in_grid]],"` and `") {"`, re-emits the header with `Epilogue.msl_buffer_decls ~start_index:2` (output at `2 + p`, params at `3 + p`, p = `Epilogue.inputs` count), then splices `q4_0_x8_digest_store` (:7069) before the tail marker `"  if (lane == 0u && base_col < params.n) {"`. `emit_q4_0_x8_epilogue_kernel` (:7132) extracts that kernel from the family source. This is the exact mechanism the review quoted; it only exists because the x8 emitter has no epilogue parameter.
- `Metal.lower` digest admission (:10966-10984): matches `(Q4_0, suffix "_x8")` and errors otherwise — the narrowness the review called out. `quant_linear_add_source` (:6976-7032) is a sibling text-patcher producing `_add` kernels by `replace_all` on ~20 store-shape patterns.
- Parametric-writeback proof the architecture already has: `emit_parametric_w4a16_linear` / `emit_parametric_q4_k_linear` / `emit_simdgroup_matrix_gemm` (:11117-11263) thread `~epilogue` through `Epilogue.emit_msl_writeback` + `msl_buffer_decls` at expression level, no text surgery. (Both parametric w4a16/q4_k kernels carry pre-existing indexing bugs recorded in exp-0155 — fix or quarantine before any device use.)
- Runtime: `Metal_runtime` assignment dispatch binds digest kernels positionally (input, weight, epilogue buffers, output, params); digest names come from `Epilogue.algebraic_kernel_name ~base`. Runtime admission errors loudly for non-Q4_0-x8 digest assignments.

## Executor slices (land in order; each commits green: `ninja -f ninja.build ocaml-test`, `ninja -f ninja.build test`, `git diff --check`)

1. **G4P11-01 — [DONE 049b0c7]** Fix `Pass_algebraic_canonicalization` to rewrite op-embedded values (`Ir.Op.map_values`/`Pointwise`/`Argument`), ending the real-bundle crash `node N reads value M with no producer`; regression test asserts `Compute_plan.of_graph` stays valid.
2. **G4P11-02 — writeback-as-parameter in `Metal_block_codegen` (Q4_0 family first)**. Give each Q4_0 emitter an `epilogue : Epilogue.t` parameter (default: current flat tail, byte-identical) and split each emitter into *topology data + body printer + tail printer*. Add `emit_q4_0_x8_digest`-equivalent printing structurally. Then delete `Metal.q4_0_x8_digest_store/rewrite/emit_q4_0_x8_epilogue_kernel` and widen the `lower` digest admission to the whole Q4_0 family (`_x4/_x8/_m16/_m32`, `split_k:1`). Gate: pre-change vs post-change digest kernel text equal (capture golden strings in a unit test before deleting the splice); flat family text unchanged; digest admission tests extended beyond `_x8`.
3. **G4P11-03 — retire `quant_linear_add_source`**. `_add` becomes `epilogue = Residual_add`-shaped emission inside the Q4_0 parametric emitters (or a declared `Add` writeback mode) rather than text rename; byte-identical `_add` kernels; delete the replacement list.
4. **G4P11-04 — port remaining families**. Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/IQ4_XS tile kernels still living only in the legacy static strings (see `legacy_quant_source` map and what `quant_source` removes) move into `Metal_block_codegen` per (quant, tile) with the same tail parameter. Legacy strings deleted only when no emitted kernel references them.
5. **G4P11-05 — runtime + tests** are per-slice companions, not a separate phase: every widening lands with runtime admission + `test.ml`/`test_epilogue_monoid.ml`/`Fold_catalog` pins for the newly servable (quant, tile) digest shapes and a loud-`Error` test for each still-unservable combination.
6. **G4P11-06 — docs/handoff.** Update `LOOP-gemma4-performance-round10.md` + new exp file + `.okf/decisions/parametric-epilogues-and-hardware-tiling.md` cross-refs; OKF validate; user-run device rebuild/bench handoff.

## Out of scope

- Device/bundle builds and benchmarks (user-run, standing constraint).
- Fixing the recorded W4A16/Q4_K parametric emitter indexing bugs inside this loop unless a slice needs those kernels on a real path (then fix, don't quarantine).
- Any behavior change to flat decode, gated routing, or the legacy `_add` path before a byte-identical parametric substitute lands.

## Relevant files

- `lib/metal_block_codegen.ml` (+ `.mli`): the parametric emitter to extend.
- `lib/metal.ml`: legacy sources, `quant_source`, `quant_linear_add_source`, digest emitters (:7069-7153), `lower` (:10966+).
- `lib/epilogue.ml/.mli`: `emit_msl_writeback`, `msl_buffer_decls ~start_index`, `canonicalize`, `inputs`, `algebraic_kernel_name` — the writeback contract every emitter tail must use.
- `lib/metal_runtime.ml`: digest dispatch/admission.
- `lib/kernel_ir.ml`: `Fused_tactic` admission + digest names.
- `test/test.ml`, `test/test_epilogue_monoid.ml`: byte-identity + admission pins (add golden-capture test in G4P11-02 before deleting the splice).
- `ninja.build`: add `epilogue.cmx` dep to `metal_block_codegen.cmx` edge when it first references `Epilogue`.

## Checkpoint state

G4P11-01 committed (`049b0c7`, both gates green). G4P11-02 is the next executor slice; start by capturing golden digest-kernel text from the current splice path into a unit test, then introduce the `~epilogue` parameter on the Q4_0 x8 emitter and route `Metal.lower` through it.
