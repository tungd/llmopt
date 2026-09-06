# Gemma 4 Performance & Throughput Parity Loop - Round 10

## Goal

Retire the Round 9 bespoke W4A16 route seam (review-rejected as the "Kernel Zoo Trap": hardcoded `m = 1` / `Groupwise_packed {4,64}` / flat-residual-add policy, a duplicated `W4A16EpiParams` emitter family, a fixed 5-buffer runtime `Some` branch, and zero decode-visible work) and replace it with an **integrated epilogue generalization**: the Round 8 Epilogue Monoid absorbs downstream pointwise subgraphs algebraically into contraction writebacks across ALL linear contractions, carried end to end through one dynamic buffer-declaration contract (`Epilogue.msl_buffer_decls` over `Epilogue.inputs`), dispatchable via digest-keyed kernel names, and admitted by `Kernel_ir` selection and formed by `constructive_lowering` region matching — with no silent compute loss at any seam.

The integrated change lands as three layers that must stay in lockstep (selection widening is only safe after emission + runtime coverage make every selected region dispatchable):

1. **Mechanism slice (landed G4P10-01..03)**: retire the Round 9 seam and its duplicate emitter family; generalize region folding in `Kernel_ir.Fused_tactic` over the `Epilogue` monoid; make `Epilogue.msl_buffer_decls` the single buffer-declaration contract for every variant and carry the full epilogue through runtime assignment dispatch with a fail-stop `Error` for anything the legacy families cannot yet serve.
2. **Decode-visible service (G4P10-04, pending)**: `Metal_block_codegen` decode emitters and `Metal.lower` region components emit digest-keyed kernels whose writebacks honor arbitrary `Epilogue.t`, so real Q4_0 decode graphs change; then un-gate `Kernel_ir` selection to every fully-folded epilogue and widen the `constructive_lowering` matcher so silu/mul/add/bias consumer chains form single regions.
3. **User-run handoff (G4P10-05, pending)**: bundle rebuild + device benchmark commands and receipt template (text only — the agent never runs device work).

ADR targets for the real Gemma 4 decode bundle (unmeasured by this mechanism slice; expected to move only after G4P10-04): schedule **4,617 → <500 commands**, **2,797 → <150 barriers**, ~762 dispatches incl ~385 standalone elementwise → toward **~140 dispatches/token**, latency toward **~35 ms**, and bit-exact 128-token parity with golden tokens `100, 45518, 107, 14947, 135383, 568, 11124, 568, 147505, 236789, ...`.

## Checkpoint state (this document)

G4P10-01..03 are committed and green on `feat/unified-optimizing-compiler` (HEAD `43d4b6e`); this LOOP file is the mid-round checkpoint the user asked for before resuming code. The unchecked G4P10-04/05 items below are executor-ready continuation specs.

## Out of scope

- Running the Gemma 4 128-token device benchmark or any `llmopt-pipeline` bundle rebuild in this slice — device work is user-run and deferred to the G4P10-05 handoff.
- Any new bespoke monolithic kernels, new `Epilogue` variants per shape/quant, or per-tactic stringly dispatch.
- Retiring legacy static Q4_0/Q4_K/W4A16 kernels until the region/parametric layer produces byte-compatible substitutes (test.ml §12/§13 pins keep them).

## Relevant files

- `lib/epilogue.ml` / `lib/epilogue.mli`: the monoid core (constructors, `compose`, `canonicalize`, `mul ~factor`, `of_expr`/`to_expr`, `map_reads`, `equal`), `inputs`, `msl_buffer_decls ~start_index` (single source of buffer-declaration layout for EVERY variant: role-named `bias`/`up`/`residual` for typed variants in `Epilogue.inputs` order, `buf_<Value_id>` for `Expr`; dtype-derived `half`/`float`), `algebraic_kernel_name ~base` (digest8 over canonical writeback + required-tensor dtypes), and `Monoid`.
- `lib/kernel_ir.ml` / `lib/kernel_ir.mli`: `Fused_tactic.Linear_epilogue { linear; epilogue; output }`, `type chain = Folded of Epilogue.t * int | External of Ir.Value.t | Nonfoldable`, `chain_of`, `linear_epilogue` (select admission still gated to flat `Residual_add`, depth + 1 = region bindings), exported ungated algebra core `fold_linear_epilogue`; select order `gated_delta → short_conv_silu → linear_epilogue → gated_product → rms_rope → rms_normalize`.
- `lib/metal.ml`: `region_tactics ~target`, `quant_linear_add_source` (textual `_add` rename of static block sources), the three parametric emitters (`emit_parametric_w4a16_linear ~start_index:6`, `emit_parametric_q4_k_linear ~start_index:6`, `emit_simdgroup_matrix_gemm ~start_index:4`) all consolidated onto `Epilogue.msl_buffer_decls`, `lower` (~:10722-10915) component assembly with `quant_add_tactics`/`quant_add_source`.
- `lib/metal_runtime.ml`: `assignment_execution` type (`Assignment_linear_epilogue { linear; epilogue; output }`), `select_quant_linear_kernel ~suffix`, `dispatch_quant_linear` (legacy flat `Residual_add` via `epilogue_mode:2`; W4A16 g64/g64_m4 unchanged; experimental `LLMOPT_EXPERIMENTAL_*` branches untouched), `encode_schedule` arms.
- `lib/metal_block_codegen.ml`: schema-driven decode emitters (Q4_0 family ~:1736-2067, base `llmopt_q4_0_linear_f16` + `_m2`/`x8 ~split_k:1`/`m16`/`m16_add`/`m32`/`m32_add`/gated x8/literal `m4_add`), `linear_kernels` (~:2069-2079).
- `lib/constructive_lowering.ml`: region matcher layer (`linear_add_match` ~:1064 fuses exactly ONE Linear + ONE Add, `executable_region` ~:1625) — unchanged this slice, widened in G4P10-04.
- `test/test.ml`: ONE giant top-level `let () =`; new suites must be TOPLEVEL before it and invoked in the tail before `print_endline "llmopt canonical W4A16/KVQ8 tests passed"` (~line 7297). `module Fold_catalog` ~line 115 (run in main tail) pins: flat linear+residual folds to `Epilogue.Residual_add`, bare linear folds `Some Epilogue.none`, fanout → `None`; select-assent pin at ~:275-290; select-is-`None` pins at ~:312 (Silu), ~:375 (Sigmoid), ~:409 (stacked residual adds) — these flip to `Some` at G4P10-04 when selection is ungated.
- `ninja.build`: kernel_ir cmi/cmx edges carry `epilogue.cmi`/`epilogue.cmx`; link order `coordinate_map applicative_expr epilogue` immediately before `kernel_ir` (round-10 compile/link wiring).

## Supporting documents

- `.okf/decisions/applicative-ir-coordinate-category-and-epilogue-monoid.md`: `EXISTING` - Round 8 ADR; this round's mechanism is the decision's Tier-3 carry generalized from the flat residual-add contract to arbitrary monoid folds.
- `.okf/experiments/exp-0153-gemma4-performance-round8-2026-09-06.md` and `exp-0154-gemma4-performance-round9-2026-09-07.md`: `EXISTING` - Round 8 foundation and the Round 9 seam record (superseded by G4P10-01; exp-0154 stays as the historical record of the retired seam).
- `.okf/experiments/exp-0155-gemma4-performance-round10-2026-09-08.md`: `CREATED` - Round 10 experiment tracking document.

## Complete when

- All checklist items below are executed and checked.
- `ninja -f ninja.build ocaml-test` and `ninja -f ninja.build test` pass (79 Python unit tests OK and every canonical OCaml suite PASS line green — currently 15 `[PASS]` lines ending `llmopt canonical W4A16/KVQ8 tests passed`).
- `git diff --check` passes with no whitespace errors.
- No silent compute loss: every region selected by `Kernel_ir` is dispatchable or fails loudly (guaranteed by the G4P10-03 fail-stop `Error`; the `encode_schedule` `None` branch means only "no region").
- Legacy static Q4_0/Q4_K/W4A16 kernels and test.ml §12/§13 legacy pins are intact; source contains no `W4A16EpiParams`, no `llmopt_w4a16_linear_f16_g64_epi`, no `w4a16_algebraic_route`, no `emit_w4a16_algebraic_kernels`.
- An immutable experiment receipt is recorded in `.okf/experiments/` and the research register `.okf/tracking.md` is advanced.
- All changes committed on `feat/unified-optimizing-compiler` with `G4P10-NN:` subjects.

---

## Checklist

### [x] G4P10-01: Retire the bespoke W4A16 route seam and duplicate emitter family
- **REPO**: `llmopt`
- **WHERE**: `lib/epilogue.ml`, `lib/epilogue.mli`, `lib/metal.ml`, `lib/metal.mli`, `lib/metal_runtime.ml`, `ninja.build`, `test/test.ml`
- **COMMIT**: `c07ba78` (`G4P10-01`)
- **IMPORTANT SYMBOLS RETIRED**: `Epilogue.w4a16_algebraic_route`, `Epilogue.w4a16_algebraic_base`, `Epilogue.algebraic_residual_add`, `Metal.emit_w4a16_algebraic_kernels`, `W4A16EpiParams`, the routed branch in `Metal.lower`, and the runtime `Some` branch.
- **KEPT**: `Epilogue.Monoid`, `compose`, `canonicalize`, `msl_buffer_decls`, `algebraic_kernel_name`, `inputs`, `equal`, `map_reads`, `of_expr`/`to_expr`, all legacy constructors, and the three parametric emitters.
- **FIX**: Round 9 duplicated an emitter family behind a shape-hardcoded route; restore one emitter family + legacy dispatch byte-identical for flat regions and rewrite test.ml §12 to pin the legacy path.
- **VERIFY**: `ninja -f ninja.build ocaml-test` then `ninja -f ninja.build test`, `git diff --check`.
- **DONE WHEN**: Commit present; source contains no `W4A16EpiParams`, no `llmopt_w4a16_linear_f16_g64_epi`; suites green.

### [x] G4P10-02: Generalize linear region epilogue folding over the Epilogue monoid
- **REPO**: `llmopt`
- **WHERE**: `lib/kernel_ir.ml`, `lib/kernel_ir.mli`, `lib/epilogue.ml`, `lib/epilogue.mli`, `ninja.build`, `test/test.ml`
- **COMMIT**: `6f07846` (`G4P10-02`)
- **IMPORTANT SYMBOLS**: `Fused_tactic.Linear_epilogue { linear; epilogue; output }` (was `Linear_additive`), `chain = Folded of Epilogue.t * int | External of Ir.Value.t | Nonfoldable`, `chain_of`, `linear_epilogue` (select-time; admission-gated to flat `Residual_add` with `depth + 1 = List.length region.bindings` so legacy dispatch stays byte-identical), exported `fold_linear_epilogue` (ungated algebra core: bare linear → `Some Epilogue.none`, full chain otherwise), `Epilogue.mul ~factor`.
- **WHY**: The Round 8 monoid must fold arbitrary downstream pointwise chains into `Epilogue.t` (`E_new = E_curr ⋄ E_consumer`) at region-formation time; Round 9's one-off route admitted only flat residual-add.
- **FIX**: Generalize the region epilogue type and fold algebra in `kernel_ir`; keep select admission narrow until emission + runtime can serve richer epilogues.
- **VERIFY**: Fold_catalog (test.ml top-level module ~:115, invoked in the main tail): flat linear+residual → `Epilogue.Residual_add`; bare linear → `Some Epilogue.none`; fanout → `None`; select-assent and select-is-`None` pins around test.ml :275-409. Full suites + `git diff --check`.
- **DONE WHEN**: Commit present and suites green.

### [x] G4P10-03: Unified dynamic epilogue buffer contract + fail-stop dispatch
- **REPO**: `llmopt`
- **WHERE**: `lib/epilogue.ml`, `lib/epilogue.mli`, `lib/metal.ml`, `lib/metal_runtime.ml`
- **COMMIT**: `43d4b6e` (`G4P10-03`)
- **IMPORTANT SYMBOLS**: `Epilogue.msl_buffer_decls ~start_index` generalized to every variant (typed variants bind role-named `bias`/`up`/`residual` via per-variant `[(value, name)]` lists in `Epilogue.inputs` order; `Expr` binds `buf_<Value_id>`; decls dtype-derived `Float16 → half` else `float`; layout agrees with `emit_msl_writeback`'s reads). The three parametric emitters all consume it (bespoke `epilogue_expr_buffer_decls` and per-variant hardcoded matches deleted). Runtime: `Assignment_linear_add` → `Assignment_linear_epilogue { linear; epilogue; output }`; `assignment_execution` returns `Error` (never silent `None`) for any epilogue the legacy kernel families cannot serve, with the message naming the region and dtype via `Ir.Dtype.quant_to_string`.
- **WHY**: Round 9's runtime bound a fixed 5-buffer list; arbitrary `Epilogue.inputs` need one declaration contract shared by emission and dynamic runtime binding.
- **FIX**: Make `msl_buffer_decls` total over variants; consolidate emitters; carry the epilogue through dispatch with fail-stop safety (selection widening is safe-by-construction only because a selected-but-unserved region now errors instead of degrading).
- **VERIFY**: Legacy flat `Residual_add` dispatch byte-identical (Block_quantized `dispatch_quant_linear ~epilogue_mode:2`; W4A16 g64/g64_m4 unchanged; legacy `has_bias`/`bias_row_stride` writeback text present); full suites + `git diff --check`.
- **DONE WHEN**: Commit present; suites green; working tree clean at `43d4b6e`.

### [x] Interim docs checkpoint (evidence for G4P10-01..03)
- **REPO**: `llmopt`
- **WHERE**: `LOOP-gemma4-performance-round10.md`, `.okf/experiments/exp-0155-gemma4-performance-round10-2026-09-08.md`, `.okf/experiments/index.md`, `.okf/tracking.md`, `.okf/log.md`
- **IMPORTANT FILES**:
  - `LOOP-gemma4-performance-round10.md`: `CREATE` - This round document (G4P10-01..03 and this checkpoint checked; G4P10-04/G4P10-05 open with continuation specs).
  - `.okf/experiments/exp-0155-...md`: `CREATE` - Round 10 experiment tracking document with PASS evidence and the supersede note for exp-0154.
  - `.okf/experiments/index.md`: `MODIFY` - Register exp-0155.
  - `.okf/tracking.md`: `MODIFY` - Advance the top slice row to the Round 10 mechanism slice; correct the Round 9 row's state to superseded.
  - `.okf/log.md`: `MODIFY` - Append the dated update entry.
- **VERIFY**: `okf_validate.py` reports no new issues beyond the 3 pre-existing non-blocking `status: accepted` warnings; `ninja -f ninja.build test` passes; `git diff --check` clean.
- **DONE WHEN**: Evidence recorded and all changes committed in one docs slice commit on `feat/unified-optimizing-compiler`.

### [x] G4P10-04b: Emission + runtime coverage for arbitrary `Epilogue.t` on the real decode path
- **REPO**: `llmopt`
- **WHERE**: `lib/metal_block_codegen.ml`, `lib/metal.ml`, `lib/metal_runtime.ml`, `test/test.ml`, `ninja.build`
- **PROGRESS**:
  - **DONE (Slice 1, commit `998906e`)**: digest-keyed Q4_0 x8 epilogue emission. Provenance REFUTED — no `x8_add`/`x4_add` twins can exist because the static q4_0 source has no plain x8/x4 body and `quant_linear_add_source` only renames STATIC kernels (it early-returns `""` for `_m4`/`_m16`/`_m32` names). Implemented `Metal.emit_q4_0_x8_epilogue_kernel ~name ~epilogue` (lib/metal.ml ~:7132, exported in metal.mli): derives the real `llmopt_q4_0_linear_f16_x8` kernel text from `quant_source`, canonicalizes the epilogue, rebuilds the header to the digest positional layout `input [[buffer(0)]]`, `weight [[buffer(1)]]`, `Epilogue.msl_buffer_decls ~start_index:2` (canonical `buf_0..{p-1}`), `output [[buffer(2+p)]]`, `params [[buffer(3+p)]]`, rewrites the single-store tail into eight column-guarded stores (`accumulator0..7`, guarded `base_col + j < params.n` for j ≥ 1) applying `emit_msl_writeback`, drops the legacy `params.has_bias` store block, and renames the kernel to the digest name. The runtime bind list `[input; weight] @ Epilogue.inputs epilogue` with output appended last places params exactly at index `3+p` — no stub change. Plain kernel text is untouched (legacy flat decode byte-identical).
  - Tests added in `test/test_epilogue_monoid.ml` (q4_0 x8 section, runs in `ocaml-test`): digest rename, canonicalized `buf_0..2` decls at buffer(2..4) for a 3-input chain, `(float)buf_0[base_col]` readback, guarded column-7 store over `accumulator7`, no `params.has_bias`, deterministic text, f32 read → `float` buffer decl.
  - **DONE (Slice 2, commit `10ef998`)**: emission + runtime wiring. `Metal.lower` region components now serve every `Block_quantized` folded epilogue whose costed tactic is the Q4_0 x8 decode kernel through the digest-keyed emitter (`region_tactics` returns an 8-tuple carrying `quant_digest` triples `(quant, tactic, epilogue)`; digest source folds through `emit_q4_0_x8_epilogue_kernel`, entries register `Operation.Linear`/`Quant Q4_0`/`Float16` with the tactic threadgroup). `Metal_runtime` `Assignment_linear_epilogue` `Block_quantized` arm first resolves a legacy `_add` kernel and keeps that path byte-identical (flat `Residual_add` non-x8), and when none exists re-derives the plain tactic, requires Q4_0 + `_x8` base, computes the digest name from the base, binds the folded epilogue input buffers dynamically, and dispatches with `[m;n;k;0]` params, `(n+7)/8` columns on `linear_f16_grid` — failing loudly for any package/tactic combination that is not servable. test.ml gains a real decode-shape (m=1 n=4096 k=3840 Q4_0) Linear-add lowering suite: plain x8 kernel registered, digest kernel source + ABI entry registered, `buf_0` at buffer(2), output/params at buffers(3)/(4), and no legacy x8 `_add` twin anywhere.
- **IMPORTANT FILES**:
  - `lib/metal.ml`: `DONE` - `region_tactics` returns an 8-tuple with `quant_digest` triples (`quant, tactic, epilogue`); `lower` folds digest source through `emit_q4_0_x8_epilogue_kernel` and registers `Kernel_abi` entries mirroring the `quant_add_tactics`/`quant_add_source` machinery.
  - `lib/metal_runtime.ml`: `DONE` - `Assignment_linear_epilogue` keeps the legacy `_add` path when a twin exists and otherwise re-derives the plain Q4_0 `_x8` tactic, resolves the digest kernel by `Epilogue.algebraic_kernel_name`, binds folded-epilogue buffers dynamically, and fails loudly when no digest/legacy kernel can serve the region.
  - `test/test.ml`: `DONE` - real decode-shape Q4_0 x8 Linear-add lowering suite (plain x8 + digest kernel, digest ABI entry, buffer indices, no legacy x8 `_add` twin); §12/§13 legacy pins green.
- **IMPORTANT SYMBOLS**: `Metal.emit_q4_0_x8_epilogue_kernel`, `Epilogue.msl_buffer_decls`, `Epilogue.inputs`, `Epilogue.algebraic_kernel_name`, `Kernel_abi.Linear_tactic.select_all`, `select_quant_linear_kernel ~suffix`, `assignment_execution` `Assignment_linear_epilogue`.
- **WHY**: Mechanism layers 1-3 land here first: before selection is ungated, the real Q4_0 `Block_quantized` decode graphs (not W4A16) must actually change emitted kernel text, else a bundle rebuild verifies nothing new.
- **FIX**: Provenance settled and both slices landed (`998906e`, `10ef998`); digest emission, `Metal.lower` region components, `Metal_runtime` dispatch with dynamic `Epilogue.inputs` binding, and lowering-service tests are all in place behind the unchanged `Kernel_ir` flat admission gate.
- **QUALITY**: No stringly dispatch; digest name is content-derived over canonical epilogue + dtypes; one buffer-declaration contract for emission and runtime; no `*_for_test` mutators; no `List.nth`/`List.mem` in hot paths.
- **DO NOT**: Un-gate `Kernel_ir` selection in this step, touch the `LLMOPT_EXPERIMENTAL_*` runtime branches, or delete legacy static kernels.
- **VERIFY**: `ninja -f ninja.build ocaml-test` then `ninja -f ninja.build test`, `git diff --check`.
- **DONE WHEN**: Commit(s) present; a folded-epilogue region lowers to a digest-keyed kernel whose writeback honors the `Epilogue.t`, runtime dispatch binds all `Epilogue.inputs` buffers dynamically, and the legacy decode path is byte-identical.
- **ESCALATE IF**: A digest name diverges between lowering and runtime, or `extract_kernel` cannot find a base kernel for a tactic that needs an epilogue twin.

### [ ] G4P10-04c: Un-gate `Kernel_ir` selection to fully-folded epilogues
- **REPO**: `llmopt`
- **WHERE**: `lib/kernel_ir.ml`, `lib/kernel_ir.mli`, `test/test.ml`
- **IMPORTANT SYMBOLS**: `Fused_tactic.linear_epilogue` (~kernel_ir.ml :2320-2362), `chain_of`, `fold_linear_epilogue` (:2364-2387), `Fused_tactic.select` order (:2441-2457).
- **FIX**: Widen admission from `Epilogue.Residual_add`-only to ANY `Folded (epilogue, depth)` with `depth + 1 = List.length region.bindings` (full-chain coverage). Flat `Residual_add` regions must stay byte-identical on the legacy `_add` path.
- **VERIFY**: Flip the select-is-`None` pins at test.ml ~:312 (Silu chain), ~:375 (Sigmoid chain), ~:409 (stacked residual adds) to `Some` with the exact folded `Epilogue.t`; extend Fold_catalog; full suites + `git diff --check`.
- **DONE WHEN**: Commit present; suites green; §12/§13 legacy pins unchanged.
- **ESCALATE IF**: Any selected-but-unserved epilogue remains (the G4P10-03 fail-stop `Error` will surface it in tests before any device run).

### [ ] G4P10-04d: Widen `constructive_lowering` region formation to pointwise consumer chains
- **REPO**: `llmopt`
- **WHERE**: `lib/constructive_lowering.ml`, `test/test.ml`
- **IMPORTANT SYMBOLS**: matcher layer ~:844-1126, `linear_add_match` ~:1064 (currently fuses exactly ONE Linear + ONE Add under an fp16 same-shape numel gate), `executable_region` ~:1625.
- **FIX**: Extend the matcher so silu/mul/add/bias chains consuming a linear output form a single executable region (multi-hop `chain_of` folds happen at `Kernel_ir` selection); the real dispatch-count drop in the decode bundle lands only here.
- **VERIFY**: New chain-absorption catalog test asserting region formation counts and fold shapes; flat regions remain byte-identical; full suites + `git diff --check`.
- **DONE WHEN**: Commit present; suites green.

### [ ] G4P10-05: Hand off bundle rebuild + device benchmark commands and receipt template (text only)
- **REPO**: `llmopt`
- **WHERE**: round-end docs update + handoff message text (no code).
- **IMPORTANT FILES**: `bin/pipeline.ml` (622 lines; output_dir param ~:131-136), `bin/bench_tokens.ml`, historical receipts `.okf/experiments/exp-0145..0154`, `bench/results/`.
- **FIX**: Reconstruct the exact bundle-rebuild and device-benchmark commands the user runs, with a receipt template whose table pins the ADR targets: schedule commands **< 500**, barriers **< 150**, dispatches toward **~140/token**, latency toward **~35 ms**, 128-token golden parity `100, 45518, 107, 14947, 135383, 568, 11124, 568, 147505, 236789, ...`.
- **VERIFY**: The commands are text-only and reproducible from the listed sources; okf validator clean.
- **DONE WHEN**: Handoff message + round-end docs delivered; exp-0155 updated to final, tracking.md/index.md/log.md advanced, suites green, branch committed.
