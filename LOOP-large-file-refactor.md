# LOOP: Split the large compiler and runtime files by responsibility

## Intent

Make the largest OCaml compilation units navigable by giving their existing
responsibilities separate owners. For example, changing schedule decoding should
lead to `Serving_schedule_codec`, while changing sequence specialization should
lead to `Serving_schedule_sequence`; both remain reachable through the existing
`Serving_schedule` API. Editing an attention shader should not require traversing
the quantized Linear catalog or runtime dispatch logic.

Use ordinary compilation units, explicit `.mli` boundaries, and module aliases
behind the existing public modules. Move implementations with their data and
helpers, then shorten orchestration functions where they currently contain whole
interpreters. Do not simply rename each monolith to `*_impl.ml`, create numbered
fragments, duplicate representations, or introduce a general plugin framework.
The chosen OCaml mechanism is module composition with shared type identity; this
split does not call for new functors, effects, GADTs, or callback registries.

This is a structural refactor: preserve current compiler decisions, generated
shader bodies, schedule encoding, dispatch ordering, and resource lifetimes.
Do not combine it with quant-kernel retirement, fusion changes, new tactics,
speculative-decoding features, bug fixes unrelated to extraction, or performance
campaigns. Line counts below explain scope; there is no target maximum or
performance threshold. Other LOOPs remain separate work queues.

The user authorized writing this handoff. Product implementation is the next
action for `/loop-execute`, not part of the planning turn. No commit, publication,
model download, or long model benchmark is implied by this document.

## Context

Repository: `/Users/tung/Projects/std23/llmopt`. Final inspection baseline: HEAD
`4cb3125`. During planning, the initial dirty Metal edits were committed in
`c40cc10`, followed by their OKF checkpoint. These include K-family m2_x4
schema-emitter ports, golden files, and tests; preserve that work. At final
inspection only this new LOOP was untracked. Re-read the current VCS state
before execution; do not reset, overwrite, or stage concurrent unrelated work.

Observed sizes and responsibilities at planning time:

| Existing file | Lines | Responsibilities to separate |
| --- | ---: | --- |
| `lib/metal.ml` | 10,484 | Shader catalogs, ABI entries, graph/tactic selection, source assembly |
| `test/test.ml` | 8,024 | Shared helpers, fixtures, one long test body, fold catalog |
| `lib/metal_runtime.ml` | 5,024 | Native handles, buffers/batches, parameter encoding, physical cache, command interpreter |
| `lib/serving_schedule.ml` | 4,517 | Command representation, validation, graph construction, specialization, binary codec |
| `lib/metal_block_codegen.ml` | 3,109 | Schema checks, quant decoders, decode/prefill/QKV emitters, epilogue specialization |
| `lib/serving_engine.ml` | 2,608 | Package contract, RoPE tables, engine/cache lifecycle, prompt/decode, speculative helpers |
| `lib/kernel_ir.ml` | 2,543 | Region representation/validation, scan recovery, fused-tactic recognition |

Evidence: these are the seven largest tracked OCaml implementation files in the
working checkout; assumption: they are the intended initial scope of “the big
files.” Smaller compiler files, Python, benchmarks, native Objective-C, and
vendored code are outside this LOOP except for necessary build/caller updates.

Read the repository's applicable instructions, [README](README.md),
[architecture](.okf/architecture.md), [backend boundary](.okf/decisions/backend-boundary.md),
and [Model Program boundary](.okf/decisions/model-program-boundary.md).
The architecture document's embedded ABI numbers can lag source: the inspected
schedule writer/reader use version **29**. Preserve the source's actual format.
`lib/layout_model.ml` already exists; its old extraction is not pending work here.

The compiler path is `bin/fx_compile.ml` → `Fx_plan.plan` → `Passes.optimize` →
`Serving_schedule.of_graph ~fusion_regions` and `Metal.lower ~fusion_regions` →
package/source files. Runtime execution reaches `Serving_engine` →
`Metal_runtime.encode_schedule` / bound plans → `Batch` →
`native/ocaml_metal_stubs.m`. Fusion recognition and serialized bindings must
continue to reach both backend emission and execution through those paths.

Existing patterns to follow: `Layout_model` is a separate domain unit;
`Serving_replay.Decode_buffers` and `Serving_specialization` already own serving
helpers; `Test_epilogue_monoid.run`, `Test_algebraic_canonicalization.run`, and
`Test_contraction_group.run` provide explicit test-suite entrypoints.

Ninja is hand-maintained in `ninja.build`, including explicit dependencies and
ordered link inputs for many executables. Add each new `.cmi`/`.cmx` edge and
every affected link input as part of its extraction, not in a later cleanup.
Internal modules depend on lower owners, never back on the public facade.
Give new units `.mli` files that expose only what sibling units actually use.
Public APIs stay at their existing paths; internal signatures may expose the
representation needed by their owning implementation family.

All module/file names introduced below are **planned**, under `lib/` unless a
`test/` path is given. Finish checks are shared rather than repeated per item.

## Work

- [x] 1. Turn `test/test.ml` into an explicit suite runner.
  Done (commits 0c88715..e29b49a, all green; llmopt-test stdout byte-identical
  to the pre-refactor baseline, exit 0).  test/test.ml is a 22-line runner that
  invokes Test_generation, Test_schedule, Test_hardware, Test_model_program,
  Test_cache, Test_quant, Test_graph (which internally calls the epilogue /
  algebraic-canonicalization / contraction-group suites at their original
  position), Test_metal, Test_fold_catalog, then the final success message.
  Helpers live in test/test_support.ml; Fold_catalog in test/test_fold_catalog.ml;
  Generation_test_engine/functor/tokenizer_fixture stayed with Test_generation.
  Cross-section bindings were duplicated as self-contained fixture constructors
  only (q4_m2_graph into Test_graph, lfm_att/lfm_rec + the identity/processor/
  gemma program builder into Test_cache).  Assertion inventory (543 call sites
  and all quoted messages) matches the original test.ml; the only added strings
  are the duplicated fixtures.  Ninja edges + link inputs updated per suite.
  Change: move `fail`, `expect`, `expect_ok`, `expect_int_array`,
  `contains_substring`, and `tensor_input` into `test/test_support.ml`.
  Keep `Generation_test_engine`, its functor application, and `tokenizer_fixture`
  with their generation/tokenizer suites. Move `Fold_catalog` into
  `test/test_fold_catalog.ml`. Split the large `let ()` into named `run`
  functions in suites for generation/protocol, cache/replay, schedule and
  specialization, Model Program/package contracts, hardware/cost, quant/GGUF,
  graph analysis/constructive lowering, and Metal emission/digests.
  Resolve local bindings reused across sections with fixture constructors or
  explicit arguments; do not create a shared mutable fixture registry. Keep
  each original assertion once, preserve setup dependencies, deterministic
  seeds, conditional real-GGUF checks, and existing suite calls. Keep the final
  success message after all suites. No library unit runs tests at initialization.
  Check: `ninja -f ninja.build ocaml-test` from the repo root invokes every old
  suite and the extracted suites. Compare the moved assertion messages and
  original call inventory to catch omissions; do not replace assertions with
  tests that merely check the new filenames. This first slice makes later
  verification easier without changing product code.

- [x] 2. Separate Kernel IR regions, scan recovery, and tactic recognition.
  Done (commit efab419, full `ninja all` green; llmopt-test stdout
  byte-identical to baseline).  kernel_ir.ml (2543) split into:
  kernel_region.ml/.mli (Slot/Primitive/Effect/Resource + region records,
  create, metadata validation, specialize, printing), kernel_scan.ml/.mli
  (recovery + fuse_triangular_recurrences), kernel_fused_tactic.ml/.mli
  (tactic select/select_all/fold_linear_epilogue, `open Kernel_region`, no
  Kernel_ir reference).  kernel_ir.ml/.mli is a facade: `include
  Kernel_region` + `module Scan = Kernel_scan` + `module Fused_tactic =
  Kernel_fused_tactic`.  Canonical records stay in Kernel_region; record
  labels therefore live under Kernel_region (and Kernel_region.Effect etc.),
  so consumers that used qualified labels / paths now spell Kernel_region.*
  (mechanical rewrite, 497 sites; types identical through facade aliases).
  Manifest-widened only where the family shares representation (Slot.t = int,
  binding/t/result records) so sibling units can construct/match regions.
  All executables were relinked; latent ninja link-order drift (some binaries
  stale since modules grew new deps) surfaced and was fixed by topologically
  sorting every link line deps-first and adding the missing sampling.cmx to
  llmopt-lfm-serving-check.
  Change: extract `Kernel_ir.Scan` (including recurrence discovery/recovery) to
  `kernel_scan.ml`. Move region ownership—`Slot`, `Primitive`, `Effect`,
  `Resource`, bindings/results, `create`, metadata validation, `specialize`, and
  printing—to `kernel_region.ml`. Move `Fused_tactic` and its `chain_of`,
  `fold_linear_epilogue`, `select`, and `select_all` to `kernel_fused_tactic.ml`,
  depending on `Kernel_region` and its queries. `kernel_ir.ml` becomes a facade:
  `include Kernel_region`, `module Scan = Kernel_scan`, and
  `module Fused_tactic = Kernel_fused_tactic`.
  Keep one canonical slot/binding/region representation. Use manifest aliases
  and re-exports where necessary so public constructors and record labels still
  refer to the same types; do not copy the variants into both units. Internal
  tactic code must not reference `Kernel_ir`, which would create a cycle.
  Check: existing region rejection, scan recovery, fold catalog, constructive
  lowering, and contraction-group cases retain their results through the public
  API. Their expected member IDs, bindings, materialization boundaries, and
  tactic choices remain the semantic evidence for this extraction.

- [x] 3. Establish a schedule core and extract binary serialization.
  Done (commit 68d886c; full `ninja all` green; llmopt-test output
  byte-identical).  serving_schedule.ml 4517 -> facade (1327): core
  1..1536 -> lib/serving_schedule_core.ml/.mli (representation, validation,
  create, of_graph, pruning, input queries); codec 2856..4517 ->
  lib/serving_schedule_codec.ml/.mli (`open Serving_schedule_core`; duplicate
  write/read_kernel_index at 3137/3225 preserved verbatim); module Sequence
  1537..2854 stays in the facade until item 4.  serving_schedule.mli
  unchanged.  New-unit .mli files were generated from inferred interfaces.
  Verification: native fixture package.llmopt regenerated at commit efab419
  (pre-move worktree) and post-move are byte-identical (SHA-256
  6ec95edc313aec79fbbd692516a6905059d3f83a59cc20993d9c9d19c5ebcbeb).
  Change: move `Command`, `Stage`, `Tensor_input`, the schedule representation,
  stage grouping, validation helpers through `validate_command`, `create`, graph
  construction/fusion attachment/barrier pruning, and input queries into
  `serving_schedule_core.ml`. Move `dtype_tag` through `of_bytes` into
  `serving_schedule_codec.ml`, depending only on the core and existing lower
  domains. Have `serving_schedule.ml` re-export the core and delegate
  `to_bytes`/`of_bytes`; temporarily keep `Sequence` there until item 4.
  Keep validated construction in the core so decoding and specialization call
  the same `create`, with no dependency back to the facade. The codec currently
  has repeated `write_kernel_index`/`read_kernel_index` definitions: preserve
  their binding behavior during the move; do not opportunistically merge them.
  Check: existing schedule round trips and invalid metadata/arity/input tests
  retain their behavior. Save the existing native fixture's `package.llmopt`
  bytes before this move and compare after regeneration; changing the OCaml
  compilation-unit layout does not justify changing format tags or field order.

- [x] 4. Separate schedule validation and sequence transformation from the core.
  Done (commit a84e2d1; full `ninja all` green; llmopt-test output
  byte-identical to baseline; llmopt-package-check decodes a real v12 package
  through the refactored codec/validation stack).
  serving_schedule_types.ml/.mli (47) = Command/Stage/Tensor_input + shared
  Int_set/value_id substrate; serving_schedule_validation.ml/.mli (1311) =
  metadata predicates + validate_command/validate_fusion_command (verbatim);
  serving_schedule_core.ml (195) = schedule t/commands/stages, create,
  attach/prune/of_graph, input queries, and re-exports types+validation so the
  codec and sequence depend on the core alone; serving_schedule_sequence.ml
  (1321) = the entire module-Sequence body with `open Serving_schedule_core`;
  facade serving_schedule.ml is now 10 lines (include core + module Sequence
  alias + to_bytes/of_bytes delegates).  Direction types -> validation -> core
  -> sequence/codec -> facade is acyclic (ocamldep).  serving_schedule.mli is
  unchanged.  Core contains no codec (dtype_tag/write_*) or sequence
  (specialize) implementations.  Ninja edges deps derived with ocamldep to
  avoid comment false-positives.
  Change: move command-related representations into
  `serving_schedule_types.ml`; the core re-exports them. Put the current
  metadata predicates and `validate_fusion_command`/`validate_command` into
  `serving_schedule_validation.ml`, depending on these types, and keep `create`
  and its seen-node/value traversal in the core. Move the entire `Sequence`
  implementation to `serving_schedule_sequence.ml`, depending on the core.
  The final public unit re-exports the core, aliases `Sequence`, and delegates
  the codec. The direction is types → validation → core → sequence/codec →
  facade, not mutually recursive modules.
  Preserve postprocessed last-token projection, shared-cache dimension
  provenance, RoPE substitution, pruning, suffix masks, paged decode, checkpoint
  specialization, and prompt-lookup specialization. Do not rename the runtime
  binding strings or broaden any recognizer.
  Check: existing prefill, suffix, direct-paged decode, shared-cache, and
  prompt-lookup tests pass through `Serving_schedule.Sequence`; the schedule
  core no longer contains codec or sequence-transform implementations.

- [x] 5. Extract Metal shader catalogs with their ABI entries.

  DONE (metal.ml 10484 -> 1581).  Owners created: metal_source_common
  (kernel_entry* + metal_header); metal_cache_source; metal_w4a16_source;
  metal_selection (module Tactic, starts item 6); metal_q8_0_source /
  metal_q5_0_source / metal_q4_0_source; metal_quant_qkv_source;
  metal_q4_k_source / metal_q5_k_source / metal_q6_k_source /
  metal_iq4_xs_source; metal_quant_source (quant_common_source, text patchers
  incl the first lexical replace_all, source selection, kquant_entries, all
  six public emit_dequant_*); metal_tensor_source (dense Linear + shared
  replace_first / second-lexical replace_all, then the tensor-primitive
  catalog: embedding/arange/diff/cumsum/tensor-primitive/fill/gather2/cast/
  pointwise/movement/reduction/update_slice); metal_normalization_source
  (RMS/L2/RoPE); metal_recurrent_source (ShortConv family,
  triangular_recurrence, gated_delta_tactic type + width/kernel_name/source/
  entry); metal_attention_source (attention_source/simd/tree + entries).
  Kept in metal.ml on purpose: block32_entries + add_block32_kernels (item 7
  dispatch), add_cache_kernels / add_kquant_kernels (Program-touching), the
  type-gated_delta re-export, and the item-6 body (has_* / *_tactics,
  quant_linear_* selectors, region helpers, private kernel type,
  lower_primary, lower, parametric emitters).  Every micro-commit: ninja all
  green, llmopt-test byte-identical, native fixture
  package.llmopt/kernel.metal SHA-256 identical (6ec95edc.../8ad477ac...).
  Source relocation alone retires nothing; no golden expectation changed.

  Change: put the four `kernel_entry*` constructors and `metal_header` into
  `metal_source_common.ml`. Move complete source/entry groups from `metal.ml`
  to these owners: `metal_w4a16_source.ml` (plain/QKV/SwiGLU/argmax),
  `metal_cache_source.ml`, `metal_quant_source.ml` (common quant declarations,
  source selection and entry assembly), `metal_normalization_source.ml`
  (RMS/L2/RoPE), `metal_attention_source.ml`, `metal_recurrent_source.ml`
  (ShortConv/triangular recurrence/Gated Delta), and
  `metal_tensor_source.ml` (dense Linear, embedding, tensor primitives,
  movement, pointwise, reduction, indexing and casts).
  Within the quant catalog, place each substantial existing literal family in
  `metal_q8_0_source.ml`, `metal_q5_0_source.ml`, `metal_q4_0_source.ml`,
  `metal_q4_k_source.ml`, `metal_q5_k_source.ml`, `metal_q6_k_source.ml`, and
  `metal_iq4_xs_source.ml`; keep Q4_0/Q5_0 QKV text in
  `metal_quant_qkv_source.ml`. These are storage-format owners for current
  source, not new tactic registrations or copies of schema-driven generators.
  Keep source strings and their matching entries together where possible;
  assembly functions take/return source and entry lists, not `Metal.Program.t`.
  Avoid a facade dependency from any catalog. Preserve literal contents and
  concatenation order, including helper declarations and duplicate-removal
  behavior. There are two lexically distinct `replace_all` helpers in
  `metal.ml`; keep each affected caller bound to the correct implementation.
  Check: existing exported `Metal.*_source`, entry lists, `emit_dequant_*`, and
  `add_*_kernels` still work; generated native fixture source and quant golden
  expectations retain their contents. Source relocation alone does not retire
  a legacy kernel or authorize updating a golden expectation.

- [x] 6. Separate Metal selection and assembly from emission bodies.

  DONE (metal.ml 1581 -> 806).  metal_selection.ml now owns module Tactic
  plus gated_delta_tactics, every has_* capability query, attention/quant
  tactic selection (attention_tactics, attention_simd_head_dimensions,
  quant_linear_tactics, quant_linear_add_tactics, graph_quant_types), and
  the fusion-region helpers (region_node_ids, without_region_nodes,
  quant_add_twin_servable, is_residual_add_epilogue, region_tactics).
  metal_primary_codegen.ml owns the private kernel representation, its
  find_input_name/input_name helpers, and lower_primary.  The three
  parametric emitters (w4a16_linear/q4_k_linear/simdgroup_matrix_gemm)
  live in metal_parametric_codegen.ml.  DEVIATION: module Program moved
  to lib/metal_program.ml (a lower codegen unit cannot depend on the
  Metal facade, and lower_primary/lower both assemble Programs);
  metal.ml aliases it and the metal.mli contract is unchanged.  metal.ml
  keeps Program/Tactic aliases, entry-construction helpers
  (attention_tactic_entries, quant_tactic_entries, quant_linear_add_entry),
  block32_entries + add_block32_kernels (item 7), add_cache_kernels /
  add_kquant_kernels, type-gated_delta re-export, and the assembly body
  lower + emit (incl quant-digest residual emission).  Every micro-commit:
  ninja all green, llmopt-test byte-identical, native fixture
  package.llmopt/kernel.metal SHA-256 identical (6ec95edc.../8ad477ac...).

  Change: move `Tactic`, graph capability queries (`has_*`),
  `gated_delta_tactics`, attention/quant tactic selection, `region_node_ids`,
  `without_region_nodes`, and `region_tactics` to `metal_selection.ml`.
  Keep source/entry construction in catalog owners; selection returns the
  existing domain choices and facts needed by assembly. Move the private
  `kernel` representation and `lower_primary` into `metal_primary_codegen.ml`,
  along with its input-name helpers. Move the three trailing
  `emit_parametric_*` / `emit_simdgroup_matrix_gemm` bodies into
  `metal_parametric_codegen.ml`.
  Leave `Program`, `lower`, public exports, and ordered source/kernel assembly
  in `metal.ml`. Delegate selection and source generation with qualified names;
  do not replace assembly with a runtime registry or move all of `lower` into
  another catch-all unit. Preserve filtering of region member nodes before
  ordinary graph selection and preserve fusion-region tactic contribution.
  Check: lowering the existing digest/fold cases still registers the same
  kernel names, buffer ABI and shader bodies; empty/unsupported paths keep their
  results. The remaining `metal.ml` explains assembly without embedded shader
  catalogs or per-operation emitter bodies.

- [x] 7. Split schema-driven block code generation by emission responsibility.
  DONE (metal_block_codegen.ml 3109 -> 43). Leaf emitters live in
  metal_block_codegen_common (geometry checks + block32/superblock templates),
  metal_q4_0_decode_codegen (x4/x8/gated decode emitters),
  metal_q4_0_prefill_codegen (m16/m32 emitters), metal_q4_0_qkv_codegen (QKV x8/m32),
  and metal_q4_0_epilogue_codegen (epilogue stores + digest_of_flat).
  Non-Q4_0 formats and their residual/add variants are in metal_block_codegen_formats.ml.
  Q4_0 schema assembly, specialized epilogue/multi-linear wrappers, and add twins
  live in metal_q4_0_codegen.ml. metal_block_codegen.ml is now a clean 43-line
  dispatch and wrapper facade; its public .mli contract is 100% preserved.
  Verified: ninja all green; test suites (OCaml + 79 Python) pass; smoke tests pass.

  Change: extract shared schema offset/geometry checks and common block32 and
  superblock loops into `metal_block_codegen_common.ml`; put the non-Q4_0
  format assemblers and their residual/add variants in
  `metal_block_codegen_formats.ml`. Move Q4_0 x4/x8 and gated decode emitters
  into `metal_q4_0_decode_codegen.ml`, m16/m32 emitters into
  `metal_q4_0_prefill_codegen.ml`, and QKV x8/m32 emitters into
  `metal_q4_0_qkv_codegen.ml`.
  Put `q4_0_*_digest_of_flat`, epilogue-store generation, and
  `q4_0_decode_epilogue_kernel` in `metal_q4_0_epilogue_codegen.ml`. Q4_0 schema
  assembly and public specialized wrappers belong in
  `metal_q4_0_codegen.ml`, calling the emitter leaves. Keep
  `metal_block_codegen.ml` as the public `linear_kernels`/`qkv_kernels` dispatch
  and wrapper facade. Move a helper to the lowest actual shared owner instead
  of duplicating it or introducing a cycle between assembler and emitters.
  Check: preserve every existing function in `metal_block_codegen.mli`,
  including the recent K-family m2_x4 additions. Reuse `test/golden/*.msl` and the
  existing schema, residual, x4/x8/m16/m32 digest tests, including zero-buffer
  epilogues and row guards. This item changes ownership, not Q4_0 arithmetic,
  schema geometry, generated names, or which templates remain available.

- [x] 8. Extract the runtime's native adapter, parameters, and physical cache.
  DONE (metal_runtime.ml 5397 -> 3868). Extracted pure parameter encoders
  into metal_parameters.ml/.mli (370 lines), native handle declarations, runtime
  record, library loading, protect, Buffer, Batch, Ring_queue, Prebaked, and
  low-level dispatch into metal_native.ml/.mli (363 lines), and physical KV cache
  implementation into metal_cache.ml/.mli (473 lines). Public Metal_runtime aliases
  these modules and retains identical native runtime/buffer identity, wire formats,
  and lifetimes. Native stubs in native/ocaml_metal_stubs.m remain untouched.
  Verified: ninja all green; 79 Python unit tests pass; OCaml test suite passes;
  all 4 smoke targets (native-schedule-smoke, w4a16-serving-smoke,
  ocaml-metal-primitives-smoke, ocaml-metal-runtime-smoke) pass on Apple M4 Pro.

  Change: create `metal_native.ml` owning the external handle declarations,
  runtime record and loading, `protect`, `Buffer`, `Batch`, `Ring_queue`,
  `Prebaked`, tensor lookup, and low-level dispatch. Native symbol strings and
  tuple layouts remain unchanged; `native/ocaml_metal_stubs.m` is not being split.
  Move `Parameters` into `metal_parameters.ml` and `Cache` into
  `metal_cache.ml`. Cache depends on the native adapter and parameters, not on
  schedule execution. Keep helpers such as `kernel_entry` in the lowest owner
  that their existing native/cache callers need. Public `Metal_runtime` aliases
  these modules and retains the same native runtime/buffer identity.
  Check: package/library loading, buffer views, cache pack/unpack and batch
  calls still compile through the existing API. Existing native fixture smoke
  commands in Finish exercise the adapter; do not infer model execution from
  successful compilation. Keep mapped archive and buffer parents alive exactly
  as before, and keep command batching/commit/abort ownership in `Batch`.

- [x] 9. Break up runtime encoding without changing its traversal or lifetime.
  Change: create `metal_execution_state.ml` owning `Execution`,
  `Execution_batch`, `execution_state` (as module-scoped `t`), value/input maps,
  workspace binding, retained resources, `with_execution_batch`, and the cache
  encoding bridge. It depends on native/cache owners. Put grid/kernel-selection
  helpers and `dispatch_output`/`dispatch_quant_linear` in
  `metal_dispatch_common.ml`; put W4 macro dispatchers and quant QKV dispatch in
  `metal_dispatch_linear.ml`. Move `assignment_execution` and its fused
  assignment dispatch into `metal_dispatch_assignment.ml`.
  In `encode_schedule`, retain one ordered traversal but extract operation
  bodies to `metal_dispatch_tensor.ml`, `metal_dispatch_normalization.ml`,
  `metal_dispatch_attention.ml`, and `metal_dispatch_recurrent.ml`; plain/dense
  and quant Linear cases go to the Linear owner. Assignment dispatch and the
  ordinary router both use the same leaf helpers, buffer map, and execution
  batch. Pass a concrete execution context for runtime/batch/cache/uniforms;
  do not thread recursive `continue` closures through the leaf functions.
  Normalize each extracted operation to return its updated state plus kernel
  names in execution order. The caller alone updates `kernels_rev` and advances
  to the next command. Input/output/barrier routing remains in
  `metal_schedule_executor.ml` with workspace preparation and the traversal;
  `execute_schedule`, `bind_schedule_plan`, `bind_decode_plan`, and `execute`
  delegate there through the `Metal_runtime` facade.
  Worked example: the existing QKV macro returns `(state, [k1; k2])`;
  the traversal applies `List.rev_append [k1; k2] state.kernels_rev` once,
  then proceeds. It must not reverse the dispatch list itself, execute a second
  fallback, submit a batch early, or release a buffer when a leaf returns.
  Preserve assignment-first routing, barrier markers, parameter bytes, grid
  sizes, copy/alias behavior, the unsupported-node error, and bound-plan command
  ordering. Route existing cases directly by typed operation; no chain of
  optional handlers whose order silently chooses semantics.
  Check: where the existing Metal environment is available, compare fixture
  outputs and reported kernel order before/after using the native smoke runners.
  Existing source/ABI tests cover the digest
  buffer positions; cache/command tests cover the moved helpers. Any missing
  focused assertion should exercise a concrete moved invariant, not introduce
  a new test framework or synthetic performance claim.

- [ ] 10. Extract independent serving-engine domains while retaining one engine owner.
  Change: move attention/recurrent bindings, head/entrypoint contract records,
  input/output metadata checks, `contract_of_parts`, `contract`, and
  `validate_packages` into `serving_engine_contract.ml`. Keep each domain's
  primary type under its owner (`Attention_binding.t`, `Recurrent_binding.t`,
  `Head.t`, and the contract unit's `t`). Move the existing `rope_table` and
  `Rope_table` implementation into `serving_rope_table.ml`.
  Move `Speculative_acceptance`, `Target_coupled_mtp`, `Medusa`, and pure ngram
  proposal logic to `serving_speculative.ml` as their existing nested domains.
  Keep `Prompt_lookup.verify_and_commit` and its engine/cache access in
  `serving_engine.ml`, composing the extracted proposal/acceptance functions
  while retaining public `Serving_engine.Prompt_lookup` result accessors.
  Keep `Step`, `Prompt`, `Sequence`, engine creation, prefill/decode/replay,
  reservations, and mutable cache/executable ownership together in the engine.
  Reuse `Serving_replay` and `Serving_specialization`; do not introduce a public
  engine-state record or pass the entire engine into the extracted pure domains.
  Check: package-contract rejection, RoPE-dependent specialization, generation
  and speculative acceptance/proposal cases retain their behavior. Contract and
  speculative helper modules can be understood without runtime lifecycle code;
  engine paths still own cache lease/release and speculative commit/rollback.

- [ ] 11. Update the architecture map to the implemented ownership.
  Change: update the relevant portions of `.okf/architecture.md` and affected
  source references to the actual extracted owners. Keep public caller paths
  and existing semantic decisions clear; this LOOP records the decomposition
  rationale, so a duplicate ADR is unnecessary. Follow the OKF skill's metadata,
  log, and validation conventions for documents actually changed. Do not rewrite
  historical experiment receipts as if the refactor reran their measurements.
  Check: each named owner exists and its documented dependency direction matches
  source and Ninja. Run the installed OKF validator on changed bundle content;
  report pre-existing issues separately rather than expanding into bundle repair.

## Finish

Run commands from the repository root. During extraction use the affected `.cmx`
or existing `ocaml-test` target; after the full refactor run the shared checks:

```sh
ninja -f ninja.build all
ninja -f ninja.build test
git diff --check
```

Use these existing fixture checks for generated-output and runtime evidence in
the available Metal environment; they introduce no user-device setup prerequisite:

```sh
ninja -f ninja.build native-schedule-smoke w4a16-serving-smoke
ninja -f ninja.build ocaml-metal-primitives-smoke ocaml-metal-runtime-smoke
```

These targets exist in the inspected `ninja.build`. The first shared pair covers
build and existing OCaml/Python tests; package smoke targets cover emitted
source, Metal compilation and package validation; the last smoke pair actually
executes small native fixtures. Report unavailable tools/devices as missing
evidence, without asking the user to set up a device or inventing an approval gate.
Fixture generation currently lists its generator as an
order-only dependency: when comparing before/after artifacts, regenerate into
fresh scratch directories with `_build/bin/llmopt-native-schedule-fixture DIR`
instead of accepting a cached fixture as new evidence. Likewise, invoke rebuilt
smoke runners directly against their verified fixture directories when Ninja
reports an already-cached success receipt. Preserve the baseline outputs outside
the regenerated directory. Do not delete unrelated `_build` or model artifacts.

Inspect Ninja dependencies with `ninja -f ninja.build -t query TARGET` when
adding a unit. A successful incremental build can hide a missing dependency;
verify new units' direct dependencies and link order rather than rewriting the
build system. Keep explicit source/golden comparisons described in the items;
ordinary round trips alone could hide a codec writer/reader change made together.

Planning-time OKF validation was conformant with zero errors and three existing
strict-mode warnings: `status: accepted` in the applicative-IR, Gemma round6,
and Gemma round7 decision files. Those warnings are not refactor work.

Completion means the seven original files delegate the named responsibilities
to their owners, no old implementation remains duplicated, every original test
still runs, and the focused checks report the actual result. Summarize before/
after sizes as information, list affected ownership boundaries, and distinguish
source/serialization evidence from the small native fixture executions. Report
unfavorable or pre-existing results precisely. No throughput comparison, full
model parity campaign, new approval checkpoint, or arbitrary file-size cutoff
is part of this refactor.

Preserve unrelated work throughout. If a commit is separately requested, stage
only the intended files/hunks; do not use broad staging in this mixed checkout.
All unfinished items remain in scope when execution resumes.

Next action: `/loop-execute LOOP-large-file-refactor.md`.

---

## Item 3 prep notes (session 3) -- serving_schedule split map

lib/serving_schedule.ml = 4,517 lines, three contiguous regions:
- Core 1..1536: `let*`, module Command (3-17), module Stage (19-33), module
  Tensor_input (35-43), type t, commands/stages, Int_set (74), validation
  helpers (78..1381 incl validate_fusion_command 252, validate_command 278),
  create (1382), attach_fusion_region (1402), prune_redundant_barriers (1474),
  of_graph (1492), tensor_inputs/runtime_inputs/opaque_count (1514-1536).
- module Sequence 1537..2854 (module `end` at 2854) -> STAYS in the facade
  until item 4.  Codec region (2856+) does NOT reference Sequence (verified).
- Codec 2856..4517: dtype_tag (2856) .. write/read helpers .. to_bytes (4451),
  of_bytes (4469) .. 4517.  Codec core deps: Command. (7), create (24),
  value_id (13), commands (4), Int_set (2), validate_command (1) -- so
  serving_schedule_core.mli must expose those internal-but-shared items plus
  modules Command/Stage/Tensor_input, type t, of_graph.
- Note the duplicate write_kernel_index/read_kernel_index definitions at 3137
  and 3225 -- move verbatim, do NOT merge (LOOP item 3 requirement).

Plan: serving_schedule_core.ml/.mli (1..1536), serving_schedule_codec.ml/.mli
(2856..4517, `open Serving_schedule_core`), serving_schedule.ml facade:
`include Serving_schedule_core` + Sequence body (1537..2854) + delegate
to_bytes/of_bytes to the codec.  serving_schedule.mli (110 lines) stays the
public contract (Command/Stage/Tensor_input/t/of_graph/Sequence/to_bytes/of_bytes).

Native fixture check: bin/native_schedule_fixture.ml writes
`<dir>/package.llmopt` (line 709).  Before the move: run
`_build/bin/llmopt-native-schedule-fixture <outdir>` (see usage line 687 for
inputs), save bytes; after the move regenerate and cmp.  _artifacts already
holds gemma4 package.llmopt artifacts that round-trip through of_bytes.

## Item 4 prep notes (session 3) -- schedule types/validation/sequence split

Current serving_schedule_core.ml line anchors (doc header shifts originals +4):
- types chunk: module Command 8-22, Stage 24-38, Tensor_input 40-48 (must stay
  one unit so variants/labels share), plus module Int_set 79 and
  `let value_id` 81 (shared substrate needed by BOTH validation (before core)
  and core/codec -> keep in types to avoid a validation<->core cycle).
- validation chunk: scalar_matches_dtype 83 .. end of validate_command ~1385
  (includes fresh_outputs 252, validate_fusion_command 257, validate_command
  283).  Deps: Ir, Tensor_shape, Kernel_region, Contraction_group + types
  (Int_set/value_id).  validate_command is called by create (1395 in core) and
  by the codec's of_bytes, so core re-exports it for the facade/codec path.
- core keeps: let (let*), type t {commands}, commands, stages (54-77), create
  1387, attach_fusion_region 1407, is_gpu_command_in_schedule 1464,
  prune_redundant_barriers 1479, of_graph 1497, tensor_inputs 1519,
  runtime_inputs 1528, opaque_count 1536.  Core = module aliases for the types
  unit + open/include the validation unit's re-exported val, then the above.
- Sequence (module body 1537..2854 in the ORIGINAL serving_schedule.ml) ->
  serving_schedule_sequence.ml with `open Serving_schedule_core`; facade then
  `module Sequence = Serving_schedule_sequence` and the facade shrinks to:
  include core + alias Sequence + to_bytes/of_bytes delegates (codec).
- New .mli files from inferred interfaces (ocamlfind ocamlopt -i), doc header.
- Direction stays acyclic: types -> validation -> core -> sequence/codec -> facade.
- Check: prefill/suffix/direct-paged/shared-cache/prompt-lookup tests via
  Serving_schedule.Sequence stay green; core no longer holds codec/sequence.

## Execution log

Item 1 completed (session 2, commits 0c88715..e29b49a).  Final state:
test/test.ml 8024 -> 22 lines (pure suite runner).  New units under test/:
test_support.ml, test_fold_catalog.ml, test_generation.ml (500),
test_schedule.ml (3045), test_hardware.ml (94), test_model_program.ml (349),
test_cache.ml (444), test_quant.ml (653), test_graph.ml (1839),
test_metal.ml (730).  All green; llmopt-test stdout byte-identical to the
64-line pre-refactor baseline; assertion call inventory (543) and message
strings preserved (only sanctioned fixture-duplication strings added).

Baseline artifacts live in _artifacts/refactor-check/ (baseline-test.out etc.).

Working notes for the remaining items:
- Ninja build-file quirks: large single-line `edit` calls are unreliable; patch
  the llmopt-test link line with a python replace and verify each edit with grep.
- Test-edge deps can be derived by scanning capitalized module identifiers
  against lib/*.ml (a python `uses()` helper did this during extraction).
- Multi-fragment suites need one `run ()` call site that preserves marker
  order, and a `;` seam between fragments.

Next action: item 2 (Kernel IR region/scan/tactic separation, kernel_ir.ml 2543
lines). Resume with `/loop-execute LOOP-large-file-refactor.md`.
