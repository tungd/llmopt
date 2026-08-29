# LOOP-model-program-execution-contract.md

> Follow-up (2026-08-29): the completed milestone was generalized after this
> checklist. Generic surfaces are now `Model_profile`, `Model_program_linker`,
> `Chat_template`, `Serving_schedule.Sequence`, `bin/pipeline.ml`,
> `bin/generate.ml`, and `bin/serve.ml`. LFM2.5 metadata remains only in the
> explicit `Lfm25_probe` fixture, probe binaries, tests, and historical
> receipts; `model.llmopt` ABI v2 declares recurrent-window width. The checked
> items below retain their implementation-time symbol names as historical
> evidence.

## GOAL

Introduce the first package-resident Model Program milestone: compile the
current LFM prefill/decode pair into a root `model.llmopt` contract, then make
the shared serving cache and engine consume compiler-declared entrypoint,
state, and specialization metadata instead of reconstructing LFM topology and
captured placeholder names.

This milestone preserves the current W4A16/group-64-KVQ8 graph, schedule,
kernel, package, tokenizer, HTTP, and generation behavior. It establishes the
execution-contract seam required before adding another model adapter.

## OUT OF SCOPE

- Implementing or loading Gemma weights.
- Adding group-32 QAT, new weight packing, or a generalized tensor-format ABI.
- Replacing `Lfm_chat`, generalizing Hugging Face/Jinja chat templates, or
  adding multimodal processor execution.
- Changing the current W4A16/group-64-KVQ8 operator, schedule, kernel, cache,
  or per-graph package formats.
- Changing HTTP/SSE request semantics, queue policy, batching, sampling, or
  benchmark protocol.
- Introducing a performance target; any timing or memory observation is
  informational only.

## EXECUTION INSTRUCTIONS

- Start by checking `git status --short --branch`, `git log -1 --oneline`, and
  the files and symbols named by the active item. The inspected baseline for
  this plan is clean `main` at `f048a63`.
- Preserve unrelated work. If an active item overlaps uncommitted changes,
  inspect and reconcile ownership before editing; never reset or overwrite it.
- Work one item at a time in dependency order. Use a red-green-refactor cycle
  for each new contract or behavior.
- Run the exact `VERIFY` command from the stated repository root. Record the
  observable result under that item as `DONE` evidence.
- Commit each completed item as one coherent slice, staging only the paths
  named by the item. Explain any extra touched path as inseparable before
  including it.
- If source evidence invalidates an assumption, repair this LOOP and its
  supporting decision before continuing. Do not silently invent a replacement
  design.

## RELEVANT FILES

### Repository: `/Users/tung/Projects/std23/llmopt`

- `python/llmopt_backend/__init__.py`: Existing Dynamo/FX capture session and
  shared tensor-archive ownership; context only in this milestone.
- `lib/serving_package.ml` and `lib/serving_package.mli`: Existing
  per-entrypoint binary package contract; remains separate from the new root
  Model Program.
- `lib/serving_schedule.ml` and `lib/serving_schedule.mli`: Generic schedule
  codec plus the current embedded `Serving_schedule.Lfm25` specializer.
- `lib/serving_cache.ml` and `lib/serving_cache.mli`: Current cache config that
  embeds `Lfm25.Config.t`.
- `lib/serving_engine.ml` and `lib/serving_engine.mli`: Current runtime contract
  reconstruction, LFM binding names, and direct LFM specialization calls.
- `lib/lfm25.ml` and `lib/lfm25.mli`: Current model facts that the compile-time
  adapter will translate into Model Program data.
- `bin/lfm_pipeline.ml`: Existing graph-pair compiler and engine-directory
  linker.
- `bin/lfm_serve.ml`, `bin/lfm_generate.ml`, and
  `bin/lfm_serving_check.ml`: Existing root callers that inject
  `Lfm25.Config.default`.
- `test/test.ml`: OCaml contract, specialization, cache, and codec tests.
- `ninja.build`: Required build graph and the only build orchestrator.
- `lib/model_program.ml` and `lib/model_program.mli` (planned): Root Model
  Program domain, validation, and binary ABI.
- `lib/lfm25_program.ml` and `lib/lfm25_program.mli` (planned): Compile-time
  LFM adapter that emits Model Program data.
- `lib/serving_specialization.ml` and
  `lib/serving_specialization.mli` (planned): Model-neutral sequence and
  paged-cache specialization implementation.
- `bin/model_program_check.ml` (planned): Offline root-program and referenced
  artifact validator.

## SUPPORTING DOCUMENTS

- [Model Program boundary](.okf/decisions/model-program-boundary.md) — Status:
  `CREATED`, `DRAFT`. Governs the root contract, ownership boundaries, first
  adapter, and later milestones. ITEM-08 updates its status using implementation
  evidence.
- [Architecture](.okf/architecture.md) — Status: `EXISTING`. Records the
  current Dynamo/FX, OCaml, package, and Metal runtime boundaries; ITEM-08
  updates only the ownership and package sections affected by this milestone.
- [FX backend boundary](.okf/decisions/fx-backend.md) — Status: `EXISTING`.
  Keeps Dynamo/FX as the public low-level graph capture boundary.
- API/schema contract — Status: `N/A`. The Model Program boundary decision and
  `Model_program.mli` together are the contract; a second schema document would
  duplicate authority.
- Migration plan — Status: `N/A`. This milestone adds a root manifest while
  retaining the existing per-graph package ABI and current engine directory
  contents.
- Rollout/rollback runbook — Status: `N/A`. This is an offline research compiler
  artifact with no external production deployment in scope.

## COMPLETE WHEN

1. `llmopt-pipeline` emits a root `model.llmopt`, and
   `llmopt-model-check <engine-directory>` loads it, validates its referenced
   tokenizer and prefill/decode packages, and reports the declared execution
   contract.
2. `Serving_cache` constructs physical KV/checkpoint layout from the declared
   persistent-state plan, not from `Lfm25.Config.t`.
3. `Serving_engine` consumes declared entrypoint inputs, outputs, state
   bindings, generation dimensions, and specialization metadata; it contains
   no `Lfm25` reference or hard-coded `l_kwargs_*` binding name.
4. The current saved LFM graph pair passes the model-program checker and the
   same static prefill/decode specialization assertions as before the move.
5. `ninja -f ninja.build test all` passes, the OKF bundle has no conformance
   errors, and the cumulative diff contains only this milestone or explicitly
   documented inseparable changes.

---

## Execution items

- [x] **ITEM-01: Define and encode Model Program ABI v1**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: New root execution-contract domain immediately above
    `Serving_package`.
  - `IMPORTANT FILES`:
    - `lib/model_program.mli`: Create the abstract public contract and accessors.
    - `lib/model_program.ml`: Create validation plus versioned binary
      serialization/deserialization.
    - `test/test.ml`: Add focused construction, validation, malformed-input,
      and binary round-trip tests.
    - `ninja.build`: Register the interface/module before serving-cache and
      engine consumers.
  - `IMPORTANT SYMBOLS`: `Model_program.Identity`,
    `Model_program.Artifact`, `Model_program.Processor`,
    `Model_program.Entrypoint`, `Model_program.Generation`,
    `Model_program.State`, `Model_program.Specialization`,
    `Model_program.create`, `Model_program.validate`,
    `Model_program.to_bytes`, `Model_program.of_bytes`,
    `Model_program.write_file`, `Model_program.of_file`,
    `Model_program.current_abi_version`, `test_model_program_round_trip`,
    `test_model_program_rejects_invalid_contract`.
  - `WHY`: The root engine directory currently has no authoritative contract
    connecting processor assets, prefill/decode packages, state bindings, and
    generation dimensions.
  - `FIX`: Add ABI v1 with canonical relative artifact paths; exactly one
    prefill and one decode entrypoint; declared token input and optional
    `logits`/`token_id` outputs; vocabulary and maximum positions; attention and
    recurrent binding lists plus cache dimensions; and generic specialization
    metadata for minimum prefill length and injected RoPE/paged-cache inputs.
    Permit an empty attention or recurrent family in the data model, but require
    at least one declared output for each entrypoint and unique binding names and
    state-layer indices.
  - `QUALITY`: Keep representation abstract behind smart constructors; reject
    non-canonical paths, duplicate roles/names/indices, non-positive declared
    dimensions, unknown tags, truncation, and trailing bytes. Preserve unknown
    model identity as provenance rather than dispatch logic.
  - `DO NOT`: Do not embed Metal handles, OCaml functions, LFM-specific variant
    constructors, quantized weight descriptors, chat-template logic, or HTTP
    configuration.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run
    `ninja -f ninja.build test`.
  - `DONE WHEN`: The new tests prove exact accessor and byte round-trip behavior
    and reject each malformed contract class named above; the existing OCaml and
    Python tests also pass.
  - `ESCALATE IF`: A required field can only be reconstructed from a compiled
    schedule and cannot be represented without embedding model-family behavior;
    update the decision and split that field into compile-time analysis data
    before proceeding.

- [x] **ITEM-02: Add the compile-time LFM Model Program adapter**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Model-family analysis seam between the two captured packages and
    the root Model Program.
  - `IMPORTANT FILES`:
    - `lib/lfm25_program.mli`: Create the LFM adapter entrypoint.
    - `lib/lfm25_program.ml`: Move current LFM topology, binding-name, head, and
      shape interpretation into compiler-side analysis.
    - `test/test.ml`: Add an LFM contract fixture covering the exact hybrid
      attention/recurrent topology and named prefill/decode roles.
    - `ninja.build`: Register the adapter only in compiler/linker and test
      targets, not as a dependency of shared serving-cache or engine modules.
  - `IMPORTANT SYMBOLS`: `Lfm25_program.of_packages`,
    `Lfm25_program.cache_bindings`, `Lfm25_program.validate_entrypoints`,
    `test_lfm25_model_program_contract`.
  - `WHY`: `Serving_engine.cache_bindings` and `Serving_engine.contract`
    currently rediscover model topology, captured placeholder names, output
    roles, vocabulary, and captured lengths at runtime.
  - `FIX`: Construct `Model_program.t` from `Lfm25.Config.t`, tokenizer and
    package artifact paths, and the compiled prefill/decode schedules. Validate
    the same dtype/shape/name invariants currently enforced by
    `Serving_engine.contract`, then store the resulting roles and bindings as
    data. Declare current generic specialization inputs and the recurrent
    minimum-prefill rule in `Model_program.Specialization`.
  - `QUALITY`: Keep all LFM spellings and layer-type traversal local to
    `Lfm25_program`; produce deterministic binding order; return contextual
    errors naming entrypoint, binding role, expected metadata, and actual
    metadata.
  - `DO NOT`: Do not move schedule rewriting, cache allocation, Metal lowering,
    or request execution into the adapter. Do not weaken the current exact
    prefill/decode input and output validation.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run
    `ninja -f ninja.build test all`.
  - `DONE WHEN`: The LFM fixture produces one valid root program with the
    current 6 attention and 10 recurrent bindings, the captured template
    lengths and output roles are schedule-derived, and generic runtime modules
    do not depend on `Lfm25_program`.
  - `ESCALATE IF`: The current captured packages expose more than one plausible
    tensor for a required role or their validated topology differs from
    `Lfm25.Config.default`; record the exact package evidence and repair the
    adapter contract rather than choosing by name proximity.

- [x] **ITEM-03: Link and validate the root model artifact**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: `llmopt-pipeline` engine-directory linker and offline validation
    CLI.
  - `IMPORTANT FILES`:
    - `bin/lfm_pipeline.ml`: Return each compiled `Serving_package.t`, invoke
      `Lfm25_program.of_packages`, and write `model.llmopt` after both package
      directories and shared assets exist.
    - `bin/model_program_check.ml`: Create an offline validator that loads the
      root program, resolves canonical paths, loads both packages, and validates
      every declared binding against its schedule.
    - `ninja.build`: Build `_build/bin/llmopt-model-check` and include it in
      `all`.
  - `IMPORTANT SYMBOLS`: `compile_single_graph`, `run`,
    `Model_program.validate_files`, `Model_program_check.run`.
  - `WHY`: Copying tokenizer, weights, and two independent packages into one
    directory does not make their relationship executable or independently
    checkable.
  - `FIX`: Emit `model.llmopt` last so partial pipeline failures never leave a
    seemingly complete root contract. The checker must validate processor and
    package artifact paths, package stages and ABIs, per-package files,
    schedules, entrypoint roles, declared outputs, and state binding
    membership; print identity, entrypoint names, state-family counts, and
    physical cache description.
  - `QUALITY`: Reuse `Serving_package.Artifact` path semantics or one shared
    canonical-path implementation; do not duplicate package validation logic.
    Errors must include the root artifact path and failing declared role.
  - `DO NOT`: Do not compile Metal, allocate GPU resources, parse HTTP input, or
    load PyTorch in the checker. Do not fold `Serving_package` bytes into
    `model.llmopt`.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run
    `ninja -f ninja.build test all` and confirm
    `test -x _build/bin/llmopt-model-check`.
  - `DONE WHEN`: A pipeline-created engine directory contains one root program,
    the checker resolves both entrypoint packages and tokenizer from it, and a
    missing or mismatched referenced artifact produces a path-specific error.
  - `ESCALATE IF`: Emitting the root program would require changing the existing
    graph, schedule, kernel, weight archive, or per-graph package ABI; keep that
    change outside this milestone and repair the linker boundary instead.

- [x] **ITEM-04: Derive serving-cache configuration from the state plan**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Logical radix/KV cache configuration boundary.
  - `IMPORTANT FILES`:
    - `lib/serving_cache.mli`: Add a Model Program state-plan constructor and
      state-plan accessor while retaining a temporary LFM compatibility
      constructor until ITEM-07 migrates all callers.
    - `lib/serving_cache.ml`: Validate and translate declared state dimensions
      into `Kv_cache.Layout.t` and `Kv_cache.Config.t`.
    - `test/test.ml`: Add hybrid-LFM and transformer-only state-plan fixtures.
  - `IMPORTANT SYMBOLS`: `Serving_cache.Config.of_state_plan`,
    `Serving_cache.Config.state_plan`, `test_serving_cache_from_state_plan`,
    `test_serving_cache_transformer_only_plan`.
  - `WHY`: `Serving_cache.Config.create` currently requires
    `Lfm25.Config.t`, counts LFM layer variants, and derives recurrent and
    attention dimensions inside the shared cache module.
  - `FIX`: Accept the declared `Model_program.State.t`, caller-provided token
    and checkpoint capacities, and page size. Validate the current physical Q8
    constraints through `Kv_cache.Layout.create`; allow a state plan with zero
    recurrent bindings and represent its checkpoint payload as zero bytes.
  - `QUALITY`: Keep radix ownership, slot allocation, byte accounting, and the
    current group-64 format behavior unchanged. Compare declared binding counts
    with physical layout counts before allocation.
  - `DO NOT`: Do not add another KV precision, group size, paging policy,
    checkpoint allocator, or change cache eviction behavior. Do not remove the
    compatibility constructor until all callers migrate in ITEM-07.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run
    `ninja -f ninja.build test`.
  - `DONE WHEN`: The existing LFM state plan reproduces the current cache layout
    and byte accounting, a zero-recurrent plan is accepted without recurrent
    payload bytes, and no new LFM-specific code is added outside the temporary
    compatibility constructor.
  - `ESCALATE IF`: `Kv_cache` requires a non-empty recurrent payload for a
    zero-recurrent plan or changing that invariant would affect radix ownership;
    record the exact failing invariant and split optional-checkpoint lifecycle
    into the later persistent-state LOOP.

- [x] **ITEM-05: Extract model-neutral sequence specialization**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Dynamic prefill/decode schedule specialization between typed
    schedules and the serving engine.
  - `IMPORTANT FILES`:
    - `lib/serving_specialization.mli`: Create the generic specialization
      interface parameterized by `Model_program.Specialization.t` and cache
      layout.
    - `lib/serving_specialization.ml`: Move shape substitution, final-token
      projection, RoPE-input replacement, dead-command pruning, and direct
      paged-Q8 attention rewriting out of `Serving_schedule.Lfm25`.
    - `lib/serving_schedule.ml` and `lib/serving_schedule.mli`: Retain schedule
      representation and codec; temporarily expose the old LFM functions as
      thin delegates for equivalence testing until ITEM-06 removes them.
    - `test/test.ml`: Compare old delegate and new generic outputs byte-for-byte
      for prefill and decode fixtures.
    - `ninja.build`: Register the module between schedule and engine.
  - `IMPORTANT SYMBOLS`: `Serving_specialization.specialize_prefill`,
    `Serving_specialization.specialize_decode`,
    `Serving_specialization.specialize_decode_paged_q8`,
    `Serving_specialization.direct_q8_attention`,
    `Serving_specialization.specialize_rope`,
    `test_generic_prefill_specialization_equivalence`,
    `test_generic_decode_specialization_equivalence`.
  - `WHY`: Most of `Serving_schedule.Lfm25` is general schedule rewriting, but
    it is coupled to a fixed recurrent window, injected input names, and
    model-branded errors.
  - `FIX`: Parameterize those values from the Model Program specialization and
    state plans. Keep operator shape inference and cache-chain legality checks
    unchanged. The temporary `Serving_schedule.Lfm25` surface must delegate to
    the new implementation rather than retain a second copy.
  - `QUALITY`: Preserve command order, value IDs, additional-output arity,
    pruning behavior, schedule serialization, and error specificity. Rename
    model-branded diagnostics to entrypoint/role diagnostics.
  - `DO NOT`: Do not add Gemma-specific rules, alter fusion selection, change
    schedule ABI tags, or weaken opaque-operation rejection.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run
    `ninja -f ninja.build test all`.
  - `DONE WHEN`: Existing LFM fixtures produce byte-identical specialized
    schedules through the old delegate and the generic implementation, and the
    implementation body exists only in `Serving_specialization`.
  - `ESCALATE IF`: Any moved rewrite depends on undeclared model topology or a
    captured placeholder spelling; add the smallest explicit Model Program
    field in ITEM-01/ITEM-02 and update the decision before continuing.

- [x] **ITEM-06: Make Serving Engine consume Model Program data**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Shared runtime validation, schedule selection, state binding, and
    execution coordination.
  - `IMPORTANT FILES`:
    - `lib/serving_engine.mli`: Replace config-plus-package construction with a
      Model Program plus resolved entrypoint runtimes.
    - `lib/serving_engine.ml`: Remove runtime LFM contract reconstruction and
      consume declared entrypoint, state, generation, and specialization data.
    - `lib/serving_schedule.ml` and `lib/serving_schedule.mli`: Remove the
      temporary `Lfm25` delegates after the engine migrates.
    - `test/test.ml`: Add invalid-program/package mismatch tests and preserve
      specialization/state binding assertions.
  - `IMPORTANT SYMBOLS`: `Serving_engine.validate_program`,
    `Serving_engine.create`, `Serving_engine.prefill_schedule`,
    `Serving_engine.decode_schedule`, `Serving_engine.validate_tokens`,
    `Serving_engine.Rope_table.create`, `test_serving_engine_program_mismatch`;
    remove private `Serving_engine.cache_bindings` and
    `Serving_engine.contract` reconstruction.
  - `WHY`: The engine currently assumes the LFM layer sequence, placeholder
    prefix, output names, vocabulary, maximum positions, and direct LFM
    specializer despite those facts being knowable at compile time.
  - `FIX`: Resolve prefill and decode runtimes through Model Program entrypoint
    artifact roles; validate schedule metadata against the declared contract;
    bind attention/recurrent state by declared names; initialize RoPE and cache
    from declared dimensions; and call `Serving_specialization` with the
    declared plan.
  - `QUALITY`: Preserve current cache lease/rollback behavior, recurrent and
    attention packing order, prebaked decode behavior landed in `f048a63`,
    schedule caches, output selection, and contextual errors. Keep the engine
    independent of the compile-time `Lfm25_program` adapter.
  - `DO NOT`: Do not modify `lib/metal_runtime.ml`,
    `lib/metal_runtime.mli`, or `native/ocaml_metal_stubs.m`; do not change
    dispatch ordering, kernel parameters, cache packing, or introduce a second
    runtime adapter registry.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run
    `ninja -f ninja.build test all` followed by
    `if rg -n 'Lfm25|l_kwargs_' lib/serving_engine.ml lib/serving_engine.mli; then exit 1; fi`.
  - `DONE WHEN`: The shared engine validates and executes solely from Model
    Program plus resolved packages, the named grep produces no matches, and all
    existing tests/build targets pass without edits to the Metal runtime slice.
  - `ESCALATE IF`: Current or new uncommitted work overlaps
    `lib/serving_engine.ml`, or preserving the `f048a63` prebaked/cache-pack
    sequence requires a Metal runtime change; stop this item and repair its
    boundary rather than overwriting or expanding scope.

- [x] **ITEM-07: Migrate LFM entrypoints to load the root program**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Existing LFM server, generator, and static serving checker call
    sites.
  - `IMPORTANT FILES`:
    - `bin/lfm_serve.ml`: Load `model.llmopt`, resolve its tokenizer and
      prefill/decode entrypoints, create cache config from its state plan, and
      construct the generic engine.
    - `bin/lfm_generate.ml`: Use the same root loader for command-line
      generation.
    - `bin/lfm_serving_check.ml`: Delegate root artifact checks to
      `Model_program_check`/shared validation and retain LFM-specific static
      schedule observations only where useful.
    - `lib/serving_cache.ml` and `lib/serving_cache.mli`: Remove the temporary
      `model:Lfm25.Config.t` constructor/accessor after the final caller
      migrates.
    - `ninja.build`: Update link dependencies and usage text while retaining the
      existing executable names for this milestone.
  - `IMPORTANT SYMBOLS`: `load`, `package`, `Generation.create`,
    `Serving_cache.Config.of_state_plan`, `Serving_engine.create`,
    `Serving_engine.validate_program`.
  - `WHY`: A root Model Program is not the runtime authority while callers
    continue to inject `Lfm25.Config.default` and infer package paths.
  - `FIX`: Make the engine-directory form load only `model.llmopt`; use its
    canonical artifact references for tokenizer and packages. Keep LFM chat
    construction and the existing HTTP/generation behavior at the integration
    edge. Remove the legacy three-positional-path form if it cannot supply a
    Model Program without reconstructing metadata.
  - `QUALITY`: Share one loader/validator path between server, generator, and
    checker. Resolve all artifacts relative to the engine root and report the
    declared role on failures.
  - `DO NOT`: Do not rename the public binaries, generalize `Lfm_chat`, alter
    OpenAI protocol types, change queue behavior, or add a model-family switch
    to the shared engine.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run
    `ninja -f ninja.build test all` and
    `if rg -n 'Lfm25' lib/serving_cache.ml lib/serving_cache.mli lib/serving_engine.ml lib/serving_engine.mli; then exit 1; fi`.
  - `DONE WHEN`: All shipped serving/generation/checker binaries compile through
    the root-program loader, no shared cache or engine source references LFM,
    and LFM-specific chat behavior remains confined to LFM integration files.
  - `ESCALATE IF`: A caller needs information absent from `model.llmopt`; add it
    through the Model Program and LFM adapter with tests, not as a new caller
    default or path convention.

- [x] **ITEM-08: Review the cumulative contract migration and synchronize OKF**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Fresh-context cumulative review, real saved-package smoke, and
    durable architecture evidence.
  - `IMPORTANT FILES`:
    - `LOOP-model-program-execution-contract.md`: Record per-item commits and
      final evidence; repair any stale file/symbol references found in review.
    - `.okf/decisions/model-program-boundary.md`: Update implementation evidence
      and mark stable only if its written status transition is satisfied.
    - `.okf/architecture.md`: Update package layout and ownership to distinguish
      root Model Program from per-entrypoint serving packages.
    - `.okf/index.md`, `.okf/decisions/index.md`, and `.okf/log.md`: Keep bundle
      navigation and history current.
    - All production/test paths touched by ITEM-01 through ITEM-07: Review the
      cumulative diff from a compacted or fresh context.
  - `IMPORTANT SYMBOLS`: `N/A - cumulative review and documentation item`.
  - `WHY`: The migration crosses compiler/linker/runtime ownership boundaries,
    so local passing tests do not alone reveal duplicated authority or a stale
    LFM dependency.
  - `FIX`: Review the cumulative diff for original intent, contract duplication,
    model-family leakage, path validation, ABI error handling, rollback safety,
    and unexplained files. Build a fresh engine from the saved current LFM
    inputs and validate the resulting root program without launching a device
    workload. Record exact results and synchronize the governing OKF documents.
  - `QUALITY`: Keep claims separated into static contract evidence and runtime
    behavior preserved by existing tests. Report any measured timing only as an
    observation; this item does not request a benchmark run.
  - `DO NOT`: Do not implement quant-format, processor/chat, Gemma, multimodal,
    or performance work during review. Do not stage unrelated artifacts or
    pre-existing work.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run:

    ```sh
    ninja -f ninja.build test all
    _build/bin/llmopt-pipeline \
      --weights _artifacts/w4-engine-megakernels/weights.llmopt \
      --tokenizer _artifacts/w4-engine-megakernels/tokenizer.llmopt \
      --prefill _artifacts/w4-engine-megakernels/prefill/graph.llmopt \
      --decode _artifacts/w4-engine-megakernels/decode/graph.llmopt \
      --output _build/model-program-lfm-smoke
    _build/bin/llmopt-model-check _build/model-program-lfm-smoke
    if rg -n 'Lfm25|l_kwargs_' \
      lib/serving_cache.ml lib/serving_cache.mli \
      lib/serving_engine.ml lib/serving_engine.mli; then exit 1; fi
    uv run /Users/tung/.agents/skills/validate/scripts/okf_validate.py .okf --strict
    git diff --check
    ```

  - `DONE WHEN`: The commands complete with a valid root program and two
    resolved serving entrypoints, shared cache/engine grep is empty, strict OKF
    validation succeeds, and a fresh-context review finds no unexplained file,
    duplicate authority, model-family branch, or regression against the written
    milestone.
  - `ESCALATE IF`: The saved `_artifacts/w4-engine-megakernels` fixture is absent
    or no longer matches the current package ABI; select one existing current
    W4 engine only after recording its exact provenance in this item, rather
    than recapturing a model or substituting an older Q8-weight artifact.

## Later

- Plan a separate LOOP for physical tensor-format descriptors and checkpoint
  import adapters, including current group-64 PTQ and Gemma QAT group-32 paths.
- Plan a separate LOOP for processor/chat-template assets, multimodal
  entrypoints, and a model-neutral serving binary.
- Plan a separate LOOP for the high-level Python compilation session that turns
  a PyTorch model plus processor and serving calls into the complete Model
  Program without manually supplied graph paths.
- Add Gemma as the second Model Adapter only after those contracts exist; use it
  to expose remaining shared-core assumptions rather than adding Gemma branches
  to runtime modules.
