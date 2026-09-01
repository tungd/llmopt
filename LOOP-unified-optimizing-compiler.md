# Unified Optimizing Compiler Transition

## Goal

Replace the first coherent slice of operation-specific fusion discovery with a
typed constructive assignment/dependency/effect/cost model. The same analyzed
graph must produce dependent fusion regions and independent antichain stages as
separate results, select target-aware quant and launch tactics from the measured
hardware profile, and emit an inspectable AOT GPU command-queue receipt.

The final slice executes the parked Gemma 4 12B sequential target run exactly
once after its full static preflight. That receipt reports the observed
compiler coverage, queue, correctness, and performance results without adding a
new success threshold.

## Governing decisions

- `.okf/decisions/constructive-primitive-loop-lowering-and-gpu-queue-compilation.md`
- `.okf/architecture.md` section 3.2.1
- `.okf/decisions/measured-hardware-probing.md`
- `.okf/decisions/dag-co-scheduling-optimizer-pass.md`
- `.okf/decisions/gguf-unsloth-dynamic-quantization.md`
- `.okf/decisions/gemma-12b-qat-mtp-speculative-benchmark.md`

## Baseline and recovery anchors

- The exact pre-existing Gemma WIP is recoverable from branch
  `wip/gemma4-12b-unified-compiler-20260901` at commit `9047771`.
- The active implementation branch starts from documentation commit `7fbd817`.
- Detached commit `3729eda` and its ancestors are read-only experimental
  evidence. Do not merge or cherry-pick that divergent chain wholesale.
- Preserve Ninja, PyTorch MPS capture, Q8-first runtime behavior, the Model
  Program ABI, and every unrelated worktree change.

## Scope

In scope:

- exact logical/physical block-quant schema modeling for formats already
  accepted by `Ir.Linear_storage`;
- reusable consumer, fan-out, dependency, effect, and stage facts in
  `Compute_plan`;
- algorithmic region formation with typed acceptance/cut reasons;
- costed target tactics fed by `Target_hardware.t`;
- replacement of covered epilogue, gated-linear, and linear-residual discovery
  in the active pipeline, with explicit legacy fallback accounting for
  uncovered semantics;
- a structured optimization/AOT queue receipt;
- one sequential LLMOpt Gemma 4 12B build-and-generation run against the pinned
  target GGUF and the existing llama.cpp sequential evidence.

Out of scope:

- wholesale import of the detached arrayjit branch or its root `Ir` module;
- full removal of all `pass_fuse_*.ml` files or `metal.ml` in this loop;
- CUDA/HIP runtime implementation;
- Gemma MTP assistant serving or a speculative LLMOpt run;
- serving-request queue policy changes;
- a new benchmark threshold or reinterpretation of the measured result.

## Executor protocol

1. Before each item, re-read the governing decision and inspect the named
   interfaces, call sites, tests, and current worktree for drift.
2. Implement one unchecked item at a time with a failing focused test first.
3. Use module-scoped abstract domain types and variants for legality, boundary,
   tactic, and receipt facts. Do not expose mutable records or encode reasons as
   free-form strings.
4. Prefer a pure analysis core. Mutation and file IO remain owned by existing
   graph, pipeline, and artifact adapters.
5. Run the focused test, then `ninja -f ninja.build test`, then `git diff --check`.
6. Check the item, record its exact evidence and commit, commit the slice, and
   continue immediately.
7. Do not start the full Gemma build or generation command before UOC-07. UOC-07
   gets one full attempt: if it exits unsuccessfully, preserve the exact output,
   record the result, and stop without an improvised user-facing retry.

## Checklist

- [x] **UOC-00 — Park Gemma WIP and establish the governing architecture**

  - **Where:** `.okf/decisions/constructive-primitive-loop-lowering-and-gpu-queue-compilation.md`, `.okf/decisions/ocannl-arrayjit-loop-lowering-integration.md`, `.okf/decisions/constructive-lowering-symbolic-operator-dsl.md`, `.okf/architecture.md`, `.okf/goal-serving-runtime.md`, `.okf/index.md`, `.okf/log.md`.
  - **Change:** Preserve the Gemma code snapshot, rewrite the compiler decision
    around one assignment/dependency/effect/cost model, keep fusion regions and
    antichain stages distinct, clarify hardware as the probed cost model, retain
    target-aware quant tactics, and distinguish the AOT GPU queue from serving.
  - **Evidence:** `uv run /Users/tung/.codex/skills/validate/scripts/okf_validate.py .okf --strict` reports a conformant bundle; the only warning names an unrelated pre-existing lifecycle value.
  - **Commit:** `7fbd8172d86f6b838cf402a8308004f5f9d174f4` (`docs: define unified optimizing compiler model`). Parked code commit: `9047771`.

- [x] **UOC-01 — Model exact block-quant schemas without importing arrayjit**

  - **Where:** create `lib/block_schema.mli` and `lib/block_schema.ml`; update
    their explicit Ninja edges in `ninja.build`; extend `test/test.ml`.
  - **Symbols:** abstract `Block_schema.t`; nested `Physical_layout`,
    `Logical_decode`, and `Target_tactic` modules; total
    `Block_schema.of_storage : Ir.Linear_storage.layout -> (t, Error.t) result`
    covering `Block_quantized` and `Groupwise_packed` storage already accepted by
    `Ir.Linear_storage`.
  - **Change:** Represent block size, byte fields, packed bit ranges, scale/min
    semantics, signedness, alignment, logical decode, and available target
    tactics as one validated schema. Port the proven W4A16 and Q4_K layout facts
    from `git show 3729eda:lib/arrayjit_bridge.ml`; do not add an arrayjit package
    or rename the repository `Ir` module.
  - **Proof:** Focused tests cover Q4_K's 256-value/144-byte physical contract,
    W4 groupwise nibble/sign/scale semantics through its existing storage type,
    all existing block-quant dtypes, invalid dtype rejection, and stable tactic
    enumeration. Run the repository test target.
  - **Evidence:** `ninja -f ninja.build test` passes all 72 Python tests and the
    OCaml suite (`llmopt canonical W4A16/KVQ8 tests passed`); `git diff --check`
    passes. Focused assertions cover exact Q4_K fields and six-bit packed
    scale/minimum ranges, composed block/sub-block decode factors, W4's 64-value
    two-complement nibble layout plus separate F16 group scale, all seven
    `Ir.Dtype.quant_type` size contracts, typed rejection, and backend-specific
    target primitives, including the direct-integer versus packed primitive
    distinction.
  - **Commit:** `26274fb` (`feat: model exact block-quant schemas`); correction
    `9dc0a5d` (`fix: specialize block schema tactic primitives`).

- [x] **UOC-02 — Make dependency and use facts a single reusable authority**

  - **Where:** `lib/compute_plan.mli`, `lib/compute_plan.ml`, `test/test.ml`.
  - **Symbols:** add abstract `Compute_plan.Use.t` and
    `Compute_plan.Edge.t`; expose `producer`, ordered `consumers`, `use_count`,
    `edge`, and `has_path` queries derived from the existing immutable plan.
  - **Change:** Derive data/state edges, fan-out, use counts, effects, critical
    paths, and ready antichains from one plan snapshot. Preserve the existing
    conservative singleton-stage policy for non-pure requests.
  - **Proof:** Focused DAG tests cover a chain, a diamond/shared producer, an
    independent antichain, and state/effect ordering; existing co-scheduling
    tests remain unchanged and pass.
  - **Evidence:** `ninja -f ninja.build test` passes all 72 Python tests and the
    OCaml suite (`llmopt canonical W4A16/KVQ8 tests passed`); `git diff --check`
    passes. Focused assertions cover deterministic producer/use relationships,
    chain and diamond paths, independent ready antichains, duplicate declared
    plus embedded references collapsing to one use, typed
    `Data_and_state` edge aggregation, state/effect dependencies, and singleton
    non-pure stages. `Edge_map` is the immutable relation authority from which
    predecessor/successor indexes are derived, while producer and ordered-use
    indexes are captured once during `of_graph`.
  - **Commit:** `597287a` (`feat: centralize compute plan dependency facts`).

- [x] **UOC-03 — Form fusion regions algorithmically and report every cut**

  - **Where:** create `lib/constructive_lowering.mli` and
    `lib/constructive_lowering.ml`; add their explicit Ninja edges in
    `ninja.build`; extend `test/test.ml`.
  - **Symbols:** abstract `Assignment.t`, `Region.t`, and `Analysis.t`; variants
    `Boundary.t`, `Legality.t`, and `Decision.t`; function
    `analyze : target:Target_hardware.t -> Compute_plan.t -> Analysis.t`.
  - **Change:** Normalize eligible semantic nodes into assignments with source
    node provenance, then grow maximal producer-consumer regions using shared
    dependency/use/effect facts. Fusion candidates follow dependent edges or
    shared-input joins; concurrency stages remain the plan's antichains. Record
    typed reasons for effect, alias/state, fan-out retention, unsupported index
    maps, numerical constraints, and target resource cuts.
  - **Proof:** Focused tests show linear-plus-add and gate/up/Silu/multiply are
    found without fixed node IDs or adjacency assumptions; an independent Q/K/V
    antichain remains separate commands; stateful, opaque, and externally-used
    values produce explicit boundaries.
  - **Evidence:** `ninja -f ninja.build test` passes all 72 Python tests and the
    OCaml suite (`llmopt canonical W4A16/KVQ8 tests passed`); `git diff --check`
    passes. Focused assertions cover provenance-bearing `Kernel_ir.Primitive`
    assignments, a non-adjacent accepted Linear-to-Add edge whose immutable
    `Compute_plan.Edge` exists, gate/up/Silu/multiply rejoin formation, separate
    Q/K/V singleton regions in one ready antichain, graph-output and external
    fan-out boundaries, stateful and opaque effect cuts, unsupported index-map
    adjacency, malformed optional-bias numerical metadata, Linear-to-Linear
    inlining cuts, and nonempty boundaries for every `Cut`. Structural Input and
    Output edges are excluded from fusion candidates; `Legality.Illegal` carries
    a nonempty boundary at the type level. The analyzer retains the target while
    deriving regions and stages from immutable `Compute_plan` facts. It exposes
    typed `Target_resource` boundary vocabulary for the later cost/legal
    selection slice, but emits no fabricated resource cuts; target legality and
    cost begin in UOC-04.
  - **Commits:** `b8bc8ab` (`feat: form constructive fusion regions`),
    `7abc781` (`fix: validate RMSNorm fusion metadata`).

- [x] **UOC-04 — Cost legal quant and launch alternatives with the probed target**

  - **Where:** `lib/kernel_abi.mli`, `lib/kernel_abi.ml`, `ninja.build`,
    `test/test.ml`.
  - **Symbols:** `Kernel_abi.Linear_tactic.Candidate.t`,
    `Kernel_abi.Linear_tactic.Selection.t`, and
    `select_costed : target:Target_hardware.t -> ... -> Selection.t option`;
    retain `select` temporarily as an explicit legacy adapter.
  - **Change:** Enumerate legal candidates from `Block_schema`, calculate a
    named cost breakdown from probed SIMD/threadgroup/SRAM/bandwidth/dispatch
    facts, and choose by cost with deterministic tie-breaking instead of list
    order. Keep the chosen tactic's schema and cost evidence inspectable.
  - **Proof:** Tests use two constructed target profiles to demonstrate that
    legal alternatives and their costs are explicit and deterministic; existing
    tactic names and legacy callers remain executable.
  - **Evidence:** `ninja -f ninja.build test` passed with 72 Python tests and
    the OCaml suite; `git diff --check` passed. The default `ninja -f
    ninja.build` target linked all declared binaries. Tests cover exact
    SIMD/threadgroup/SRAM legality, schema and Metal tactic provenance,
    Q4_K launch/reuse facts, roofline composition, target geometry, low-SRAM
    rejection, deterministic tie-breaking, and unchanged legacy selection.
  - **Commit:** `1a71b1b` (`feat: add costed linear tactic selection`).

- [x] **UOC-05 — Route covered fusion through constructive regions**

  - **Where:** `lib/constructive_lowering.ml`, `lib/passes.mli`,
    `lib/passes.ml`, `lib/metal.ml`, `test/test.ml`.
  - **Symbols:** `Constructive_lowering.lower` and
    `Passes.Optimization.region_analysis`.
  - **Change:** Run dependency analysis before operation-specific epilogue and
    linear fusion. Lower accepted linear-residual and gated-linear regions
    through the typed region result into the existing executable IR/Metal ABI;
    “linear-activation” denotes the Linear-to-Silu edges inside that executable
    gated Linear/Silu/Mul region, not standalone activation lowering. Invoke
    existing matchers only for uncovered candidates and record those
    invocations as legacy fallbacks; leave other fusion families unchanged.
  - **Proof:** Existing epilogue/gated/linear-add graph tests pass through the
    unified path and assert source membership plus zero legacy fallbacks for
    covered cases. Unsupported shapes still execute through a named fallback
    and retain an explicit cut reason.
  - **Evidence:** `ninja -f ninja.build test` passed the OCaml suite and all 72
    Python tests; `ninja -f ninja.build` linked the default targets; and `git
    diff --check` passed. Focused tests cover non-adjacent Linear+Add,
    W4A16+Add, gated Linear/Silu/Mul boundaries, malformed fallback parity,
    independent Q/K/V antichains, exact costed Metal tactic plus generic
    package fallback, and preserved legacy registry selection.
  - **Commit:** `09a05c5` (`feat: route covered fusion through constructive regions`).

- [x] **UOC-06 — Emit the optimization and AOT command-queue receipt**

  - **Where:** `lib/passes.mli`, `lib/passes.ml`, `bin/pipeline.ml`,
    `test/test.ml`.
  - **Symbols:** nested abstract modules `Passes.Receipt` and
    `Passes.Queue_stage`; `Passes.Optimization.receipt`; deterministic
    `Passes.Receipt.to_yojson`.
  - **Change:** Record target profile, source-node membership, accepted/cut
    candidates and reasons, unified/legacy lowering counts, selected tactics and
    cost breakdowns, kernels, antichain stages, dependencies, and barriers.
    Write `optimization-receipt.json` beside each prefill/decode/verification
    package before Metal/Model Program linking completes.
  - **Proof:** JSON round-trip/snapshot tests distinguish a fused dependent
    region from independent commands in one stage and from a sequential barrier;
    pipeline tests or fixtures confirm the artifact is emitted deterministically.
  - **Evidence:** `ninja -f ninja.build test` passed the OCaml suite and all 72
    Python tests; `ninja -f ninja.build` linked the default targets; and `git
    diff --check` passed. Focused tests parse the canonical receipt, retain the
    target profile, accepted/cut decisions, aggregate accepted/cut/unified and
    total/named legacy counts, exact costed tactic and roofline composition,
    emitted ABI entries, fused source membership, independent Q/K/V commands in
    one concurrent stage, and sequential state stages with compiler barriers.
    `Passes.Optimization.receipt ~kernels` remains backend-neutral while the
    pipeline writes the exact cache-augmented Metal kernel list to
    `optimization-receipt.json` for every prefill/decode/verification package
    before package and metallib linking. The deterministic `Receipt.write_file`
    fixture parses back to the same JSON; no Gemma build or generation was run.
  - **Commit:** `41a96ec` (`feat: emit optimization queue receipts`).

- [ ] **UOC-07 — Execute and record the one-shot Gemma 4 12B transition run**

  - **Where:** `bench/build_gemma4_engine.py`, `bin/bench_tokens.ml`,
    `bench/results/`, a new `.okf/experiments/exp-*.md`,
    `.okf/experiments/index.md`, and `.okf/log.md` only as required to retain the
    exact run and receipt.
  - **Preflight:** Before the attempt, verify the git commit and clean intended
    diff, no competing Gemma/llama benchmark process, exact 6,716,356,800-byte
    target symlink, SHA-256
    `90fd44e29e0d7cffeb0fd00dc73cfdab9ed0b0e95306ecf7821ea634c940c370`,
    free disk for
    the augmented archive and engine, hardware profile, Python imports, Ninja
    test/build targets, package/model checks, and optimization receipt schema.
    Derive the existing 37-token prompt IDs from the packaged tokenizer without
    running model generation.
  - **Attempt:** Create one fresh empty output directory, run
    `bench/build_gemma4_engine.py` for the target prefill/decode engine, run the
    package/model checks, then invoke `llmopt-bench-tokens` once for 128 greedy
    tokens with the derived prompt IDs. Do not invoke MTP serving in this loop.
  - **Receipt:** Preserve command, exit status, timestamps, source commit,
    hardware profile, exact model identity, prompt IDs, optimization receipts,
    unified/legacy/cut counts, kernels/stages/dependencies/barriers, token output,
    correctness observation, TTFT, elapsed time, tok/s, and exact deltas from
    the existing llama.cpp sequential `27.94 tok/s` result. Record failure
    output just as exactly as success output and do not retry.
  - **Proof:** Run `git diff --check` and strict OKF validation after recording
    the result. The measured receipt, not a threshold invented by this loop,
    supplies the transition outcome.
  - **Commit:** Record after completion.

## Completion audit

This loop is complete when every item has an evidence entry and commit, the
repository tests and OKF validation results are recorded, and the single Gemma
attempt has an immutable receipt. Checklist completion means the experiment was
executed and reported; it does not convert an unfavorable measurement into a
favorable compiler result.
