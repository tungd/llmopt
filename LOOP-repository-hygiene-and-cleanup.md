# LOOP: Repository Hygiene, Dead Code Elimination, and Build Consistency

## Goal

Perform a comprehensive cleanup and hygiene pass across the `llmopt` repository: audit and eliminate dead code
in `lib/` and `python/`, reconcile stale build artifacts and `.okf` indices, verify ninja build rules and warning cleanliness,
and ensure submodule integration (`vendor/ocannl`) builds deterministically without drift.

---

## Out of scope

- Implementing new compiler features or new kernel optimizations.
- Deleting probe fixtures (`Lfm25_probe`) required by regression tests.
- Re-architecting active GGUF or Model Program execution ABIs.
- Modifying upstream `vendor/ocannl` code.

---

## Relevant files

- `ninja.build`: Ninja build orchestrator defining OCaml, C++, Metal, and Python test targets.
- `lib/ir.ml`, `lib/passes.ml`, `lib/metal.ml`: Core compiler sources to audit for unused symbols and deprecated variants.
- `python/llmopt_backend/`: Python Dynamo/FX capture library to audit for dead code and type annotations.
- `bench/`: Benchmark scripts and trace runners to audit for stale receipt paths and unused helpers.
- `.okf/index.md`, `.okf/tracking.md`, `.okf/log.md`: Knowledge federation bundle indices and research logs.
- `.gitignore`: Repository ignore patterns for build artifacts, temporary caches, and metallib binaries.

---

## Supporting documents

| Document | Path | Status | Purpose |
|---|---|---|---|
| ADR: Model Program Boundary | `.okf/decisions/model-program-boundary.md` | `EXISTING` | Governs clean separation of generic serving vs. model probes. |
| ADR: GGUF & Unsloth UD Quantization | `.okf/decisions/gguf-unsloth-dynamic-quantization.md` | `EXISTING` | Defines active weight format vs. legacy Q8 paths. |
| Prior Art: OCANNL & arrayjit | `.okf/prior-art/ocannl-arrayjit.md` | `EXISTING` | Governs `vendor/ocannl` integration boundary. |

---

## Complete when

1. `ninja -f ninja.build all test` compiles cleanly with zero OCaml compiler warnings and 100% test pass rate.
2. Unused dead code, unreferenced internal helpers, and abandoned experimental stubs in `lib/` are audited and removed.
3. `.gitignore` comprehensively covers `_build/`, `_artifacts/`, `.pytest_cache/`, and local temporary engine directories.
4. `.okf/index.md` and `.okf/tracking.md` accurately reflect all current ADRs, prior art, probe models, and active LOOP milestones.
5. All 7 repository LOOP files have consistent status, linked supporting documents, and verifiable next actions.

---

## Execution items

- [ ] **ITEM-01: Audit and Update `.gitignore` and Clean Stale Build Artifacts**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Repository root `.gitignore` and build directories.
  - `IMPORTANT FILES`:
    - `.gitignore`: Add missing patterns for transient Python caches (`.pytest_cache/`, `__pycache__/`), temporary engine outputs (`/tmp/repro-*`), and native build intermediates.
    - `ninja.build`: Ensure build output directories (`_build/ocaml/`, `_build/bin/`, `_build/native/`) are cleanly reproducible.
  - `IMPORTANT SYMBOLS`: `N/A - build-and-vcs-hygiene`
  - `WHY`: Accumulation of transient caches and test artifacts clutters `git status` and risks committing temporary binaries.
  - `FIX`: Audit repository untracked files; add comprehensive ignore rules for Python bytecode, pytest metadata, and temporary engine bundles; clean transient scratch files.
  - `QUALITY`: Do not ignore legitimate test fixtures in `test/` or reference traces in `bench/traces/`.
  - `DO NOT`: Do not delete committed OKF receipts or reference benchmark JSONs in `bench/results/`.
  - `VERIFY`: Run `git status --ignored` from `/Users/tung/Projects/std23/llmopt` and verify no transient caches appear in untracked status.
  - `DONE WHEN`: `git status` shows only tracked and intentional untracked files, and `.gitignore` covers all transient build outputs.
  - `ESCALATE IF`: Test fixtures rely on untracked temporary directories; make fixture paths explicit.

- [ ] **ITEM-02: Compiler Warning Audit and Strict Flag Cleanliness**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: OCaml compilation flags in `ninja.build` and source modules in `lib/`.
  - `IMPORTANT FILES`:
    - `ninja.build`: Enable strict warning checks (`-warn-error +a-4-29-40-42-44-48-70`) to catch unused variables, unhandled match cases, and deprecated constructs.
    - `lib/*.ml`: Fix any compiler warnings surfaced by the strict flags (e.g. unused `let` bindings, dead pattern cases).
  - `IMPORTANT SYMBOLS`: `ocaml_flags` in `ninja.build`.
  - `WHY`: Unchecked compiler warnings mask silent defects, unused variables, and drift across OCaml modules.
  - `FIX`: Add `-warn-error` flags to `ocaml_flags` in `ninja.build`; resolve all resulting compiler warnings across `lib/` and `test/`.
  - `QUALITY`: Preserve all `.mli` interface contracts; do not silence warnings by adding global `-w -a`.
  - `DO NOT`: Do not disable warnings for entire modules to bypass fixing dead variables.
  - `VERIFY`: Run `ninja -f ninja.build all test` from `/Users/tung/Projects/std23/llmopt`.
  - `DONE WHEN`: Complete build and test suite compile with zero warnings under strict warning-as-error flags.
  - `ESCALATE IF`: A third-party library interface generates unavoidable warnings; document exact disabled warning code in `ninja.build`.

- [ ] **ITEM-03: Audit and Clean Dead / Deprecated Code in `lib/`**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Compiler and runtime modules in `lib/`.
  - `IMPORTANT FILES`:
    - `lib/ir.ml`, `lib/ir.mli`: Audit unused legacy opcode constructors and dead helpers.
    - `lib/passes.ml`, `lib/passes.mli`: Ensure pass registration is clean and unreferenced passes are documented or pruned.
    - `lib/metal.ml`: Audit unused MSL helper functions and obsolete shader string variants.
    - `test/test.ml`: Verify that tests compile cleanly against the cleaned interfaces.
  - `IMPORTANT SYMBOLS`: `Ir.Op.t`, `Passes.all_passes`, `Metal.emit_source`.
  - `WHY`: Rapid development across multiple milestones leaves abandoned experimental helper functions and stale comments.
  - `FIX`: Grep for unreferenced private functions in `lib/` and remove dead code while preserving public `.mli` contracts.
  - `QUALITY`: Run `ninja -f ninja.build test` after each module edit to verify zero behavioral regressions.
  - `DO NOT`: Do not delete probe fixtures (`Lfm25_probe`) or GGUF dequantizers used by regression benchmarks.
  - `VERIFY`: Run `ninja -f ninja.build test` from `/Users/tung/Projects/std23/llmopt`.
  - `DONE WHEN`: All modules in `lib/` are free of unreferenced private helpers and all unit tests pass.
  - `ESCALATE IF`: A function appears unused in OCaml but is called from C/Objective-C stubs; verify with `grep -rn` across `native/`.

- [ ] **ITEM-04: Audit Python Backend and Benchmark Suite Cleanliness**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Python capture packages and benchmark harnesses in `python/` and `bench/`.
  - `IMPORTANT FILES`:
    - `python/llmopt_backend/`: Audit for unused imports, Python 3.12+ syntax compatibility, and clean exception handling.
    - `bench/`: Verify all benchmark runners (`llama_cpp_server_bench.py`, `gguf_fx_parity.py`, `lfm25_benchsuite.py`) execute with `--help` and follow standard JSON output schemas.
    - `ninja.build`: Ensure `python_test` runs all active test suites in `python/tests/`.
  - `IMPORTANT SYMBOLS`: `llmopt_backend`, `QuantizationTest`, `ManifestTest`, `FxGraphBinaryTest`.
  - `WHY`: Python backend scripts must remain cleanly decoupled from model-specific hardcoding and maintain strict typing.
  - `FIX`: Run `python3 -m unittest discover -s python/tests` and fix any deprecation warnings, stale imports, or unhandled exceptions.
  - `QUALITY`: Maintain formatting cleanliness and type annotations across all Python modules.
  - `DO NOT`: Do not break compatibility with PyTorch 2.6+ Dynamo FX capture interfaces.
  - `VERIFY`: Run `ninja -f ninja.build test` from `/Users/tung/Projects/std23/llmopt`.
  - `DONE WHEN`: All 64 Python unit tests pass cleanly in $< 1.5\text{s}$ with zero warnings.
  - `ESCALATE IF`: PyTorch Dynamo capture interface changes in upstream Python environment; update `python/llmopt_backend/` accordingly.

- [ ] **ITEM-05: Reconcile `.okf` Knowledge Bundle and Research Tracking**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: OKF knowledge documents in `.okf/`.
  - `IMPORTANT FILES`:
    - `.okf/index.md`: Verify all ADRs, prior art notes, and benchmark reports are cataloged with accurate status descriptions.
    - `.okf/tracking.md`: Update open research questions, completed slices, and current milestone targets.
    - `.okf/log.md`: Record the hygiene, vendoring, and arrayjit architectural milestones.
  - `IMPORTANT SYMBOLS`: `N/A - documentation-reconciliation`
  - `WHY`: The OKF bundle is the durable authority for the project; stale links or out-of-date tracking documents mislead future development.
  - `FIX`: Audit every markdown file in `.okf/decisions/` and `.okf/prior-art/`, verifying file paths, cross-links, and status fields (`approved`, `stable`, `deprecated`). Update `.okf/tracking.md` to reflect the active roadmap.
  - `QUALITY`: Ensure zero broken markdown links across all `.okf` documents.
  - `DO NOT`: Do not delete historical experiment receipts or retrospective logs.
  - `VERIFY`: Run a link-check scan across `.okf/` to ensure all referenced files exist on disk.
  - `DONE WHEN`: `.okf/index.md` and `.okf/tracking.md` are completely synchronized with the repository state.
  - `ESCALATE IF`: An existing ADR contradicts current code behavior; update the ADR with a dated errata note.

- [ ] **ITEM-06: Cumulative Repository Integrity and Build Verification**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Full build graph and test suite across OCaml, C++, Metal, and Python.
  - `IMPORTANT FILES`:
    - `ninja.build`: Top-level build graph.
    - `README.md`: Ensure build instructions, CLI invocations, and benchmark claims match current code.
  - `IMPORTANT SYMBOLS`: `all`, `test`, `probe-lfm25`.
  - `WHY`: Final confirmation that all subsystems compile, link, and test cleanly from a fresh build state.
  - `FIX`: Perform a clean rebuild (`ninja -f ninja.build clean && ninja -f ninja.build all test`); verify `README.md` examples and commands.
  - `QUALITY`: Build must be 100% reproducible with zero manual intervention.
  - `DO NOT`: Do not leave uncommitted temporary files in the repository.
  - `VERIFY`: Run `ninja -f ninja.build clean && ninja -f ninja.build all test` from `/Users/tung/Projects/std23/llmopt`.
  - `DONE WHEN`: Clean rebuild passes 100% of native OCaml tests, Metal builds, and Python test suites.
  - `ESCALATE IF`: Clean build fails due to missing build-edge dependencies in `ninja.build`; add explicit dependencies.
