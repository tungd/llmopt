# LOOP: Constructive Lowering via Symbolic Operator DSL and Algebraic Inlining

## Goal

Replace fragile AST subgraph pattern matching (`fusion_query.ml`, `pass_fuse_rms_norm.ml`, `pass_fuse_swiglu_ffn.ml`)
with a typed OCaml symbolic operator DSL (`Math_dsl`) and constructive algebraic loop inlining, proving
bit-exact parity on RMSNorm and SwiGLU while eliminating thousands of lines of brittle AST query debt.

---

## Out of scope

- Modifying GGUF weight ingestion, tensor unpacking, or `Weight_archive.Dtype`.
- Modifying serving runtime queueing, OpenAI protocol serialization, or HTTP handling.
- Altering macro-level sequence scheduling or Model Program execution contracts.
- Implementing training / backward autodiff passes.

---

## Relevant files

- `vendor/ocannl/arrayjit/lib/low_level.mli`, `vendor/ocannl/arrayjit/lib/low_level.ml`: Imperative scalar IR (`scalar_t`, `For_loop`, `virtual_llc`, `simplify_llc`).
- `vendor/ocannl/arrayjit/lib/assignments.ml`: Assignment representations and index projections.
- `lib/math_dsl.mli`, `lib/math_dsl.ml`: Strongly-typed OCaml symbolic operator DSL constructing scalar expression trees.
- `lib/pass_constructive_lower.mli`, `lib/pass_constructive_lower.ml`: Ingestion and algebraic inlining pass virtualizing intermediate tensors.
- `lib/pass_fuse_rms_norm.ml`: Legacy 686-line RMSNorm AST pattern matcher (target for retirement).
- `lib/pass_fuse_swiglu_ffn.ml`: Legacy 500-line SwiGLU AST pattern matcher (target for retirement).
- `lib/fusion_query.mli`, `lib/fusion_query.ml`: Legacy AST subgraph query engine.
- `test/test_constructive_lowering.ml`: Unit test suite verifying symbolic AST construction, algebraic inlining, and bit-exact kernel output.
- `ninja.build`: Build configuration registering new compiler modules and test targets.

---

## Supporting documents

| Document | Path | Status | Purpose |
|---|---|---|---|
| ADR: Constructive Lowering & Symbolic DSL | `.okf/decisions/constructive-lowering-symbolic-operator-dsl.md` | `APPROVED` | Authoritative decision defining the symbolic DSL and algebraic inlining architecture. |
| ADR: OCANNL arrayjit Integration | `.okf/decisions/ocannl-arrayjit-loop-lowering-integration.md` | `APPROVED` | Framework decision for loop-nest lowering and multi-backend codegen. |
| Prior Art: OCANNL & arrayjit | `.okf/prior-art/ocannl-arrayjit.md` | `STABLE` | In-depth analysis of `arrayjit`'s virtualization and inlining passes. |

---

## Complete when

1. `Math_dsl` provides overloaded infix operators (`+!`, `*!`, `-!`, `/!`, `**!`) and unary intrinsics (`rsqrt`, `silu`, `sigmoid`) constructing typed scalar expressions.
2. `Pass_constructive_lower` ingests PyTorch FX operations into symbolic trees and inlines single-use intermediate activations into consumer loop nests with zero intermediate memory allocation.
3. RMSNorm and SwiGLU operations lower cleanly via constructive inlining without invoking `fusion_query.ml`.
4. Unit tests confirm **max absolute difference = 0.0** against PyTorch eager and legacy `metal.ml` reference kernels.
5. Legacy AST pattern matching for RMSNorm and SwiGLU is safely retired.
6. `ninja -f ninja.build test` passes cleanly.

---

## Execution items

- [ ] **ITEM-01: Implement Typed OCaml Symbolic Operator DSL (`Math_dsl`)**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Core symbolic DSL in `lib/math_dsl.mli` and `lib/math_dsl.ml`.
  - `IMPORTANT FILES`:
    - `lib/math_dsl.mli`, `lib/math_dsl.ml`: Create `Math_dsl` exposing overloaded operators (`+!`, `*!`, `-!`, `/!`, `**!`), unary math functions (`rsqrt`, `silu`, `sigmoid`, `exp`, `tanh`), scalar constants, and tensor indexing (`get`).
    - `test/test_constructive_lowering.ml`: Unit test verifying that symbolic operator expressions build typed `Low_level.scalar_t` trees and evaluate accurately on scalar test inputs.
    - `ninja.build`: Register `_build/ocaml/math_dsl.cmx` in the build graph.
  - `IMPORTANT SYMBOLS`: `Math_dsl.( +! )`, `Math_dsl.( *! )`, `Math_dsl.rsqrt`, `Math_dsl.silu`, `Math_dsl.get`, `test_math_dsl_tree_construction`.
  - `WHY`: The compiler needs an expressive, strongly-typed OCaml vocabulary to model tensor math equations without verbose AST record construction.
  - `FIX`: Define infix operators that wrap `Low_level.Binop`, `Low_level.Unop`, `Low_level.Constant`, and `Low_level.Get` with precision tags (`Ops.Single_prec` / `Ops.Half_prec`).
  - `QUALITY`: Enforce strict type safety: prevent mixing incompatible precision types without explicit scalar cast operators.
  - `DO NOT`: Do not perform eager floating-point calculation inside operators; they must strictly construct symbolic AST expressions.
  - `VERIFY`: Run `ninja -f ninja.build test` from `/Users/tung/Projects/std23/llmopt` and verify AST construction unit tests pass.
  - `DONE WHEN`: `Math_dsl` compiles cleanly and unit tests confirm exact expression tree construction for standard compound formulas.
  - `ESCALATE IF`: Symbol naming conflict with standard `Base` or `Stdlib` operators; ensure explicit module scoping via `open Math_dsl`.

- [ ] **ITEM-02: Implement Algebraic Virtualization & Inlining Pass**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Loop lowering pipeline in `lib/pass_constructive_lower.mli` and `lib/pass_constructive_lower.ml`.
  - `IMPORTANT FILES`:
    - `lib/pass_constructive_lower.mli`, `lib/pass_constructive_lower.ml`: Implement `virtualize_intermediates` and `inline_scalar_trees` analyzing tensor use counts, tagging single-use intermediate buffers as `Virtual`, and substituting their `Math_dsl` trees into consumer loop bodies.
    - `test/test_constructive_lowering.ml`: Unit test validating that multi-step assignment sequences collapse into single loop nests with zero temporary buffer allocations.
    - `ninja.build`: Register `_build/ocaml/pass_constructive_lower.cmx` in the build graph.
  - `IMPORTANT SYMBOLS`: `Pass_constructive_lower.virtualize_intermediates`, `Pass_constructive_lower.inline_scalar_trees`, `test_algebraic_inlining`.
  - `WHY`: Fusion should happen automatically by replacing temporary memory writes with register-local scalar substitutions rather than by graph pattern matching.
  - `FIX`: Perform liveness analysis over the computation DAG. If a tensor node is consumed by only one downstream operation within the same block, mark its storage placement as `Virtual` and substitute its RHS scalar expression at the consumer's read site.
  - `QUALITY`: Preserve reduction boundaries: do not inline an expression across an unsynchronized parallel reduction boundary without private accumulators.
  - `DO NOT`: Do not allocate DRAM/global buffer slots in `Serving_memory_plan` for virtualized nodes.
  - `VERIFY`: Run `ninja -f ninja.build test` from `/Users/tung/Projects/std23/llmopt` and verify temporary buffer allocations drop to zero for fused test chains.
  - `DONE WHEN`: A 4-step sequence (`sqr -> sum -> rsqrt -> mul`) inlines into a single loop body with only input and output tensors remaining materialized.
  - `ESCALATE IF`: Multi-consumer intermediate tensors are encountered; retain materialization for nodes with use-count > 1 unless explicitly duplicated.

- [ ] **ITEM-03: Route RMSNorm Ingestion Through Constructive Lowering**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: RMSNorm lowering in `lib/fx_plan.ml` and `lib/pass_constructive_lower.ml`.
  - `IMPORTANT FILES`:
    - `lib/fx_plan.ml`: Translate PyTorch FX `aten.rms_norm` (or decomposed `pow/mean/rsqrt/mul` chains) into `Math_dsl` symbolic expressions.
    - `lib/pass_constructive_lower.ml`: Apply constructive inlining to lower RMSNorm into a single-pass fused reduction/scaling loop.
    - `test/test_constructive_lowering.ml`: Unit test comparing constructive RMSNorm output against `pass_fuse_rms_norm.ml` reference output across shapes `[1, 2048]`, `[4, 4096]`, `[8, 8192]`.
  - `IMPORTANT SYMBOLS`: `Fx_plan.lower_rms_norm_constructive`, `test_rmsnorm_constructive_parity`.
  - `WHY`: RMSNorm is the most frequent normalization operator; routing it through constructive lowering proves the end-to-end viability of the new pipeline.
  - `FIX`: Ingest decomposed FX nodes into `Math_dsl` (`inv_std = rsqrt (mean (sqr x) +! eps); out = x *! inv_std *! weight`). The inlining pass automatically emits the fused loop.
  - `QUALITY`: Guarantee bit-exact numerical parity ($max\_diff = 0.0$) against existing reference kernels.
  - `DO NOT`: Do not invoke `fusion_query.ml` or `pass_fuse_rms_norm.ml` for graphs processed by the constructive path.
  - `VERIFY`: Run `ninja -f ninja.build test` from `/Users/tung/Projects/std23/llmopt` and assert numerical delta is exactly 0.0.
  - `DONE WHEN`: RMSNorm compiles through constructive lowering and produces exact matching FP16 output across all test shapes.
  - `ESCALATE IF`: PyTorch FX includes an unrecognized decomposition variant; add missing scalar intrinsic to `Math_dsl`.

- [ ] **ITEM-04: Route SwiGLU & Activation Products Through Constructive Lowering**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: SwiGLU FFN lowering in `lib/fx_plan.ml` and `lib/pass_constructive_lower.ml`.
  - `IMPORTANT FILES`:
    - `lib/fx_plan.ml`: Ingest PyTorch FX `aten.silu(gate) * up` into `Math_dsl.silu gate *! up`.
    - `lib/pass_constructive_lower.ml`: Apply constructive inlining to fold the activation product directly into the Linear store-back epilogue.
    - `test/test_constructive_lowering.ml`: Unit test verifying bit-exact parity for SwiGLU FFN against `pass_fuse_swiglu_ffn.ml`.
  - `IMPORTANT SYMBOLS`: `Fx_plan.lower_swiglu_constructive`, `test_swiglu_constructive_parity`.
  - `WHY`: SwiGLU accounts for a major portion of feed-forward computation; constructive lowering eliminates the dedicated 500-line AST pattern matcher.
  - `FIX`: Ingest dual Gate/Up linear projections and their activation product into `Math_dsl`. Inlining folds `silu(gate) * up` into the writeback site without intermediate DRAM activation stores.
  - `QUALITY`: Ensure support for both standalone SwiGLU and epilogue-fused SwiGLU (where activation is fused directly into the down-projection input).
  - `DO NOT`: Do not allocate separate memory buffers for the intermediate `gate` or `up` activations when epilogue fusion is active.
  - `VERIFY`: Run `ninja -f ninja.build test` from `/Users/tung/Projects/std23/llmopt`.
  - `DONE WHEN`: SwiGLU compiles through constructive inlining with zero DRAM round-trips for intermediate gate activations and bit-exact numerical parity.
  - `ESCALATE IF`: Non-standard activation encountered (e.g. GeGLU or Squared ReLU); express via `Math_dsl` primitives.

- [ ] **ITEM-05: Cumulative Verification & Deprecation of Legacy Pattern Matchers**
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Compiler pass orchestration in `lib/passes.ml` and `lib/pass_fuse_*.ml`.
  - `IMPORTANT FILES`:
    - `lib/passes.ml`: Switch default pass pipeline to use `Pass_constructive_lower` for normalization and activation layers.
    - `lib/pass_fuse_rms_norm.ml`, `lib/pass_fuse_swiglu_ffn.ml`: Mark legacy pattern matchers as deprecated or retire if fully superseded.
    - `test/test.ml`: Run full end-to-end model compilation suite (SmolLM2, Qwen3, Gemma4) confirming all models compile with zero opaque fallback dispatches.
  - `IMPORTANT SYMBOLS`: `Passes.default_pipeline`, `test_full_model_constructive_lowering`.
  - `WHY`: Completing the migration requires verifying that full model graphs compile cleanly without legacy pattern-matching queries.
  - `FIX`: Configure `Passes.default_pipeline` to run `Pass_constructive_lower`. Validate full-model graph lowering for all probe models in the test suite.
  - `QUALITY`: Verify that no regressions occur in model serving throughput or memory plan allocation.
  - `DO NOT`: Do not leave unused dead code in active pass pipelines.
  - `VERIFY`: Run `ninja -f ninja.build all test` from `/Users/tung/Projects/std23/llmopt` and verify 100% test pass rate.
  - `DONE WHEN`: All probe models compile and execute with zero reliance on legacy RMSNorm/SwiGLU AST query matchers, with 100% test suite passage.
  - `ESCALATE IF`: Full-model execution exhibits numerical divergence; isolate offending layer and inspect inlined scalar AST.
