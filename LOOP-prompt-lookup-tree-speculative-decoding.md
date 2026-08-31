# LOOP-prompt-lookup-tree-speculative-decoding: Weightless Prompt-Lookup (N-Gram) Tree Speculative Decoding

## GOAL
Implement and benchmark **Weightless Prompt-Lookup (N-Gram) Tree Speculative Decoding** in LLMOpt on Apple Silicon GPU.
Prompt-lookup speculative decoding requires **zero additional weights or assistant models**, using n-gram pattern matching over the context/prompt history to propose candidate token sequences ($K \in [2, 6]$) and verifying them in a single target forward pass using Metal tree-attention masks.

## OUT OF SCOPE
- Training or fine-tuning neural draft models (Medusa head weights / Eagle / MTP).
- Modifying offline GGUF dequantizers or existing AOT megakernel fusions.

## RELEVANT FILES
- `lib/serving_engine.ml`: Serving engine, speculative acceptance, Medusa tree module, and new `Prompt_lookup` module.
- `lib/serving_engine.mli`: Interface signatures for `Prompt_lookup` and speculative coordination.
- `test/test.ml`: OCaml unit tests for n-gram matching, candidate tree generation, and acceptance verification.
- `bench/bench_prompt_lookup_speculative.py`: Sustained generation benchmark measuring real prompt-lookup speedup and token parity.
- `python/tests/test_bench_prompt_lookup_speculative.py`: Python unit tests for benchmark harness.
- `.okf/decisions/prompt-lookup-tree-speculative-decoding.md`: OKF decision record and mathematical formulation.

## SUPPORTING DOCUMENTS
- `.okf/decisions/prompt-lookup-tree-speculative-decoding.md`: `CREATED` - Mathematical formulation, complexity, and contract for weightless n-gram tree speculative decoding.

## COMPLETE WHEN
1. `Serving_engine.Prompt_lookup` is implemented in `lib/serving_engine.ml` and exported in `lib/serving_engine.mli`.
2. Unit tests in `test/test.ml` pass 100% via `ninja -f ninja.build test all`.
3. `bench/bench_prompt_lookup_speculative.py` and unit tests verify n-gram proposal, tree verification, and exact text output parity.
4. All changes committed cleanly and documented in the `.okf/` bundle.

---

### [ ] ITEM-01: Implement `Serving_engine.Prompt_lookup` N-Gram Matcher
- **REPO**: `/Users/tung/Projects/std23/llmopt`
- **WHERE**: `lib/serving_engine.ml` and `lib/serving_engine.mli`
- **IMPORTANT FILES**: `lib/serving_engine.ml`, `lib/serving_engine.mli`
- **IMPORTANT SYMBOLS**: `Serving_engine.Prompt_lookup.find_candidate_ngram`, `Serving_engine.Prompt_lookup.propose_linear`, `Serving_engine.Prompt_lookup.propose_tree`
- **WHY**: Weightless speculative decoding requires fast n-gram matching against token history to generate candidate draft tokens.
- **FIX**:
  1. Implement `find_candidate_ngram ~min_ngram ~max_ngram ~max_draft_tokens ~tokens` scanning backwards through `tokens` to match the suffix of length $n \in [\text{max\_ngram}, \text{min\_ngram}]$.
  2. Implement `propose_linear` returning candidate token arrays.
  3. Implement `propose_tree` building a `Medusa.tree` from candidate n-grams.
- **QUALITY**: Zero-allocation scanning loop, bounded time complexity $O(N \cdot n)$, graceful fallback to empty proposal when no match exists.

---

### [ ] ITEM-02: Add OCaml Unit Tests for `Prompt_lookup`
- **REPO**: `/Users/tung/Projects/std23/llmopt`
- **WHERE**: `test/test.ml`
- **IMPORTANT FILES**: `test/test.ml`
- **IMPORTANT SYMBOLS**: `test_prompt_lookup_matching`, `test_prompt_lookup_tree_integration`
- **WHY**: Need explicit test coverage for exact suffix matching, multiple match disambiguation, boundary conditions (empty context, suffix at end), and tree conversion.
- **FIX**: Add tests covering single matches, longest-match preference, missing matches, and end-to-end integration with `Serving_engine.Medusa.run`.
- **QUALITY**: `eval $(opam env) && ninja -f ninja.build test all` passes with 100% green.

---

### [ ] ITEM-03: Implement Python Benchmark Harness & Unit Tests
- **REPO**: `/Users/tung/Projects/std23/llmopt`
- **WHERE**: `bench/bench_prompt_lookup_speculative.py` and `python/tests/test_bench_prompt_lookup_speculative.py`
- **IMPORTANT FILES**: `bench/bench_prompt_lookup_speculative.py`, `python/tests/test_bench_prompt_lookup_speculative.py`
- **IMPORTANT SYMBOLS**: `PromptLookupMatcher`, `bench_prompt_lookup_sustained`
- **WHY**: Need an end-to-end reproducible benchmark harness to measure prompt-lookup acceptance rates, drafting latency, and speedup on real repetitive/code/RAG prompt benchmarks.
- **FIX**:
  1. Implement Python reference n-gram matcher and speculative runner.
  2. Write unit tests in `python/tests/test_bench_prompt_lookup_speculative.py`.
- **QUALITY**: Strict execution tracking, atomic JSON receipt output, zero fake/hardcoded numbers.

---

### [ ] ITEM-04: Document OKF Decision and Baseline Integration
- **REPO**: `/Users/tung/Projects/std23/llmopt`
- **WHERE**: `.okf/decisions/prompt-lookup-tree-speculative-decoding.md`, `.okf/decisions/index.md`, `.okf/goal-serving-runtime.md`
- **IMPORTANT FILES**: `.okf/decisions/prompt-lookup-tree-speculative-decoding.md`, `.okf/decisions/index.md`, `.okf/goal-serving-runtime.md`
- **WHY**: Synchronize OKF bundle with the weightless speculative decoding design, equations, and verified test evidence.
- **FIX**: Create decision record and update goal document.
- **QUALITY**: Consistent with Model Program ABI and verified test suite.
