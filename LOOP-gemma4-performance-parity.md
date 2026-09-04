# Gemma 4 Performance & Throughput Parity Loop

## Goal

Close the performance and throughput gap between `llmopt` and `llama.cpp` for Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal without expanding the kernel zoo.

Target performance profile on Apple M4 Pro:
- **Prefill Target**: Match or exceed `113.69 tok/s` (~325 ms for prompt-37; current baseline ~2.95 s, 9.1x gap).
- **Decode Target**: Match or exceed `27.37 tok/s` (~36.5 ms/tok; current baseline ~74.2 ms/tok, 2.0x gap).
- **Parity Invariant**: Maintain bit-exact token parity on prompt `2,105,9731,...` generating `100, 45518, 107, 14947, 135383,...`.
- **Architectural Invariant**: Avoid bespoke monolithic kernel zoo proliferation by relying on isolated dequant micro-ops, parameterized execution templates, and compiler DAG fusion.

## Out of scope

- Creating bespoke one-off monolithic kernels per quant format (e.g. `q4_0_llama_mv_add`, `q4_0_gated_llama_mv`).
- Model-specific hardcoded hacks outside of generic compiler IR passes and hardware profiles.
- Speculative assistant (MTP) serving in this milestone (this milestone is purely base sequential prefill + decode).
- Re-quantization of model weights or changes to the pinned GGUF checkpoint format.

## Relevant files

- `lib/metal.ml`: Metal code generation for dequant micro-ops, generic GEMV template ($M=1$), and tiled prefill GEMM ($M \ge 4$).
- `lib/metal_runtime.ml`: Metal runtime dispatch, grid sizing, buffer binding, and tactic selection.
- `lib/serving_engine.ml`: Engine orchestration for prefill and decode phases, memory plan binding, and token execution.
- `native/ocaml_metal_stubs.m`: Low-level Metal runtime stubs, `llmopt_ring_queue`, and command buffer dispatch loop.
- `lib/pass_fuse_linear_bias.ml`: Graph fusion passes for QKV and Gate-Up projections.
- `lib/serving_schedule.ml`: Schedule validation, metadata matching, and dispatch plan generation.
- `bin/pipeline.ml`: Pipeline compiler tool (`llmopt-pipeline`) for generating AOT engine packages.
- `.okf/decisions/gemma4-batched-prefill-and-pipelined-queue.md`: Governing OKF decision record.

## Supporting documents

- `.okf/decisions/gemma4-batched-prefill-and-pipelined-queue.md`: `CREATED` - Architectural decision rejecting the kernel zoo, specifying $M \ge 4$ prefill tiling, and specifying depth-2 queue pipelining.
- `.okf/decisions/parametric-epilogues-and-hardware-tiling.md`: `EXISTING` - Epilogue inlining and hardware tiling background.
- `.okf/decisions/llama-cpp-target.md`: `EXISTING` - Target parity definition against `llama.cpp`.
- `.okf/experiments/exp-0146-gemma4-performance-parity-2026-09-04.md`: `DRAFT` - Benchmark receipt and experiment tracking document to be created upon completing G4P-05.

## Complete when

- All 5 checklist items below are executed and checked.
- All unit and integration test suites pass (`ninja -f ninja.build all test`).
- Gemma 4 12B prefill TTFT drops from ~2.95s to < 450 ms (approaching llama.cpp 325 ms).
- Gemma 4 12B decode TPOT drops towards ~36.5 ms/tok.
- Output tokens match the pinned llama.cpp golden sequence: `100, 45518, 107, 14947, 135383`.
- An immutable benchmark receipt is recorded in `bench/results/` and linked in `.okf/experiments/`.

---

## Checklist

### [x] G4P-01: Reconcile Q4_0 lowering and clean up bespoke `_llama_mv` kernels
- **REPO**: `llmopt`
- **WHERE**: `lib/metal.ml`, `lib/metal_runtime.ml`, `lib/pass_fuse_linear_bias.ml`, `lib/serving_schedule.ml`
- **IMPORTANT FILES**:
  - `lib/metal.ml`: `MODIFY` - Clean up bespoke `_llama_mv` additions, preserve canonical dequant micro-ops and 4-column vector dequantization.
  - `lib/metal_runtime.ml`: `MODIFY` - Ensure Q4_0 uses generic quant linear dispatch without bespoke entrypoints.
  - `lib/pass_fuse_linear_bias.ml`: `VERIFY` - Keep QKV and Gate-Up fusion support for Q4_0.
  - `lib/serving_schedule.ml`: `VERIFY` - Keep metadata matching for fused Q4_0 operations.
- **IMPORTANT SYMBOLS**: `llmopt_q4_0_linear_f16_x4`, `fuse_quant_qkv`, `quant_qkv_linear_metadata_matches`.
- **WHY**: The user explicitly rejected expanding the bespoke kernel zoo. We must preserve clean compiler abstractions where dequantization is an orthogonal micro-op and fusion is handled via DAG transformations.
- **FIX**:
  1. Remove experimental one-off `_llama_mv` kernel variants from `lib/metal.ml`.
  2. Keep the vectorized 4-column unrolling inside canonical `llmopt_q4_0_linear_f16_x4` using SIMD `simd_sum` and clean nibble unpack.
  3. Ensure `fuse_quant_qkv` in `lib/pass_fuse_linear_bias.ml` and `lib/serving_schedule.ml` cleanly support Q4_0.
- **QUALITY**: Zero redundant handwritten kernels. Code compiles cleanly with no unused compiler warnings.
- **DO NOT**: Introduce bespoke handwritten monolithic kernels for specific quant combinations.
- **VERIFY**: Run `ninja -f ninja.build all test` from `/Users/tung/Projects/std23/llmopt`. Expected output: all 79 Python tests and OCaml tests pass.
- **DONE WHEN**: Git diff shows clean removal of `_llama_mv` zoo kernels, canonical Q4_0 vector unrolling is active, and `ninja -f ninja.build all test` exits 0.
- **ESCALATE IF**: Test suite fails or regressions in existing W4A16/KVQ8 tests occur.
- **DONE**: Commit `070a811` (`refactor(metal): reconcile Q4_0 lowering and clean up bespoke llama_mv kernels`). Removed experimental `_llama_mv` kernels from `lib/metal.ml`. Kept canonical 4-column vectorized unrolling and clean Q4_0 QKV/Gate-Up DAG fusions. Verified with `ninja -f ninja.build all test` (all 79 Python tests and OCaml tests passed).

---

### [x] G4P-02: Implement batched prefill GEMM ($M \ge 4$) in Metal code-gen
- **REPO**: `llmopt`
- **WHERE**: `lib/metal.ml`
- **IMPORTANT FILES**:
  - `lib/metal.ml`: `MODIFY` - Add parameterized small-$M$ batched GEMM template for quantized linear projections.
- **IMPORTANT SYMBOLS**: `llmopt_quant_linear_f16_batched`, `quant_tactic_entries`.
- **WHY**: Currently for prompt tokens $M=37$, prefill streams the 6.08 GB model weights 37 separate times from DRAM (~225 GB traffic, ~1.5–2.0 s purely in DRAM transfer). Tiling $M$ in threadgroup memory allows streaming weights once across the batch.
- **FIX**:
  1. Add a small-$M$ batched GEMM kernel template (`llmopt_q4_0_linear_f16_m_tiled` or generic quantized batched template) that stages $M$ tokens of activation in threadgroup SRAM.
  2. Have threads collaboratively load activations into threadgroup memory, then stream contiguous weight blocks once, accumulating across all $M$ output rows simultaneously.
  3. Register the batched tactic in `lib/metal.ml` for linear and fused gated linear operations when $M \ge 4$.
- **QUALITY**: Adheres to the compiler template model: inner loop uses the layout's standard dequantization formula, avoiding bespoke kernel proliferation.
- **DO NOT**: Hardcode $M=37$; parameterize by tile size $M_{\text{tile}} \in \{4, 8, 16, 32\}$ with loop bounds check for ragged tails.
- **VERIFY**: Run `ninja -f ninja.build all test` to ensure emitter compiles and generates valid MSL.
- **DONE WHEN**: Shader compilation succeeds, and generated prefill metallib contains the batched $M$-tiled kernel.
- **ESCALATE IF**: Threadgroup SRAM exceeds 32 KB per threadgroup or register pressure causes spilling.
- **DONE**: Commit `37fa5f5` (`feat(metal): implement batched prefill GEMM for Q4_0 linear, QKV, and gated operations`). Implemented `llmopt_q4_0_linear_f16_m4_add`, `llmopt_q4_0_gated_linear_f16_m4`, and `llmopt_q4_0_qkv_linear_f16_m4` with bounds checks. Preserved `_m4` entries in `quant_tactic_entries`. Verified with `ninja -f ninja.build all test` (all 79 Python and OCaml tests passed, including `xcrun metal` MSL compilation).

---

### [x] G4P-03: Wire batched prefill GEMM into compiler pipeline and runtime dispatch
- **REPO**: `llmopt`
- **WHERE**: `lib/metal_runtime.ml`, `lib/serving_engine.ml`, `bin/pipeline.ml`
- **IMPORTANT FILES**:
  - `lib/metal_runtime.ml`: `MODIFY` - Wire grid configuration and dispatch arguments for batched quantized linear when $M \ge 4$.
  - `lib/serving_engine.ml`: `MODIFY` - Ensure prefill execution paths bind the batched dispatch plan when $M > 1$.
  - `bin/pipeline.ml`: `VERIFY` - Ensure AOT compiler pipeline selects batched tactics for prefill graphs.
- **IMPORTANT SYMBOLS**: `dispatch_quant_linear`, `encode_schedule`, `compile_single_graph`.
- **WHY**: The batched prefill kernel must be invoked during prefill compilation and execution rather than falling back to per-token loops.
- **FIX**:
  1. In `lib/metal_runtime.ml`, add dispatch handler for `llmopt_q4_0_linear_f16_m_tiled` setting grid dimensions $((\text{cols} + \text{tile}_N - 1) / \text{tile}_N \times 32, (M + \text{tile}_M - 1) / \text{tile}_M, 1)$.
  2. Update `lib/serving_engine.ml` to pass the full prompt batch $M=37$ to the prefill execution plan.
  3. Verify AOT pipeline compiles prefill package with batched GEMM dispatches.
- **QUALITY**: Clean type-safe parameter passing and error propagation.
- **DO NOT**: Alter single-token decode dispatch logic ($M=1$).
- **VERIFY**: Run `_build/bin/llmopt-pipeline` to build prefill package and verify dispatch record count decreases while DRAM reads drop.
- **DONE WHEN**: Prefill package compiles cleanly with batched GEMM kernels and executes without error.
- **ESCALATE IF**: Metal validation layer reports invalid grid sizing or out-of-bounds buffer access.
- **DONE**: Commit `abb91f9` (`feat(metal): wire batched prefill GEMM into runtime dispatch and grid sizing`). Wired `Quant_qkv_tactic.kernel_name` and grid sizing for `_m4` QKV, and wired gated linear dispatch and grid sizing for `llmopt_q4_0_gated_linear_f16_m4` when $m \ge 4$. Linear and linear_add are routed to `llmopt_q4_0_linear_f16_m4(_add)` via `m4_quant_registration`. Verified with `ninja -f ninja.build all test` (all 79 Python and OCaml tests passed).

---

### [x] G4P-04: Implement pipelined decode execution in `Ring_queue`
- **REPO**: `llmopt`
- **WHERE**: `native/ocaml_metal_stubs.m`, `lib/metal_runtime.ml`, `lib/serving_engine.ml`
- **IMPORTANT FILES**:
  - `native/ocaml_metal_stubs.m`: `MODIFY` - Update `llmopt_worker_loop` to support depth-2 command buffer pipelining without blocking CPU.
  - `lib/metal_runtime.ml`: `MODIFY` - Expose queue pipeline depth configuration.
  - `lib/serving_engine.ml`: `MODIFY` - Support asynchronous decode submission via the ring queue.
- **IMPORTANT SYMBOLS**: `llmopt_worker_loop`, `llmopt_ring_submit`, `Prebaked.execute`.
- **WHY**: Currently `Prebaked.execute` and `llmopt_worker_loop` execute synchronously: `[cmd commit]; [cmd waitUntilCompleted];`, introducing CPU-GPU driver bubbles on every token. Metal's hardware queue can overlap step $t+1$ encoding/submission with step $t$ GPU execution.
- **FIX**:
  1. In `native/ocaml_metal_stubs.m`, pipeline `llmopt_worker_loop`: commit command buffer $t+1$ to the queue before waiting on completion of step $t$.
  2. Use GPU-resident token buffer for argmax so the GPU can begin the next step without waiting for host readback.
  3. Signal CQ completion asynchronously so host reads output tokens concurrently with ongoing generation.
- **QUALITY**: No race conditions on ring buffer indices; proper memory barrier acquire/release semantics.
- **DO NOT**: Break synchronous execution fallback for debug modes (`LLMOPT_PROFILE_BATCH=1`).
- **VERIFY**: Run decode benchmark and verify that GPU utilization is continuous and inter-token idle bubbles disappear.
- **DONE WHEN**: Queue worker achieves depth-2 pipelining, and inter-token decode latency drops towards target (~36.5 ms).
- **ESCALATE IF**: Data hazard or race condition detected in KV cache or output token buffer.
- **DONE**: Pipelined `llmopt_worker_loop` with depth-2 queue and GPU-resident token chaining (`llmopt_chain_token_id`). Exposed `Ring_queue.pipeline_depth` via `LLMOPT_QUEUE_DEPTH` env control. Wired `decode_executable.ring_queue` into `Serving_engine.execute_static_decode` and `reset`. Verified with `ninja -f ninja.build all test` (all 79 Python tests and OCaml tests passed).

---

### [ ] G4P-05: Benchmark full Gemma 4 12B sequential generation and record OKF receipt
- **REPO**: `llmopt`
- **WHERE**: `bench/results/`, `.okf/experiments/`, `.okf/tracking.md`
- **IMPORTANT FILES**:
  - `bench/results/uoc-gemma4-performance-parity-2026-09-04.txt`: `CREATE` - Benchmark execution stdout, TTFT, TPOT, and generated token IDs.
  - `.okf/experiments/exp-0146-gemma4-performance-parity-2026-09-04.md`: `CREATE` - Immutable experiment report comparing results with `llama.cpp`.
  - `.okf/tracking.md`: `MODIFY` - Update research register with the verified performance results.
- **IMPORTANT SYMBOLS**: `_build/bin/llmopt-bench-tokens`.
- **WHY**: Establish an immutable, reproducible record of performance gains, TTFT, TPOT, and token parity against `llama.cpp`.
- **FIX**:
  1. Re-build the optimized Gemma 4 engine bundle with the pipeline compiler.
  2. Run `_build/bin/llmopt-bench-tokens` for 32 generation tokens on prompt `2,105,9731,...`.
  3. Verify that first five tokens are exactly `100, 45518, 107, 14947, 135383`.
  4. Measure TTFT and TPOT and record full comparison table against llama.cpp target (`prefill 325 ms`, `decode 36.5 ms`).
  5. Document results in `bench/results/`, `.okf/experiments/exp-0146-*.md`, and `.okf/tracking.md`.
- **QUALITY**: Accurate, unvarnished reporting of actual hardware timings and token parity.
- **DO NOT**: Fabricate timings or claim parity beyond observed empirical evidence.
- **VERIFY**: `git diff --check`, `ninja -f ninja.build all test`, and verify that experiment file links resolve.
- **DONE WHEN**: All 5 checklist items are checked, tests pass, and the immutable experiment receipt is recorded.
- **ESCALATE IF**: Generated tokens diverge from golden sequence `100, 45518, 107, 14947, 135383`.
