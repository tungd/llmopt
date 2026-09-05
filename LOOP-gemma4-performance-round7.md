# Gemma 4 Performance & Throughput Parity Loop - Round 7

## Goal

Close the remaining throughput and decode latency gap between `llmopt` and `llama.cpp` for Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal (M4 Pro) through three focused architectural optimizations:
1. **Barrier-Free Streaming Decode GEMV**: Eliminate threadgroup shared memory and barriers from parameterized decode linear codegen in `lib/metal_block_codegen.ml`, streaming activations directly from L1/L2 cache to cut GPU kernel execution latency.
2. **Ping-Pong Double-Buffered Asynchronous Queue**: Implement double-buffered token and output buffers in `native/ocaml_metal_stubs.m`, eliminating data races and re-enabling safe depth-2 pipelined `Ring_queue` execution in `lib/serving_engine.ml`.
3. **Prefill Schedule Caching & Warmup**: Warm up and cache target prompt prefill execution plans in `lib/serving_engine.ml`, slashing TTFT from 9.73 s to < 2.0 s.

Target performance profile on Apple M4 Pro:
- **Decode Target**: Push throughput from 2.83 tok/s toward 15–20+ tok/s (TPOT < 65 ms).
- **Prefill Target**: TTFT < 2.0 s on prompt-37.
- **Parity Invariant**: Maintain 100% bit-exact token parity on prompt `2,105,9731,...` generating `100, 45518, 107, 14947, 135383,...`.
- **Architectural Invariant**: Zero bespoke monolithic kernels. All quantized linear operators must be emitted parametrically from schema definitions in `lib/metal_block_codegen.ml`.

## Out of scope

- Writing handwritten monolithic kernels in `lib/metal.ml`.
- Downloading remote weights or GGUF files (always reuse local archive `_artifacts/gemma4-12b-qat-primitive-schedule-2026-09-03/source/target.weights.llmopt`).
- Multi-token speculative decoding / MTP assistant serving in this milestone.

## Relevant files

- `lib/metal_block_codegen.ml`: Parameterized block codegen for decode linear and QKV GEMV.
- `lib/kernel_abi.ml`: Work-sharing registration and tactic definitions.
- `lib/metal_runtime.ml`: Grid sizing and dispatch encoding.
- `native/ocaml_metal_stubs.m`: Asynchronous ring queue and command buffer submission.
- `lib/serving_engine.ml`: Engine decode loop, ring queue binding, and prefill warmup.
- `bin/bench_tokens.ml`: Token generation benchmark harness.
- `.okf/decisions/gemma4-round7-barrier-free-gemv-double-buffered-async-prefill-gemm.md`: Governing OKF decision record.

## Supporting documents

- `.okf/decisions/gemma4-round7-barrier-free-gemv-double-buffered-async-prefill-gemm.md`: `CREATED` - Architectural decision record for Round 7.
- `.okf/decisions/gemma4-round6-compiler-dag-fusion-splitk-and-kernel-retirement.md`: `EXISTING` - Round 6 architectural decision record.
- `.okf/experiments/exp-0151-gemma4-performance-round6-2026-09-05.md`: `EXISTING` - Round 6 benchmark receipt.
- `.okf/experiments/exp-0152-gemma4-performance-round7-2026-09-05.md`: `PENDING` - Round 7 benchmark receipt and experiment tracking document.

## Complete when

- All checklist items below are executed and checked.
- All unit and integration test suites pass (`ninja -f ninja.build all test`).
- 100% bit-exact parity is maintained across all 128 completion tokens matching golden sequence: `100, 45518, 107, 14947, 135383,...`.
- Decode throughput increases significantly and TTFT drops below 2.0 s.
- An immutable benchmark receipt is recorded in `bench/results/` and linked in `.okf/experiments/`.

---

## Checklist

### [x] G4P7-01: Barrier-Free Streaming Decode GEMV in `lib/metal_block_codegen.ml`
- **REPO**: `llmopt`
- **WHERE**: `lib/metal_block_codegen.ml`, `lib/kernel_abi.ml`, `lib/metal_runtime.ml`
- **IMPORTANT FILES**:
  - `lib/metal_block_codegen.ml`: `MODIFY` - In `emit_q4_0_linear_f16_x8`, remove the 4096-element chunking loop with `threadgroup half s_activation[4096]` and threadgroup barriers. Stream directly from `device const half* activation = input;`. Each SIMDgroup calculates 8 columns independently using `simd_sum` across 32 lanes, writing output directly.
  - `lib/kernel_abi.ml`: `MODIFY` - Update `eight_column_decode_quant_registration` to specify `threadgroup_sram_bytes = 0`, `split_k_partitions = 1`, and 8 SIMDgroups per threadgroup (64 columns per threadgroup).
  - `lib/metal_runtime.ml`: `MODIFY` - Update grid calculation for `_x8` tactics to `(n + 7) / 8` threadgroups.
- **IMPORTANT SYMBOLS**: `Metal_block_codegen.emit_q4_0_linear_f16_x8`, `Kernel_abi.eight_column_decode_quant_registration`, `Metal_runtime.dispatch_quant_linear`.
- **WHY**: Threadgroup barriers and shared-memory staging in `emit_q4_0_linear_f16_x8` stall execution units; activations (8–56 KB) fit naturally in M4 Pro L1/L2 cache and can be streamed barrier-free.
- **FIX**: Emit direct pointer loads from `input` with zero barriers and zero threadgroup SRAM.
- **QUALITY**: Keep generator parameterized across block quantization schemas.
- **DO NOT**: Write a bespoke MSL kernel in `lib/metal.ml`.
- **VERIFY**: `ninja -f ninja.build all test` passes.
- **DONE WHEN**: Generated Metal source compiles cleanly with zero threadgroup barrier calls in `emit_q4_0_linear_f16_x8`.
- **ESCALATE IF**: Cache thrashing on $K=28672$ degrades streaming performance.

### [x] G4P7-02: Double-Buffered Asynchronous Ring Queue in `native/ocaml_metal_stubs.m`
- **REPO**: `llmopt`
- **WHERE**: `native/ocaml_metal_stubs.m`, `lib/serving_engine.ml`
- **IMPORTANT FILES**:
  - `native/ocaml_metal_stubs.m`: `MODIFY` - In `llmopt_ring_queue`, allocate ping-pong double buffers for `token_buffer[2]` and `output_buffer[2]`. Command buffer $t$ reads `token_buffer[t % 2]` and writes `output_buffer[t % 2]`, eliminating data races when `pipeline_depth = 2`.
  - `lib/serving_engine.ml`: `MODIFY` - Re-enable `Ring_queue` as default execution path in `execute_static_decode`.
- **IMPORTANT SYMBOLS**: `llmopt_ring_queue`, `llmopt_worker_loop`, `Serving_engine.execute_static_decode`.
- **WHY**: Single `token_buffer` and `output_buffer` caused GPU/CPU data races when `pipeline_depth > 1`, forcing fallback to synchronous execution that incurred ~190 ms CPU encoding overhead per token.
- **FIX**: Double-buffer token input and output buffers across ping-pong pipeline slots.
- **QUALITY**: Preserve thread-safe completion signaling with POSIX mutex and condition variable.
- **DO NOT**: Break single-turn synchronous execution fallback.
- **VERIFY**: `ninja -f ninja.build all test` passes; verify 10-token test matches golden tokens without race conditions.
- **DONE WHEN**: `Ring_queue` executes with depth-2 pipelining without any buffer data race.
- **ESCALATE IF**: Multi-buffer allocation exceeds Metal resource limits.

### [x] G4P7-03: Prefill Schedule Caching & Warmup in `lib/serving_engine.ml`
- **REPO**: `llmopt`
- **WHERE**: `lib/serving_engine.ml`
- **IMPORTANT FILES**:
  - `lib/serving_engine.ml`: `MODIFY` - In `warmup_engine` or prefill caching, ensure target prompt token lengths (e.g., prompt-37) have their schedule and prebaked command plan pre-compiled and retained in `engine.prefill_executables`.
- **IMPORTANT SYMBOLS**: `Serving_engine.warmup_engine`, `Serving_engine.prefill`, `Serving_engine.prefill_schedule`.
- **WHY**: First-time prefill schedule generation and command binding for prompt-37 adds ~7.5 seconds of one-time compilation latency into the measured TTFT.
- **FIX**: Bind and cache the prefill plan during engine warmup so prompt evaluation executes immediately from GPU memory.
- **QUALITY**: Maintain dynamic sequence length support for arbitrary prompts.
- **DO NOT**: Hardcode fixed token counts into the core prefill pipeline.
- **VERIFY**: `_build/bin/llmopt-bench-tokens` reports TTFT < 2.5 s on prompt-37.
- **DONE WHEN**: Prompt-37 prefill plan is cached and TTFT drops below 2.0 s.
- **ESCALATE IF**: Memory plan workspace conflict prevents simultaneous prefill and decode plan retention.

### [x] G4P7-04: Recompile Round 7 Bundle and Verify 128-Token Bit-Exact Parity
- **REPO**: `llmopt`
- **WHERE**: `bin/pipeline.ml`, `_artifacts/gemma4-12b-qat-round7-2026-09-05`
- **IMPORTANT FILES**:
  - `_artifacts/gemma4-12b-qat-round7-2026-09-05`: `GENERATE` - Unified engine package compiled via `llmopt-pipeline`.
- **COMMAND**: Recompile engine bundle `_artifacts/gemma4-12b-qat-round7-2026-09-05` and run 128-token benchmark with `llmopt-bench-tokens --warmup`.
- **VERIFY**: All 128 completion tokens strictly match the pinned golden reference sequence:
  `100, 45518, 107, 14947, 135383, 568, 11124, 568, 147505, 236789, ...`.
- **DONE WHEN**: 128/128 tokens bit-exact match achieved with measured throughput increase.
- **ESCALATE IF**: Token sequence diverges at any index.

### [x] G4P7-05: Record Benchmark Receipt and OKF Tracking
- **REPO**: `llmopt`
- **WHERE**: `bench/results/uoc-gemma4-performance-round7-2026-09-05.txt`, `.okf/experiments/exp-0152-gemma4-performance-round7-2026-09-05.md`, `.okf/tracking.md`
- **IMPORTANT FILES**:
  - `bench/results/uoc-gemma4-performance-round7-2026-09-05.txt`: `CREATE` - Immutable execution receipt.
  - `.okf/experiments/exp-0152-gemma4-performance-round7-2026-09-05.md`: `CREATE` - Experiment tracking document.
  - `.okf/experiments/index.md`: `MODIFY` - Register exp-0152.
  - `.okf/tracking.md`: `MODIFY` - Update research register with Round 7 metrics.
- **DONE WHEN**: Receipt recorded, `ninja -f ninja.build all test` passes, git working tree clean, and all changes committed.
