# Gemma 4 Performance & Throughput Parity Loop - Round 5

## Goal

Close the remaining performance and throughput gap between `llmopt` and `llama.cpp` for Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal without expanding the kernel zoo.

Target performance profile on Apple M4 Pro:
- **Prefill Target**: Match or exceed `113.69 tok/s` (~325 ms for prompt-37; exp-0149: 1,231 ms GPU / 2,309 ms TTFT).
- **Decode Target**: Match or exceed `27.37 tok/s` (~36.5 ms/tok; exp-0149: 81.88 ms median / 76.23 ms steady-state; cursor: 69.71 ms).
- **Parity Invariant**: Maintain bit-exact token parity on prompt `2,105,9731,...` generating `100, 45518, 107, 14947, 135383,...`.
- **Architectural Invariant**: Avoid bespoke monolithic kernel zoo proliferation by relying on isolated dequant micro-ops, parameterized execution templates, and compiler DAG fusion.

## Out of scope

- Creating bespoke one-off monolithic kernels per quant format (e.g. `q4_0_llama_mv_add`, `q4_0_gated_llama_mv`).
- Model-specific hardcoded hacks outside of generic compiler IR passes and hardware profiles.
- Speculative assistant (MTP) serving in this milestone (this milestone is purely base sequential prefill + decode).
- Downloading remote weights or GGUF files (always reuse local archive `_artifacts/gemma4-12b-qat-primitive-schedule-2026-09-03/source/target.weights.llmopt`).

## Relevant files

- `lib/generation.ml`: Native generation driver and persistent sequence cursor routing (`Generation.Native_engine`).
- `native/ocaml_metal_stubs.m`: Low-level Metal command buffer and encoder execution (`llmopt_worker_loop`, `caml_llmopt_prebaked_execute`, `caml_llmopt_metal_batch_dispatch_all`).
- `lib/metal_block_codegen.ml`: Multi-SIMDgroup ($N_{\text{simd}}=2$, 64-thread) register-safe $M=32$ prefill GEMM (`llmopt_q4_0_linear_f16_m32`, `llmopt_q4_0_linear_f16_m32_add`).
- `lib/metal.ml`: Multi-SIMDgroup $M=32$ QKV linear kernel (`llmopt_q4_0_qkv_linear_f16_m32`).
- `lib/kernel_abi.ml`: Kernel ABI registration for `m32_quant_registration` with threshold `m >= 32`.
- `lib/metal_runtime.ml`: Runtime grid sizing, tactic selection, and dispatch routing for 64-thread $M=32$ prefill and barrier control.
- `bin/pipeline.ml`: Pipeline compiler tool (`llmopt-pipeline`) for generating AOT engine packages.
- `.okf/decisions/gemma4-round5-sequence-cursor-barrier-elision-multi-simd.md`: Governing OKF decision record.

## Supporting documents

- `.okf/decisions/gemma4-round5-sequence-cursor-barrier-elision-multi-simd.md`: `CREATED` - Architectural decision specifying default sequence cursor, serial barrier elision, and multi-SIMDgroup M=32 prefill GEMM.
- `.okf/decisions/gemma4-round4-m32-prefill-wide-fused-decode.md`: `EXISTING` - Round 4 architectural decision record.
- `.okf/decisions/gemma4-round3-prebaked-prefill-wide-decode-and-mask-elision.md`: `EXISTING` - Round 3 architectural decision record.
- `.okf/decisions/gemma4-round2-sram-staging-and-m32-tiling.md`: `EXISTING` - Round 2 architectural decision record.
- `.okf/decisions/gemma4-batched-prefill-and-pipelined-queue.md`: `EXISTING` - Round 1 architectural decision record.
- `.okf/decisions/llama-cpp-target.md`: `EXISTING` - Target parity definition against `llama.cpp`.
- `.okf/experiments/exp-0149-gemma4-performance-round4-2026-09-04.md`: `EXISTING` - Round 4 benchmark receipt.
- `.okf/experiments/exp-0150-gemma4-performance-round5-2026-09-05.md`: `PENDING` - Benchmark receipt and experiment tracking document.

## Complete when

- All 5 checklist items below are executed and checked.
- All unit and integration test suites pass (`ninja -f ninja.build all test`).
- Gemma 4 12B prefill TTFT drops toward target (< 1000 ms, approaching llama.cpp 325 ms).
- Gemma 4 12B decode TPOT drops toward target (< 60 ms/tok, approaching llama.cpp 36.5 ms).
- Output tokens match the pinned llama.cpp golden sequence: `100, 45518, 107, 14947, 135383,...`.
- An immutable benchmark receipt is recorded in `bench/results/` and linked in `.okf/experiments/`.

---

## Checklist

### [x] G4P5-01: Default Lease-Backed Sequence Cursor in Native Generation
- **REPO**: `llmopt`
- **WHERE**: `lib/generation.ml`
- **IMPORTANT FILES**:
  - `lib/generation.ml`: `MODIFY` - Enable `Serving_engine.Sequence` cursor by default in `Native_engine.prompt` and `Native_engine.decode` (support opt-out via `LLMOPT_DISABLE_SEQUENCE_CURSOR=1`).
- **IMPORTANT SYMBOLS**: `Generation.Native_engine.prompt`, `Generation.Native_engine.decode`.
- **WHY**: Without the sequence cursor, each decode token performs a full radix tree prefix match and allocates slot arrays on the host, adding 12-15 ms/token of unnecessary host CPU latency.
- **GATE**: `ninja -f ninja.build all test` passes; 32-token benchmark confirms median decode drops to < 70 ms with 100% bit-exact parity.

### [x] G4P5-02: Serial Compute Encoder Barrier Elision
- **REPO**: `llmopt`
- **WHERE**: `native/ocaml_metal_stubs.m`, `lib/metal_runtime.ml`
- **IMPORTANT FILES**:
  - `native/ocaml_metal_stubs.m`: `MODIFY` - In `llmopt_worker_loop`, `caml_llmopt_prebaked_execute`, and `caml_llmopt_metal_batch_dispatch_all`, do not issue `[enc memoryBarrierWithScope:MTLBarrierScopeBuffers]` on serial compute encoders when concurrent dispatch is disabled (or allow opt-in/opt-out via flag).
- **IMPORTANT SYMBOLS**: `llmopt_worker_loop`, `caml_llmopt_prebaked_execute`.
- **WHY**: Metal guarantees in-order sequential execution across dispatches within a serial compute command encoder; 635 memory barriers per token cause repeated L1 cache flushes and execution unit drain stalls.
- **GATE**: `ninja -f ninja.build all test` passes; benchmark confirms bit-exact parity maintained.

### [x] G4P5-03: Multi-SIMDgroup ($N_{\text{simd}} = 2$) 64-Thread Register-Safe $M=32$ Prefill GEMM
- **REPO**: `llmopt`
- **WHERE**: `lib/metal_block_codegen.ml`, `lib/metal.ml`, `lib/kernel_abi.ml`, `lib/metal_runtime.ml`, `test/test.ml`
- **IMPORTANT FILES**:
  - `lib/metal_block_codegen.ml`: `MODIFY` - Re-architect `emit_q4_0_linear_f16_m32` and `emit_q4_0_linear_f16_m32_add` to use a 64-thread threadgroup (2 SIMDgroups): SIMDgroup 0 computes rows 0..15, SIMDgroup 1 computes rows 16..31. Each thread holds only 16 accumulators, avoiding register spill while streaming weights once per column.
  - `lib/metal.ml`: `MODIFY` - Re-architect `llmopt_q4_0_qkv_linear_f16_m32` to use 64 threads (2 SIMDgroups) with 16 rows per SIMDgroup.
  - `lib/kernel_abi.ml`: `MODIFY` - Restore `problem.m >= 32` for `m32_quant_registration` with `threads_per_threadgroup = 64`.
  - `lib/metal_runtime.ml`: `MODIFY` - Set threadgroup size to `(64, 1, 1)` and grid size to `((m + 31) / 32) * n * 64` for m32 kernels; select m32 when `m >= 32`.
  - `test/test.ml`: `MODIFY` - Update unit tests for `m=37` selecting `_m32`.
- **IMPORTANT SYMBOLS**: `llmopt_q4_0_linear_f16_m32`, `llmopt_q4_0_linear_f16_m32_add`, `llmopt_q4_0_qkv_linear_f16_m32`, `m32_quant_registration`.
- **WHY**: Cuts weight streaming passes for prompt-37 from 3 down to 2 (from 18.2 GB to 12.2 GB) without register spilling, speeding up prefill by ~33%.
- **GATE**: `ninja -f ninja.build all test` passes cleanly.

### [x] G4P5-04: Compile Round 5 bundle and verify 128-token bit-exact parity
- **REPO**: `llmopt`
- **WHERE**: `bin/pipeline.ml`, `_artifacts/gemma4-12b-qat-round5-2026-09-05`
- **COMMAND**: Compile bundle `_artifacts/gemma4-12b-qat-round5-2026-09-05` using `llmopt-pipeline` from existing weights, then run 128-token benchmark with `llmopt-bench-tokens`.
- **GATE**: Verify 128 completion tokens match pinned golden sequence bit-for-bit:
  `100, 45518, 107, 14947, 135383, 568, 11124, 568, 147505, 236789, ...`.

### [x] G4P5-05: Record benchmark receipt and experiment tracking
- **REPO**: `llmopt`
- **WHERE**: `bench/results/uoc-gemma4-performance-round5-2026-09-05.txt`, `.okf/experiments/exp-0150-gemma4-performance-round5-2026-09-05.md`, `.okf/tracking.md`
- **GATE**: Commit all changes with clean git tree and test validation.
