# Gemma 4 Performance & Throughput Parity Loop - Round 3

## Goal

Close the remaining performance and throughput gap between `llmopt` and `llama.cpp` for Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal without expanding the kernel zoo.

Target performance profile on Apple M4 Pro:
- **Prefill Target**: Match or exceed `113.69 tok/s` (~325 ms for prompt-37; current baseline ~1,409 ms, 4.3x gap).
- **Decode Target**: Match or exceed `27.37 tok/s` (~36.5 ms/tok; current baseline ~86.4 ms/tok, 2.37x gap).
- **Parity Invariant**: Maintain bit-exact token parity on prompt `2,105,9731,...` generating `100, 45518, 107, 14947, 135383,...`.
- **Architectural Invariant**: Avoid bespoke monolithic kernel zoo proliferation by relying on isolated dequant micro-ops, parameterized execution templates, and compiler DAG fusion.

## Out of scope

- Creating bespoke one-off monolithic kernels per quant format (e.g. `q4_0_llama_mv_add`, `q4_0_gated_llama_mv`).
- Model-specific hardcoded hacks outside of generic compiler IR passes and hardware profiles.
- Speculative assistant (MTP) serving in this milestone (this milestone is purely base sequential prefill + decode).
- Downloading remote weights or GGUF files (always reuse local archive `_artifacts/gemma4-12b-qat-primitive-schedule-2026-09-03/source/target.weights.llmopt`).

## Relevant files

- `native/ocaml_metal_stubs.m`: Low-level Metal runtime stubs, `llmopt_ring_queue`, worker loop, and batch dispatch loop.
- `lib/metal_block_codegen.ml`: Code generation for quantized linear decode ($M=1$, $N_{\text{tile}}=8$) and prefill ($M \ge 16$) templates.
- `lib/kernel_abi.ml`: Kernel ABI and cost-model tactic registration for $N_{\text{tile}}=8$ wide decode linear candidate.
- `lib/metal.ml`: Metal code generation, header templates, and tactic tables.
- `lib/metal_runtime.ml`: Metal runtime dispatch, grid sizing, buffer binding, and tactic selection.
- `lib/serving_engine.ml`: Engine orchestration for prefill and decode phases, auxiliary buffer caching, and token execution.
- `bin/pipeline.ml`: Pipeline compiler tool (`llmopt-pipeline`) for generating AOT engine packages.
- `.okf/decisions/gemma4-round3-prebaked-prefill-wide-decode-and-mask-elision.md`: Governing OKF decision record.

## Supporting documents

- `.okf/decisions/gemma4-round3-prebaked-prefill-wide-decode-and-mask-elision.md`: `CREATED` - Architectural decision specifying vectorized dispatch, wide decode GEMV, and prefill buffer caching.
- `.okf/decisions/gemma4-round2-sram-staging-and-m32-tiling.md`: `EXISTING` - Round 2 architectural decision record.
- `.okf/decisions/gemma4-batched-prefill-and-pipelined-queue.md`: `EXISTING` - Round 1 architectural decision record.
- `.okf/decisions/llama-cpp-target.md`: `EXISTING` - Target parity definition against `llama.cpp`.
- `.okf/experiments/exp-0147-gemma4-performance-round2-2026-09-04.md`: `EXISTING` - Round 2 benchmark receipt.
- `.okf/experiments/exp-0148-gemma4-performance-round3-2026-09-04.md`: `PENDING` - Benchmark receipt and experiment tracking document.

## Complete when

- All 5 checklist items below are executed and checked.
- All unit and integration test suites pass (`ninja -f ninja.build all test`).
- Gemma 4 12B prefill TTFT drops toward target (< 800 ms, approaching llama.cpp 325 ms).
- Gemma 4 12B decode TPOT drops toward target (< 60 ms/tok, approaching llama.cpp 36.5 ms).
- Output tokens match the pinned llama.cpp golden sequence: `100, 45518, 107, 14947, 135383,...`.
- An immutable benchmark receipt is recorded in `bench/results/` and linked in `.okf/experiments/`.

---

## Checklist

### [x] G4P3-01: Vectorized batch buffer binding & pipeline state caching
- **REPO**: `llmopt`
- **WHERE**: `native/ocaml_metal_stubs.m`
- **IMPORTANT FILES**:
  - `native/ocaml_metal_stubs.m`: `MODIFY` - In `llmopt_worker_loop` and `caml_llmopt_metal_batch_dispatch_all`, cache `current_pipeline` to avoid redundant `[enc setComputePipelineState:r->pipeline]` calls, and use `[enc setBuffers:r->buffers offsets:r->offsets withRange:NSMakeRange(0, r->buffer_count)]` instead of per-buffer message loops.
- **IMPORTANT SYMBOLS**: `llmopt_worker_loop`, `caml_llmopt_metal_batch_dispatch_all`.
- **WHY**: In both decode and prefill execution, iterating individual `[enc setBuffer:offset:atIndex:]` calls and re-binding identical compute pipelines causes thousands of unnecessary Objective-C runtime message dispatches per step.
- **FIX**:
  1. In `llmopt_worker_loop`, track `current_pipeline`, call `setComputePipelineState:` only when `r->pipeline != current_pipeline`, and call `setBuffers:offsets:withRange:`.
  2. In `caml_llmopt_metal_batch_dispatch_all`, track `current_pipeline`, call `setComputePipelineState:` only on state change, and pack buffer arrays to dispatch with `setBuffers:offsets:withRange:`.
- **GATE**: `ninja -f ninja.build all test` passes cleanly.

### [ ] G4P3-02: Wide decode GEMV ($N_{\text{tile}} = 8$)
- **REPO**: `llmopt`
- **WHERE**: `lib/metal_block_codegen.ml`, `lib/kernel_abi.ml`, `lib/metal_runtime.ml`, `lib/metal.ml`, `test/test.ml`
- **IMPORTANT FILES**:
  - `lib/metal_block_codegen.ml`: `MODIFY` - Implement `emit_q4_0_linear_f16_x8` computing 8 columns per SIMD group with cooperative threadgroup SRAM staging.
  - `lib/kernel_abi.ml`: `MODIFY` - Register `eight_column_decode_quant_registration` for Q4_0 with `activation_reuse = 8` and `output_values_per_threadgroup = 64`.
  - `lib/metal.ml`: `MODIFY` - Register kernel entry `llmopt_q4_0_linear_f16_x8` in `registered_kernels`.
  - `lib/metal_runtime.ml`: `MODIFY` - Add `_x8` suffix grid calculation: `(n + 7) / 8`.
  - `test/test.ml`: `MODIFY` - Update decode linear tactic test expectations.
- **IMPORTANT SYMBOLS**: `llmopt_q4_0_linear_f16_x8`, `eight_column_decode_quant_registration`.
- **WHY**: In decode GEMV ($M=1$), computing 8 columns per SIMD group cuts the total dispatched threadgroups in half, doubles activation value register reuse across columns, and reduces threadgroup barrier overhead.
- **GATE**: `ninja -f ninja.build all test` passes cleanly.

### [ ] G4P3-03: Prefill auxiliary mask buffer caching & retained token buffer
- **REPO**: `llmopt`
- **WHERE**: `lib/serving_engine.ml`
- **IMPORTANT FILES**:
  - `lib/serving_engine.ml`: `MODIFY` - Add auxiliary inputs cache (`auxiliary_inputs_cache`) and retained prefill token buffers (`prefill_token_buffers`) to `Serving_engine.t`. Reuse pre-allocated GPU buffers for identical token counts instead of reallocating host bytes and Metal buffers on each prefill.
- **IMPORTANT SYMBOLS**: `transformer_auxiliary_inputs`, `prefill`, `prefill_token_buffers`, `auxiliary_inputs_cache`.
- **WHY**: Allocating and copying position IDs and attention mask buffers on every prefill step causes driver buffer churn and synchronization latency.
- **GATE**: `ninja -f ninja.build all test` passes cleanly.

### [ ] G4P3-04: Compile Round 3 bundle and verify 128-token bit-exact parity
- **REPO**: `llmopt`
- **WHERE**: `bin/pipeline.ml`, `_artifacts/gemma4-12b-qat-round3-2026-09-04`
- **COMMAND**: Compile bundle `_artifacts/gemma4-12b-qat-round3-2026-09-04` using `llmopt-pipeline` from existing weights, then run 128-token benchmark with `llmopt-bench-tokens`.
- **GATE**: Verify 128 completion tokens match pinned golden sequence bit-for-bit:
  `100, 45518, 107, 14947, 135383, 568, 11124, 568, 147505, 236789, ...`.

### [ ] G4P3-05: Record benchmark receipt and experiment tracking
- **REPO**: `llmopt`
- **WHERE**: `bench/results/uoc-gemma4-performance-round3-2026-09-04.txt`, `.okf/experiments/exp-0148-gemma4-performance-round3-2026-09-04.md`, `.okf/tracking.md`
- **GATE**: Commit all changes with clean git tree and test validation.
