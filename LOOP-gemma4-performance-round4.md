# Gemma 4 Performance & Throughput Parity Loop - Round 4

## Goal

Close the remaining performance and throughput gap between `llmopt` and `llama.cpp` for Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal without expanding the kernel zoo.

Target performance profile on Apple M4 Pro:
- **Prefill Target**: Match or exceed `113.69 tok/s` (~325 ms for prompt-37; exp-0148: 1,428 ms).
- **Decode Target**: Match or exceed `27.37 tok/s` (~36.5 ms/tok; exp-0148: 84.17 ms median / 83.19 ms steady-state).
- **Parity Invariant**: Maintain bit-exact token parity on prompt `2,105,9731,...` generating `100, 45518, 107, 14947, 135383,...`.
- **Architectural Invariant**: Avoid bespoke monolithic kernel zoo proliferation by relying on isolated dequant micro-ops, parameterized execution templates, and compiler DAG fusion.

## Out of scope

- Creating bespoke one-off monolithic kernels per quant format (e.g. `q4_0_llama_mv_add`, `q4_0_gated_llama_mv`).
- Model-specific hardcoded hacks outside of generic compiler IR passes and hardware profiles.
- Speculative assistant (MTP) serving in this milestone (this milestone is purely base sequential prefill + decode).
- Downloading remote weights or GGUF files (always reuse local archive `_artifacts/gemma4-12b-qat-primitive-schedule-2026-09-03/source/target.weights.llmopt`).

## Relevant files

- `lib/metal_block_codegen.ml`: Code generation for $M=32$ batched prefill GEMM (`llmopt_q4_0_linear_f16_m32`, `llmopt_q4_0_linear_f16_m32_add`).
- `lib/kernel_abi.ml`: Kernel ABI registration for `m32_quant_registration` with `decoded_weight_reuse = 32`.
- `lib/metal.ml`: Metal code generation for $M=32$ prefill (`llmopt_q4_0_qkv_linear_f16_m32`, `llmopt_q4_0_gated_linear_f16_m32`) and wide $N_{\text{tile}}=8$ decode (`llmopt_q4_0_qkv_linear_f16_x8`, `llmopt_q4_0_gated_linear_f16_x8`).
- `lib/metal_runtime.ml`: Runtime grid sizing, tactic selection, and dispatch routing for $M=32$ prefill and $N_{\text{tile}}=8$ decode.
- `lib/serving_engine.ml`: Engine orchestration and prefill dispatch plan caching.
- `bin/pipeline.ml`: Pipeline compiler tool (`llmopt-pipeline`) for generating AOT engine packages.
- `.okf/decisions/gemma4-round4-m32-prefill-wide-fused-decode.md`: Governing OKF decision record.

## Supporting documents

- `.okf/decisions/gemma4-round4-m32-prefill-wide-fused-decode.md`: `CREATED` - Architectural decision specifying M=32 prefill GEMM, wide decode QKV/gated GEMV, and prefill plan caching.
- `.okf/decisions/gemma4-round3-prebaked-prefill-wide-decode-and-mask-elision.md`: `EXISTING` - Round 3 architectural decision record.
- `.okf/decisions/gemma4-round2-sram-staging-and-m32-tiling.md`: `EXISTING` - Round 2 architectural decision record.
- `.okf/decisions/gemma4-batched-prefill-and-pipelined-queue.md`: `EXISTING` - Round 1 architectural decision record.
- `.okf/decisions/llama-cpp-target.md`: `EXISTING` - Target parity definition against `llama.cpp`.
- `.okf/experiments/exp-0148-gemma4-performance-round3-2026-09-04.md`: `EXISTING` - Round 3 benchmark receipt.
- `.okf/experiments/exp-0149-gemma4-performance-round4-2026-09-04.md`: `PENDING` - Benchmark receipt and experiment tracking document.

## Complete when

- All 5 checklist items below are executed and checked.
- All unit and integration test suites pass (`ninja -f ninja.build all test`).
- Gemma 4 12B prefill TTFT drops toward target (< 1000 ms, approaching llama.cpp 325 ms).
- Gemma 4 12B decode TPOT drops toward target (< 70 ms/tok, approaching llama.cpp 36.5 ms).
- Output tokens match the pinned llama.cpp golden sequence: `100, 45518, 107, 14947, 135383,...`.
- An immutable benchmark receipt is recorded in `bench/results/` and linked in `.okf/experiments/`.

---

## Checklist

### [x] G4P4-01: Batched $M=32$ prefill GEMM code generation & tactic registration
- **REPO**: `llmopt`
- **WHERE**: `lib/metal_block_codegen.ml`, `lib/metal.ml`, `lib/kernel_abi.ml`, `lib/metal_runtime.ml`
- **IMPORTANT FILES**:
  - `lib/metal_block_codegen.ml`: `MODIFY` - Implement `emit_q4_0_linear_f16_m32` and `emit_q4_0_linear_f16_m32_add` computing 32 rows per SIMD group with boundary predicates `r < params.m`.
  - `lib/metal.ml`: `MODIFY` - Implement `llmopt_q4_0_qkv_linear_f16_m32` and `llmopt_q4_0_gated_linear_f16_m32`, register entries in `registered_kernels` and `filter_quant_entries`.
  - `lib/kernel_abi.ml`: `MODIFY` - Add `m32_quant_registration` with `decoded_weight_reuse = 32`, `output_values_per_threadgroup = 256`, placed before `m16_quant_registration` in `table`.
  - `lib/metal_runtime.ml`: `MODIFY` - Add `_m32` and `_m32_add` grid sizing (`((m + 31) / 32) * n`), wire `Quant_qkv_tactic` and gated linear selection when `m >= 32`.
- **IMPORTANT SYMBOLS**: `llmopt_q4_0_linear_f16_m32`, `llmopt_q4_0_linear_f16_m32_add`, `llmopt_q4_0_qkv_linear_f16_m32`, `llmopt_q4_0_gated_linear_f16_m32`, `m32_quant_registration`.
- **WHY**: For prompt-37, M=16 requires 3 full weight streaming passes (18.24 GB). M=32 reduces weight passes to 2 (12.16 GB), saving 33% of prefill memory bandwidth.
- **GATE**: `ninja -f ninja.build all test` passes cleanly.

### [x] G4P4-02: Wide decode QKV & Gated GEMV ($N_{\text{tile}} = 8$)
- **REPO**: `llmopt`
- **WHERE**: `lib/metal.ml`, `lib/metal_runtime.ml`
- **IMPORTANT FILES**:
  - `lib/metal.ml`: `MODIFY` - Implement `llmopt_q4_0_qkv_linear_f16_x8` and `llmopt_q4_0_gated_linear_f16_x8` computing 8 columns per SIMD group with threadgroup SRAM activation staging. Register entries in `registered_kernels`.
  - `lib/metal_runtime.ml`: `MODIFY` - Select `_x8` kernels for decode ($m=1$) in `Quant_qkv_tactic` and `gated_linear`, compute grid size with `(n + 7) / 8`.
- **IMPORTANT SYMBOLS**: `llmopt_q4_0_qkv_linear_f16_x8`, `llmopt_q4_0_gated_linear_f16_x8`.
- **WHY**: Halves decode threadgroups for all QKV and Gated MLP projections, doubling activation register reuse across columns.
- **GATE**: `ninja -f ninja.build all test` passes cleanly.

### [x] G4P4-03: Retained prefill plan caching in Serving_engine
- **REPO**: `llmopt`
- **WHERE**: `lib/serving_engine.ml`
- **IMPORTANT FILES**:
  - `lib/serving_engine.ml`: `MODIFY` - Cache compiled prefill dispatches in `Serving_engine.t` keyed by token count, avoiding schedule AST re-encoding.
- **IMPORTANT SYMBOLS**: `Serving_engine.t`, `prefill`.
- **WHY**: Re-encoding schedule AST on every prefill step adds CPU dispatch overhead.
- **GATE**: `ninja -f ninja.build all test` passes cleanly.

### [ ] G4P4-04: Compile Round 4 bundle and verify 128-token bit-exact parity
- **REPO**: `llmopt`
- **WHERE**: `bin/pipeline.ml`, `_artifacts/gemma4-12b-qat-round4-2026-09-04`
- **COMMAND**: Compile bundle `_artifacts/gemma4-12b-qat-round4-2026-09-04` using `llmopt-pipeline` from existing weights, then run 128-token benchmark with `llmopt-bench-tokens`.
- **GATE**: Verify 128 completion tokens match pinned golden sequence bit-for-bit:
  `100, 45518, 107, 14947, 135383, 568, 11124, 568, 147505, 236789, ...`.

### [ ] G4P4-05: Record benchmark receipt and experiment tracking
- **REPO**: `llmopt`
- **WHERE**: `bench/results/uoc-gemma4-performance-round4-2026-09-04.txt`, `.okf/experiments/exp-0149-gemma4-performance-round4-2026-09-04.md`, `.okf/tracking.md`
- **GATE**: Commit all changes with clean git tree and test validation.
