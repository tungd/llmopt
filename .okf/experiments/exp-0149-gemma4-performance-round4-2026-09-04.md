---
type: Experiment
title: 'Wide decode QKV GEMV (N_tile=8), retained prefill plan caching, and balanced GEMM tiling for Gemma 4'
description: 'Implements wide N_tile=8 decode QKV GEMV (llmopt_q4_0_qkv_linear_f16_x8), caches prebaked prefill dispatch execution plans in Serving_engine.t avoiding schedule AST re-encoding, characterizes register pressure thresholds for batched prefill GEMM (M=32 vs M=16), and achieves record-low inter-token median latency of 81.88 ms (steady-state 76.23 ms) with 100% bit-exact parity.'
tags: [experiment, compiler, prefill, gemm, gemv, qkv, sram-staging, metal, gemma4]
status: stable
generated: { by: antigravity/2.0, at: '2026-09-05T06:10:00+07:00' }
sources:
  - id: decision-round4
    resource: /.okf/decisions/gemma4-round4-m32-prefill-wide-fused-decode.md
    title: Gemma 4 Round 4 M=32 Batched Prefill GEMM, Wide Decode QKV/Gated GEMV, and Prefill Plan Caching
  - id: decision-round3
    resource: /.okf/decisions/gemma4-round3-prebaked-prefill-wide-decode-and-mask-elision.md
    title: Gemma 4 Round 3 Prebaked Pipeline Dispatch, Wide Decode GEMV, and Prefill Auxiliary Buffer Caching
  - id: decision-round2
    resource: /.okf/decisions/gemma4-round2-sram-staging-and-m32-tiling.md
    title: Gemma 4 Round 2 SRAM Activation Staging and M=16/32 Prefill Tiling
  - id: block-codegen
    resource: /lib/metal_block_codegen.ml
    title: Batched M=32 prefill GEMM code generation
  - id: metal-kernels
    resource: /lib/metal.ml
    title: Wide N_tile=8 decode QKV GEMV and batched QKV/gated linear kernels
  - id: kernel-abi
    resource: /lib/kernel_abi.ml
    title: Kernel ABI registration and cost model for batched prefill GEMM
  - id: metal-runtime
    resource: /lib/metal_runtime.ml
    title: Metal runtime grid sizing, tactic selection, and dispatch routing
  - id: engine
    resource: /lib/serving_engine.ml
    title: Retained prefill dispatch plan caching in Serving_engine.t
  - id: prior-device-run
    resource: /.okf/experiments/exp-0148-gemma4-performance-round3-2026-09-04.md
    title: Round 3 wide decode GEMV and prefill cache receipt
  - id: device-run-receipt
    resource: /bench/results/uoc-gemma4-performance-round4-2026-09-04.txt
    title: Gemma 4 Round 4 performance execution receipt
---

# Result

Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal achieves a new record-low inter-token median latency of `81.88 ms` (with warm steady-state decode dropping to `76.23 ms`), while maintaining 100% bit-exact parity across all 128 generated completion tokens:

1. **Wide Decode QKV GEMV ($N_{\text{tile}} = 8$)**:
   - `llmopt_q4_0_qkv_linear_f16_x8` computes 8 columns across Q, K, and V projections per SIMD group using threadgroup SRAM activation staging.
   - Halves the number of threadgroup dispatches across all 48 QKV projection layers during decode, doubling activation register reuse across projected columns.
   - Balances decode GEMV register pressure: linear projections compute 8 columns (`_x8`), dual-projection gated linear computes 4 gate + 4 up = 8 columns (`llmopt_q4_0_gated_linear_f16`), and QKV computes 8 columns (`_x8`). Keeping each thread to $\le 8$ accumulators eliminates register spilling and drops inter-token median decode latency from `84.17 ms` (exp-0148) to `81.88 ms` (exp-0149), with steady-state down to `76.23 ms`.

2. **Retained Prefill Dispatch Plan Caching**:
   - In `lib/serving_engine.ml`, prefill execution dispatches are cached inside `Serving_engine.t` in a dedicated `prefill_executables` hashtable keyed by token count.
   - On repeated prefill evaluations with matching sequence length, the prebaked dispatch list is executed directly without re-encoding schedule AST or repeatedly traversing runtime graph representations.

3. **Characterization of $M=32$ Prefill GEMM Register Spill Thresholds**:
   - Lowering 32 rows unrolled in a single SIMD group (`in_ptrs[32]` + `acc[32]`) requires $>120$ 32-bit registers, triggering local stack spilling to DRAM on Apple Silicon GPUs when combined with intermediate values.
   - Setting the threshold for $M=32$ selection in `lib/kernel_abi.ml` and `lib/metal_runtime.ml` to `m >= 64` routes prompt-37 requests to the register-safe $M=16$ batched prefill GEMM (`llmopt_q4_0_linear_f16_m16_add`, `llmopt_q4_0_qkv_linear_f16_m16`), avoiding register spilling and ensuring prefill GPU execution runs in `1.231 s` without DRAM stack traffic.

4. **Bit-Exact Token Parity**:
   - All 128 completion tokens strictly match the golden sequence from exp-0145, exp-0146, exp-0147, and exp-0148 with 100% bit-exact fidelity:
     `100, 45518, 107, 14947, 135383, 568, 11124, 568, 147505, 236789, ...`.

# Static Validation

`ninja -f ninja.build all test` passes cleanly with all 79 Python unit tests and all canonical OCaml test suites passing. Clean orthogonal dequant micro-ops and parameterized templates preserved throughout.

# Device Execution Receipt

Full 128-token autoregressive generation on Apple Metal succeeded
(`_artifacts/gemma4-12b-qat-round4-2026-09-04`):
- **TTFT**: `2.309476 s`.
- **Elapsed**: `13.718344 s`.
- **Generation Throughput**: `9.330572 tok/s`.
- **Inter-token Mean Latency**: `89.833607 ms`.
- **Inter-token Median Latency**: `81.881046 ms` (-2.29 ms/token faster than exp-0148; steady-state decode reaches 76.23 ms).
- **Bit-Exact Parity**: All 128 tokens match the golden sequence with 100% bit-exact fidelity.

Evidence: `bench/results/uoc-gemma4-performance-round4-2026-09-04.txt` records the measured 128-token Metal execution receipt.
