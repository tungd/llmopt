---
type: Experiment
title: 'Wide decode GEMV (N_tile=8), vectorized dispatch, and prefill cache for Gemma 4'
description: 'Implements wide N_tile=8 decode GEMV (llmopt_q4_0_linear_f16_x8) doubling register activation reuse and halving decode threadgroups, vectorizes batch buffer binding and caches pipeline state in the worker loop and batch dispatch, caches prefill auxiliary mask and token buffers, and sets a new throughput record of 10.48 tok/s with 100% bit-exact parity.'
tags: [experiment, compiler, prefill, gemm, gemv, sram-staging, metal, gemma4]
status: stable
generated: { by: antigravity/2.0, at: '2026-09-04T18:52:00+07:00' }
sources:
  - id: decision-round3
    resource: /.okf/decisions/gemma4-round3-prebaked-prefill-wide-decode-and-mask-elision.md
    title: Gemma 4 Round 3 Prebaked Pipeline Dispatch, Wide Decode GEMV, and Prefill Auxiliary Buffer Caching
  - id: decision-round2
    resource: /.okf/decisions/gemma4-round2-sram-staging-and-m32-tiling.md
    title: Gemma 4 Round 2 SRAM Activation Staging and M=16/32 Prefill Tiling
  - id: block-codegen
    resource: /lib/metal_block_codegen.ml
    title: Wide N_tile=8 decode GEMV micro-ops and cooperative SRAM staging
  - id: kernel-abi
    resource: /lib/kernel_abi.ml
    title: Kernel ABI registration and cost model for eight-column decode GEMV
  - id: runtime-stubs
    resource: /native/ocaml_metal_stubs.m
    title: Vectorized batch buffer binding and compute pipeline state caching
  - id: engine
    resource: /lib/serving_engine.ml
    title: Prefill auxiliary mask and token buffer caching
  - id: prior-device-run
    resource: /.okf/experiments/exp-0147-gemma4-performance-round2-2026-09-04.md
    title: Round 2 M=16 prefill GEMM and SRAM decode staging receipt
  - id: device-run-receipt
    resource: /bench/results/uoc-gemma4-performance-round3-2026-09-04.txt
    title: Gemma 4 Round 3 performance execution receipt
---

# Result

Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal achieves its highest serving throughput yet, setting a new project record while strictly avoiding monolithic handwritten kernel proliferation:

1. **Wide Decode GEMV Tiling ($N_{\text{tile}} = 8$)**:
   - `llmopt_q4_0_linear_f16_x8` computes 8 columns per SIMD group (64 columns per 256-thread threadgroup) directly accumulating into 8 floating-point registers per thread.
   - Halves the total threadgroup dispatches during decode linear projections (e.g. down to 60 threadgroups for $N=3840$), doubles register activation value reuse across columns, and reduces threadgroup barrier overhead.
   - Inter-token median latency improves to `84.17 ms` (with steady-state down to `83.19 ms`, a ~2.75 ms/token reduction vs exp-0147).

2. **Vectorized Batch Buffer Binding & Pipeline State Caching**:
   - In `native/ocaml_metal_stubs.m` (`llmopt_worker_loop` and `caml_llmopt_metal_batch_dispatch_all`), compute pipeline state is cached across adjacent dispatches, and buffer bindings are issued via vector `[enc setBuffers:offsets:withRange:]`.
   - Eliminates thousands of redundant Objective-C message dispatches per token during decode and throughout batch schedule encoding.

3. **Prefill Auxiliary Buffer & Token Cache**:
   - In `lib/serving_engine.ml`, prefill input token buffers, RoPE slice tables, and causal/sliding-window attention mask buffers are retained and cached by shape and token count in `Serving_engine.t`.
   - Reuses GPU memory without reallocating host byte arrays or Metal buffers on each request.

4. **Bit-Exact Token Parity & New Throughput Record**:
   - Total generation elapsed time drops to `12.217 s` (fastest 128-token run recorded).
   - Serving throughput reaches `10.48 tok/s` (highest achieved).
   - 100% bit-exact sequence parity maintained across all 128 generated tokens on the pinned prompt:
     `100, 45518, 107, 14947, 135383, 568, 11124, 568, 147505, 236789, ...`.

# Static Validation

`ninja -f ninja.build all test` passes cleanly with all 79 Python unit tests and all canonical OCaml test suites passing. Clean orthogonal dequant micro-ops and parameterized templates preserved throughout.

# Device Execution Receipt

Full 128-token autoregressive generation on Apple Metal succeeded
(`_artifacts/gemma4-12b-qat-round3-2026-09-04`):
- **TTFT**: `1.427907 s`.
- **Total Elapsed**: `12.216533 s` (-0.170 s vs exp-0147, fastest end-to-end benchmark).
- **Generation Throughput**: `10.477604 tok/s` (+0.144 tok/s, highest serving throughput achieved).
- **Inter-token Mean Latency**: `84.949813 ms` (-1.48 ms/token faster).
- **Inter-token Median Latency**: `84.168911 ms` (-1.77 ms/token faster; steady-state down to 83.19 ms).
- **Bit-Exact Parity**: All 128 tokens match the golden sequence from exp-0145, exp-0146, and exp-0147 with 100% bit-exact fidelity.

Evidence: `bench/results/uoc-gemma4-performance-round3-2026-09-04.txt` records the measured 128-token Metal execution receipt.
