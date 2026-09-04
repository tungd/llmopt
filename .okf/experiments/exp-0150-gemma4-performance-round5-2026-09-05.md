---
type: Experiment
title: 'Persistent sequence cursor, serial barrier elision, and register-safe 2-SIMDgroup M=32 prefill GEMM for Gemma 4'
description: 'Promotes Serving_engine.Sequence persistent cursor to default in native generation, elides 635 redundant buffer memory barriers per token on serial compute command encoders, lowers batched M=32 prefill GEMM across 2 SIMDgroups (64 threads, 16 rows/SIMDgroup) eliminating register spill while streaming weights once per column, and achieves record-low inter-token median latency of 81.44 ms (mean 86.38 ms, TTFT 2.272 s, throughput 9.67 tok/s) with 100% bit-exact parity.'
tags: [experiment, compiler, sequence-cursor, barrier-elision, prefill, gemm, multi-simd, metal, gemma4]
status: stable
generated: { by: antigravity/2.0, at: '2026-09-05T06:40:00+07:00' }
sources:
  - id: decision-round5
    resource: /.okf/decisions/gemma4-round5-sequence-cursor-barrier-elision-multi-simd.md
    title: Gemma 4 Round 5 Sequence Cursor, Serial Barrier Elision, and Multi-SIMDgroup M=32 Prefill GEMM
  - id: decision-round4
    resource: /.okf/decisions/gemma4-round4-m32-prefill-wide-fused-decode.md
    title: Gemma 4 Round 4 M=32 Batched Prefill GEMM, Wide Decode QKV/Gated GEMV, and Prefill Plan Caching
  - id: block-codegen
    resource: /lib/metal_block_codegen.ml
    title: Multi-SIMDgroup 64-thread M=32 prefill GEMM code generation
  - id: metal-kernels
    resource: /lib/metal.ml
    title: Multi-SIMDgroup 64-thread M=32 QKV linear kernel
  - id: kernel-abi
    resource: /lib/kernel_abi.ml
    title: Kernel ABI registration for m32 GEMM with m >= 32
  - id: metal-runtime
    resource: /lib/metal_runtime.ml
    title: Metal runtime grid sizing and dispatch routing for multi-SIMDgroup M=32
  - id: generation
    resource: /lib/generation.ml
    title: Default persistent sequence cursor routing in Native_engine
  - id: metal-stubs
    resource: /native/ocaml_metal_stubs.m
    title: Serial compute command encoder barrier elision
  - id: prior-device-run
    resource: /.okf/experiments/exp-0149-gemma4-performance-round4-2026-09-04.md
    title: Round 4 wide decode QKV and prefill plan caching receipt
  - id: device-run-receipt
    resource: /bench/results/uoc-gemma4-performance-round5-2026-09-05.txt
    title: Gemma 4 Round 5 performance execution receipt
---

# Result

Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal achieves a new record-low inter-token median latency of `81.44 ms` (with mean latency improving to `86.38 ms`, TTFT dropping to `2.272 s`, and throughput reaching `9.67 tok/s`), while maintaining 100% bit-exact parity across all 128 generated completion tokens:

1. **Default Persistent Sequence Cursor Routing**:
   - In `lib/generation.ml`, `Native_engine.prompt` initializes and attaches `Serving_engine.Sequence` as the primary decode path, avoiding radix tree prefix traversals and dynamic slot allocation from host CPU on each decode step.
   - Provides graceful fallback to `Serving_engine.decode` if sequence cursor allocation is disabled via `LLMOPT_DISABLE_SEQUENCE_CURSOR=1` or if step conditions require re-anchoring.

2. **Serial Compute Command Encoder Barrier Elision**:
   - In `native/ocaml_metal_stubs.m`, gated `memoryBarrierWithScope:MTLBarrierScopeBuffers` behind `llmopt_enable_serial_barriers()` (enabled via `LLMOPT_ENABLE_SERIAL_BARRIERS=1`).
   - Elides 635 redundant barrier calls per token on serial compute command encoders, preventing execution unit drain stalls and L1 cache flushes between strictly ordered sequential dispatches.

3. **Multi-SIMDgroup ($N_{\text{simd}} = 2$) 64-Thread Register-Safe $M=32$ Prefill GEMM**:
   - In `lib/metal_block_codegen.ml` and `lib/metal.ml`, re-architected `emit_q4_0_linear_f16_m32`, `emit_q4_0_linear_f16_m32_add`, and `llmopt_q4_0_qkv_linear_f16_m32` to partition each 32-row tile across 2 SIMDgroups: SIMDgroup 0 computes rows 0..15, SIMDgroup 1 computes rows 16..31.
   - Each thread maintains only 16 accumulators (`acc[16]`) and 16 input pointers (`in_ptrs[16]`), consuming ~48 registers per thread—eliminating local memory stack spills to DRAM.
   - Restored selection threshold `m >= 32` in `lib/kernel_abi.ml` and `lib/metal_runtime.ml`. For prompt-37, weight streaming passes are reduced from 3 passes (18.2 GB) to 2 passes (12.2 GB), achieving faster prefill with zero register spills.

4. **Bit-Exact Token Parity**:
   - All 128 completion tokens strictly match the golden sequence from exp-0145, exp-0146, exp-0147, exp-0148, and exp-0149 with 100% bit-exact fidelity:
     `100, 45518, 107, 14947, 135383, 568, 11124, 568, 147505, 236789, ...`.

# Static Validation

`ninja -f ninja.build all test` passes cleanly with all 79 Python unit tests and all canonical OCaml test suites passing. Clean orthogonal dequant micro-ops and parameterized templates preserved throughout.

# Device Execution Receipt

Full 128-token autoregressive generation on Apple Metal succeeded
(`_artifacts/gemma4-12b-qat-round5-2026-09-05`):
- **TTFT**: `2.272004 s` (-37.47 ms faster than exp-0149).
- **Elapsed**: `13.242031 s` (-476.31 ms total elapsed time reduction).
- **Generation Throughput**: `9.666191 tok/s` (+0.336 tok/s improvement).
- **Inter-token Mean Latency**: `86.378165 ms` (-3.46 ms/token improvement).
- **Inter-token Median Latency**: `81.437111 ms` (-0.44 ms/token improvement, new project record).
- **Bit-Exact Parity**: All 128 tokens match the golden sequence with 100% bit-exact fidelity.

Evidence: `bench/results/uoc-gemma4-performance-round5-2026-09-05.txt` records the measured 128-token Metal execution receipt.
