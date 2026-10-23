---
type: Decision
title: 'Gemma 4 Round 7 Barrier-Free Decode GEMV, Double-Buffered Async Ring Queue, and Prefill GEMM Plan Caching'
description: 'Architectural decision for Round 7 closing the performance gap to llama.cpp: (1) Eliminates threadgroup barriers and threadgroup SRAM staging in decode GEMV in favor of direct L1/L2 activation streaming; (2) Implements ping-pong double buffering in the asynchronous Ring_queue to eliminate token/output buffer data races and allow safe depth-2 overlapping; (3) Caches prefill execution plans for prompt token counts to slash TTFT.'
tags: [decision, architecture, compiler, gemv, barrier-free, double-buffering, async, prefill, gemma4]
status: stable
decision_status: approved
generated: { by: antigravity/2.0, at: '2026-09-05T22:45:00+07:00' }
sources:
  - id: decision-round6
    resource: /.okf/decisions/gemma4-round6-compiler-dag-fusion-splitk-and-kernel-retirement.md
    title: Gemma 4 Round 6 Compiler DAG Fusion, Split-K GEMV, and Bespoke Kernel Retirement
  - id: experiment-round6
    resource: /.okf/experiments/exp-0151-gemma4-performance-round6-2026-09-05.md
    title: Gemma 4 Round 6 Performance Experiment Receipt
  - id: metal-block-codegen
    resource: /lib/metal_block_codegen.ml
    title: Parameterized Block Codegen
  - id: metal-stubs
    resource: /native/ocaml_metal_stubs.m
    title: Metal Runtime Stubs and Ring Queue
  - id: serving-engine
    resource: /lib/serving_engine.ml
    title: Serving Engine and Static Execution
---

# Context

In Round 6 ([`exp-0151`](file:///Users/tung/Projects/std23/llmopt/.okf/experiments/exp-0151-gemma4-performance-round6-2026-09-05.md)), `llmopt` achieved complete bespoke kernel retirement: all 846 lines of handwritten monolithic kernels in `lib/metal.ml` were deleted in favor of pure schema-driven block codegen (`lib/metal_block_codegen.ml`), maintaining 100% bit-exact parity across all 128 generated completion tokens.

However, profiling revealed three key bottlenecks capping performance at 2.83 tok/s (vs `llama.cpp`'s 27.94 tok/s):
1. **Threadgroup Barrier Stalls in Decode GEMV**: `emit_q4_0_linear_f16_x8` uses `threadgroup half s_activation[4096]` with two `threadgroup_barrier` calls per 4096-chunk. Because $K \in [4096, 28672]$, the activation vector (8 KB to 56 KB) easily fits in Apple M4 Pro L1/L2 cache. Barriers force SIMDgroup execution units to stall and synchronize instead of streaming independent dot products.
2. **Data Race in Ring Queue Depth > 1**: The asynchronous worker loop in `native/ocaml_metal_stubs.m` used a single `token_buffer` and `output_buffer`. Submitting token $t+1$ while token $t$ was executing on the GPU caused data races, forcing Round 6 to fall back to synchronous `Prebaked.execute`. In synchronous mode, the host incurs ~190 ms of CPU dispatch encoding and roundtrip overhead per token even though GPU execution is only 72 ms.
3. **Cold Prefill Schedule Compilation**: Prefill TTFT measured 9.73 s because prompt-37 was not warmed up in `warmup_engine` (which only warmed up 4 tokens), forcing first-time schedule creation and prebaking inside `Driver.run`.

# Decisions

1. **Barrier-Free Streaming Decode GEMV**:
   - In `lib/metal_block_codegen.ml`, remove `s_activation` threadgroup staging and barriers from `emit_q4_0_linear_f16_x8`.
   - Each SIMDgroup streams directly from `device const half* activation = input` (serviced by GPU cache at terabytes-per-second bandwidth).
   - Each SIMDgroup computes 8 columns independently using `simd_sum` reduction without intra-threadgroup synchronization.

2. **Ping-Pong Double-Buffered Ring Queue**:
   - In `native/ocaml_metal_stubs.m`, allocate double-buffered slots (`token_buffers[2]` and `output_buffers[2]`) in `llmopt_ring_queue`.
   - Dispatch $t$ reads `token_buffers[t % 2]` and writes `output_buffers[t % 2]`. Command $t+1$ binds slot $(t+1) % 2$.
   - Re-enable `Ring_queue` as default in `lib/serving_engine.ml`, enabling safe depth-2 overlapping of CPU encoding and GPU execution.

3. **Prefill Schedule Caching & Warmup**:
   - In `lib/serving_engine.ml`, extend `warmup_engine` or pre-compile prefill plans for the target prompt length, ensuring zero compilation bubbles during generation.

4. **Invariants**:
   - 100% bit-exact parity across all 128 completion tokens against the pinned golden sequence:
     `100, 45518, 107, 14947, 135383, 568, 11124, 568, 147505, 236789, ...`.
   - Zero remote downloads: strictly reuse local weights at `_artifacts/gemma4-12b-qat-primitive-schedule-2026-09-03/source/target.weights.llmopt`.
   - Zero handwritten monolithic kernels in `lib/metal.ml`.
