---
type: Decision
title: 'M=32 Prefill GEMM Tiling, Wide Decode QKV/Gated GEMV (N_tile=8), and Retained Prefill Plan Execution'
description: 'Reduce prefill weight streaming passes by 33% via M=32 GEMM tiling, halve decode QKV and gated threadgroups via N_tile=8 GEMV, and eliminate schedule encoding overhead via retained prefill dispatch plans.'
tags: [decision, compiler, prefill, gemm, gemv, sram, metal, gemma4, zero-jit]
status: stable
generated: { by: human:tung, at: '2026-09-04T19:15:00+07:00' }
sources:
  - id: gemma4-round3
    resource: /.okf/decisions/gemma4-round3-prebaked-prefill-wide-decode-and-mask-elision.md
    title: Prebaked Pipeline Dispatch, Wide Decode GEMV (N_tile=8), and Prefill Auxiliary Buffer Caching
  - id: gemma4-round2
    resource: /.okf/decisions/gemma4-round2-sram-staging-and-m32-tiling.md
    title: Threadgroup SRAM Staging and M=32 Prefill Tiling for Gemma 4
  - id: llama-cpp-target
    resource: /.okf/decisions/llama-cpp-target.md
    title: llama.cpp performance target
---

# Decision

Following Round 3 (`exp-0148`), which reduced inter-token decode latency to 84.17 ms median (83.19 ms steady-state) and prefill TTFT to 1,428 ms, `llmopt` advances to Round 4 to close the remaining throughput gap against llama.cpp (113.7 tok/s prefill, 27.4 tok/s decode):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Batched M=32 Prefill GEMM Tiling                                         │
│                                                                             │
│  - Bottleneck: Prefill currently uses M=16 tiling. For prompt-37, this      │
│    requires ceil(37 / 16) = 3 full streaming passes over the 6.1 GB model  │
│    weights (totaling 18.24 GB memory traffic).                              │
│  - Solution: Implement M=32 batched GEMM templates                          │
│    (llmopt_q4_0_linear_f16_m32, llmopt_q4_0_linear_f16_m32_add,             │
│     llmopt_q4_0_qkv_linear_f16_m32, llmopt_q4_0_gated_linear_f16_m32) with │
│    boundary predicates (row_base + r < params.m). Register m32_quant in     │
│    the cost model with decoded_weight_reuse = 32.                           │
│  - Result: Reduces prompt-37 weight passes from 3 down to 2 passes           │
│    (ceil(37 / 32) = 2), cutting weight traffic from 18.24 GB to 12.16 GB     │
│    (33% memory bandwidth reduction) and driving TTFT under 1.0 s.           │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Wide Decode QKV & Gated GEMV Tiling (N_tile = 8)                         │
│                                                                             │
│  - Bottleneck: While general linear decode was widened to N_tile = 8 in     │
│    Round 3, llmopt_q4_0_qkv_linear_f16 and llmopt_q4_0_gated_linear_f16     │
│    still use N_tile = 4 (col_groups = (n + 3) / 4), dispatching twice as    │
│    many threadgroups as necessary across all 48 decoder layers.              │
│  - Solution: Provide llmopt_q4_0_qkv_linear_f16_x8 and                      │
│    llmopt_q4_0_gated_linear_f16_x8 computing 8 columns per SIMD group with  │
│    threadgroup SRAM staging and 8-wide float accumulator registers.         │
│  - Result: Halves threadgroup count for QKV and Gated MLP projections,      │
│    doubling activation register reuse and reducing threadgroup barrier      │
│    synchronizations during decode.                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Retained Prefill Plan Execution in Serving_engine                        │
│                                                                             │
│  - Bottleneck: Metal_runtime.execute_schedule traverses and encodes the     │
│    entire schedule AST on every prefill step, allocating parameter objects  │
│    and building dispatches dynamically.                                     │
│  - Solution: Cache the compiled batch dispatch list for a given token count │
│    along with retained prefill buffers, executing dispatches via direct     │
│    vectorized batch dispatch on repeated or steady queries.                 │
│  - Result: Eliminates CPU-side schedule traversal overhead during prefill.  │
└─────────────────────────────────────────────────────────────────────────────┘
```
