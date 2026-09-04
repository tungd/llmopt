---
type: Decision
title: 'Prebaked Pipeline Dispatch, Wide Decode GEMV (N_tile=8), and Prefill Auxiliary Buffer Caching'
description: 'Eliminate host message dispatch and redundant pipeline state setting via vectorized command buffers and state caching, double decode activation reuse via N_tile=8 GEMV, and eliminate prefill buffer allocation churn via retained aux and token caches.'
tags: [decision, compiler, prefill, gemm, gemv, sram, metal, gemma4, zero-jit]
status: stable
generated: { by: human:tung, at: '2026-09-04T18:42:00+07:00' }
sources:
  - id: gemma4-round2
    resource: /.okf/decisions/gemma4-round2-sram-staging-and-m32-tiling.md
    title: Threadgroup SRAM Staging and M=32 Prefill Tiling for Gemma 4
  - id: gemma4-batched-prefill
    resource: /.okf/decisions/gemma4-batched-prefill-and-pipelined-queue.md
    title: Batched Prefill GEMM and Pipelined Queue Execution for Gemma 4
  - id: llama-cpp-target
    resource: /.okf/decisions/llama-cpp-target.md
    title: llama.cpp performance target
---

# Decision

Following Round 2 (`exp-0147`), which reduced prefill TTFT to 1,409 ms (4.6x faster than exp-0145 baseline) and stabilized decode throughput at 10.33 tok/s, `llmopt` addresses the remaining host dispatch bottlenecks and SIMD-level activation reuse:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Vectorized Batch Buffer Binding & Pipeline State Caching                  │
│                                                                             │
│  - Bottleneck: Both llmopt_worker_loop and caml_llmopt_metal_batch_dispatch │
│    iterate over buffers with individual [enc setBuffer:offset:atIndex:]     │
│    Objective-C messages and unconditionally invoke [enc                     │
│    setComputePipelineState:], producing ~4,000 redundant messages per       │
│    decode token and during prefill command stream emission.                 │
│  - Solution: Cache current_pipeline to eliminate redundant state switches   │
│    across adjacent identical pipelines, and use the vectorized driver call   │
│    [enc setBuffers:offsets:withRange:] for all dispatches.                  │
│  - Result: Eliminates driver-level message dispatch overhead, streamlining   │
│    the execution loop in both decode and batch prefill.                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Wide Decode GEMV Tiling (N_tile = 8)                                     │
│                                                                             │
│  - Bottleneck: In llmopt_q4_0_linear_f16_x4, each SIMD group handles 4      │
│    columns, requiring 120 threadgroups for N=3840 and reloading activation  │
│    into registers across 2x more threadgroups.                              │
│  - Solution: Implement llmopt_q4_0_linear_f16_x8, computing 8 columns per   │
│    SIMD group (64 columns per 256-thread threadgroup). Accumulate into 8    │
│    float registers per thread, reusing the threadgroup SRAM activation      │
│    values across 8 columns simultaneously.                                  │
│  - Result: Halves total threadgroups dispatched during decode linear layers, │
│    doubles register activation reuse, and reduces threadgroup barrier churn.│
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Prefill Auxiliary Mask & Token Buffer Caching                            │
│                                                                             │
│  - Bottleneck: Every prefill invocation currently allocates host bytes and  │
│    creates new Metal GPU buffers for position_ids, causal attention mask,    │
│    sliding window mask, and input token ids, triggering GPU driver memory    │
│    allocation and synchronizations.                                         │
│  - Solution: Retain prefill token buffer and cache auxiliary mask buffers   │
│    in Serving_engine.t keyed by token count, reusing pre-allocated GPU      │
│    buffers across turns.                                                    │
│  - Result: Zero GPU buffer allocations during prefill execution.            │
└─────────────────────────────────────────────────────────────────────────────┘
```
