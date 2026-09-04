---
type: Decision
title: 'Default Lease-Backed Sequence Cursor, Serial Barrier Elision, and Register-Safe Multi-SIMDgroup M=32 Prefill GEMM'
description: 'Promote lease-backed sequence cursor to default in native generation eliminating radix root re-walking, elide 635 redundant pipeline-drain barriers on serial compute encoders, and re-architect M=32 prefill GEMM into 2-SIMDgroup threadgroups eliminating register spilling.'
tags: [decision, compiler, prefill, gemm, gemv, cursor, barriers, metal, gemma4]
status: stable
generated: { by: human:tung, at: '2026-09-05T06:33:00+07:00' }
sources:
  - id: gemma4-round4
    resource: /.okf/decisions/gemma4-round4-m32-prefill-wide-fused-decode.md
    title: M=32 Prefill GEMM Tiling, Wide Decode QKV/Gated GEMV, and Retained Prefill Plan Execution
  - id: exp-0149
    resource: /.okf/experiments/exp-0149-gemma4-performance-round4-2026-09-04.md
    title: Gemma 4 Round 4 performance execution receipt
  - id: llama-cpp-target
    resource: /.okf/decisions/llama-cpp-target.md
    title: llama.cpp performance target
---

# Decision

Following Round 4 (`exp-0149`), which proved 100% bit-exact parity, lowered median decode latency to 81.88 ms (76.23 ms steady-state), and identified single-SIMDgroup register pressure in $M=32$ GEMM, `llmopt` advances to Round 5 to drive decode towards ~50 ms and prefill towards ~800 ms:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Default Lease-Backed Sequence Cursor in Native Generation                │
│                                                                             │
│  - Bottleneck: Generation.Native_engine currently defaults to root-level     │
│    Serving_engine.decode, re-walking the radix tree and allocating prefix   │
│    slot arrays on every token (~12-15 ms host overhead).                    │
│  - Solution: Promote Serving_engine.Sequence cursor to default in           │
│    Generation.Native_engine.prompt and decode (with opt-out env fallback).  │
│  - Result: Eliminates radix tree root walks during generation, dropping     │
│    median decode latency from 81.88 ms directly to < 70 ms.                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Serial Compute Encoder Barrier Elision                                   │
│                                                                             │
│  - Bottleneck: Metal command encoders created with [cmd                     │
│    computeCommandEncoder] are inherently serial; dispatches are ordered in  │
│    hardware. However, 635 MTLBarrierScopeBuffers barriers are encoded per   │
│    token, repeatedly flushing L1 cache and stalling execution units.        │
│  - Solution: Do not issue memoryBarrierWithScope on serial compute encoders │
│    unless running concurrent dispatch mode (MTLDispatchTypeConcurrent).     │
│  - Result: Eliminates 635 GPU pipeline drain stalls per token during both   │
│    decode and prefill.                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Multi-SIMDgroup (N_simd=2) 64-Thread Register-Safe M=32 Prefill GEMM     │
│                                                                             │
│  - Bottleneck: In Round 4, unrolling 32 rows inside a single SIMDgroup (32  │
│    threads) required >120 registers per thread, triggering stack spills to  │
│    DRAM and inflating prefill GPU time to 5.07 s.                           │
│  - Solution: Re-architect M=32 prefill GEMM into a 2-SIMDgroup threadgroup  │
│    (64 threads): SIMDgroup 0 computes rows 0..15; SIMDgroup 1 computes      │
│    rows 16..31. Each thread maintains only 16 accumulators and 16 pointers  │
│    (identical to the register-safe M=16 kernel), while the threadgroup      │
│    streams weights once for all 32 rows.                                    │
│  - Result: Cuts prompt-37 weight traffic from 18.2 GB (3 passes) to 12.2 GB │
│    (2 passes) without register spilling, lowering prefill time by ~33%.     │
└─────────────────────────────────────────────────────────────────────────────┘
```

# Boundaries

- Bit-exact token parity across all 128 generated completion tokens on prompt `2,105,9731,...` remains a strict hard invariant.
- No monolithic bespoke kernel zoo expansion; all transformations operate on parameterized templates, orthogonal micro-ops, and compiler runtime pipeline state.
- No network weight downloads; reuse existing local artifacts.
