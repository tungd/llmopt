---
type: Decision
title: 'Threadgroup SRAM Staging and M=32 Prefill Tiling for Gemma 4'
description: 'Eliminate 3.6 GB/token of redundant activation DRAM reloads during decode via cooperative threadgroup SRAM staging, and slash prefill weight passes from 10 to 2 via M=32 tiling, closing the remaining throughput gap with llama.cpp without bespoke kernels.'
tags: [decision, compiler, prefill, gemm, gemv, sram, metal, gemma4, zero-jit]
status: stable
generated: { by: human:tung, at: '2026-09-04T18:05:00+07:00' }
sources:
  - id: gemma4-batched-prefill
    resource: /.okf/decisions/gemma4-batched-prefill-and-pipelined-queue.md
    title: Batched Prefill GEMM and Pipelined Queue Execution for Gemma 4
  - id: parametric-epilogues
    resource: /.okf/decisions/parametric-epilogues-and-hardware-tiling.md
    title: Adopt Parametric Epilogue Inlining and Hardware-Aware Prefill Tiling
  - id: llama-cpp-target
    resource: /.okf/decisions/llama-cpp-target.md
    title: llama.cpp performance target
---

# Decision

Following Round 1 (`exp-0146`), which reduced prefill latency by 3.9x (1,668 ms) and decode latency by 1.7x (82.7 ms), `llmopt` addresses the two primary hardware bottlenecks limiting Gemma 4 parity with `llama.cpp` (target: prefill 325 ms, decode 36.5 ms):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Threadgroup SRAM Activation Staging in Decode GEMV (M=1)                 │
│                                                                             │
│  - Bottleneck: 8 SIMD groups per threadgroup reload activation[k] directly  │
│    from DRAM. For N=15,360, this causes 3,840 redundant reloads (29.5 MB    │
│    per projection, ~3.6 GB DRAM traffic per decode token across 48 layers). │
│  - Solution: Stage activation into threadgroup half s_act[...] once per     │
│    threadgroup cooperatively using all 256 threads before computing dot     │
│    products across columns.                                                 │
│  - Result: Eliminates 3.6 GB/token of DRAM traffic, dropping decode time    │
│    toward the 36.5 ms llama.cpp target.                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Prefill GEMM Tile Expansion (M_tile = 32)                                │
│                                                                             │
│  - Bottleneck: M_tile = 4 requires ceil(37/4) = 10 tiles for prompt-37.     │
│    The 6.08 GB weights are streamed from DRAM 10 separate times (60.8 GB    │
│    traffic, consuming ~818 ms GPU time).                                    │
│  - Solution: Expand prefill tiling to M_tile = 32 with boundary predicates. │
│    For prompt-37, ceil(37/32) = 2 tiles, streaming weights only twice.      │
│  - Result: 80% reduction in prefill weight streaming passes (60.8 GB down   │
│    to 12.16 GB), dropping TTFT toward the 325 ms target.                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Preservation of Compiler Template Abstractions (No Kernel Zoo)           │
│                                                                             │
│  - Both optimizations operate within generic parameterized templates:      │
│    - Micro-op unpack remains decoupled from memory staging.                 │
│    - Grid sizing and tactic dispatch select templates dynamically by M.     │
│    - Zero handwritten bespoke kernels per operation/quant combination.       │
└─────────────────────────────────────────────────────────────────────────────┘
```
