---
type: Decision
title: 'Batched Prefill GEMM and Pipelined Queue Execution for Gemma 4'
description: 'Close the prefill and decode performance gap with llama.cpp on Apple Silicon Metal by implementing M-tiled prefill GEMM (M >= 4) to eliminate redundant weight streaming from DRAM, and pipelined depth-2 queue execution to eliminate host-GPU driver bubbles, while rejecting bespoke kernel-zoo proliferation.'
tags: [decision, compiler, prefill, gemm, pipelining, ring-queue, metal, gemma4, zero-jit]
status: stable
generated: { by: human:tung, at: '2026-09-04T16:55:00+07:00' }
sources:
  - id: parametric-epilogues
    resource: /.okf/decisions/parametric-epilogues-and-hardware-tiling.md
    title: Adopt Parametric Epilogue Inlining and Hardware-Aware Prefill Tiling
  - id: srpt-queueing
    resource: /.okf/decisions/srpt-queueing-serving-scheduler.md
    title: SRPT and Queueing-Theoretic Serving Scheduler
  - id: llama-cpp-target
    resource: /.okf/decisions/llama-cpp-target.md
    title: llama.cpp performance target
---

# Decision

`llmopt` adopts two core architectural mechanisms to reach performance and throughput parity with `llama.cpp` for Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal, while explicitly **rejecting the kernel zoo anti-pattern**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Rejection of Kernel Zoo in Favor of Compiler Micro-Ops                   │
│                                                                             │
│  - No bespoke monolithic kernels (e.g. q4_0_llama_mv_add, q4_0_gated_mv)    │
│  - Orthogonal Dequant Micro-Op: pure block unpack + scale formula           │
│  - Parameterized Loop Templates: vector-streaming GEMV (M=1), tiled (M>=4) │
│  - Composable Epilogues: DAG fusion for residual add, GeGLU, QKV splits    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Batched Prefill GEMM Tiling (M >= 4)                                     │
│                                                                             │
│  - Current Bottleneck: M=37 runs row-by-row, streaming 6.08 GB weights 37x │
│    from DRAM (~225 GB traffic, taking 1.5 - 2.0s on DRAM PHY bandwidth)    │
│  - Solution: Tile M in threadgroup SRAM; stream weights ONCE across tile    │
│  - Result: 6.08 GB DRAM traffic total (~30 ms streaming time on M4 Pro),    │
│    dropping TTFT from ~2.95s to < 350 ms (matching llama.cpp 325 ms)        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Pipelined Decode Execution via the AOT Queue Model                       │
│                                                                             │
│  - Current Bottleneck: Synchronous encode-commit-wait creates CPU bubbles   │
│  - Solution: Utilize Ring_queue with configurable pipeline depth (D=2)      │
│  - Submit step t+1 before blocking on step t; keep token argmax in GPU VRAM │
│  - Result: Zero GPU idle bubbles, reducing inter-token decode latency       │
└─────────────────────────────────────────────────────────────────────────────┘
```

# Context & Comparative Analysis

Benchmarking Gemma 4 (12B QAT UD-Q4_K_XL) on Apple M4 Pro reveals a substantial performance gap between `llmopt` and `llama.cpp`:

| Metric | Target (`llama-bench`) | Current `llmopt` | Gap | Root Cause |
| :--- | :--- | :--- | :--- | :--- |
| **Prefill (prompt-37)** | `113.69 tok/s` (325 ms) | `12.5 tok/s` (2,957 ms) | **9.1x slower** | 37x redundant DRAM weight streaming |
| **Decode (gen-32)** | `27.37 tok/s` (36.5 ms/tok) | `13.5 tok/s` (74.2 ms/tok) | **2.0x slower** | Synchronous CPU-GPU bubbles & memory coalescing |

Inspection of high-performance single-purpose runtimes (such as Perplexity's open-source `lily` for Apple Silicon) and `llama.cpp` confirms that the prefill gap is entirely dominated by weight reuse over prompt batches ($M \ge 4$), while the decode gap is driven by driver synchronization bubbles and memory bus coalescing.

# Principles

### 1. Rejecting the Kernel Zoo
Adding one-off handwritten kernels for each quantization layout and operation combination creates an unmaintainable combinatorial explosion ($N_{\text{quant}} \times M_{\text{fusion}} \times K_{\text{tiling}}$). `llmopt` remains an optimizing compiler:
- The dequantization formula for each layout (Q4_0, Q4_K, Q8_0, etc.) is an isolated micro-op functor.
- The execution geometry is an abstract template (GEMV for $M=1$, tiled GEMM for $M \ge 4$).
- Epilogues (residual add, GeGLU, multi-head split) remain composable DAG transformations.

### 2. Batched Prefill GEMM ($M \ge 4$)
For prompt processing ($M=37$), activations for all $M$ tokens are loaded into threadgroup storage or registers. The weights are read contiguously from DRAM **once**, computing the matrix-matrix product across all $M$ tokens concurrently. This slashes DRAM traffic from 225 GB to 6.08 GB.

### 3. Asynchronous Queue Pipelining (Depth $D=2$)
`llmopt`'s existing `Ring_queue` (`native/ocaml_metal_stubs.m`) already implements submission (`sq`) and completion (`cq`) ring buffers. By enabling depth-2 pipelining in the worker loop:
- Command buffer $t+1$ is encoded and committed to Metal while command buffer $t$ is actively executing on the GPU.
- The GPU hardware command processor transitions between tokens with zero idle gap.
- Argmax token IDs remain resident in GPU buffers between consecutive decode steps.
