---
type: Decision
title: 'Fewest-Hops Whole-Block Megakernel Compiler Architecture'
description: 'Replace the 864-command operator-at-a-time micro-dispatch model with whole-block megakernels (SwiGLU FFN, ShortConv, Attention, LM-Head) that execute an entire decode step in approximately 34 hops with zero DRAM activation round-trips.'
tags: [decision, compiler, megakernel, fusion, metal, fewest-hops, decode, tpot, ttft, sram]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T07:05:00Z' }
sources:
  - id: target-llama-cpp
    resource: /decisions/llama-cpp-target.md
    title: Primary external performance target (llama.cpp)
  - id: macro-fusions
    resource: /decisions/macro-operator-fusions.md
    title: Initial pairwise macro-operator fusions
  - id: local-passes
    resource: /lib/passes.ml
    title: Compiler optimization pass pipeline
  - id: local-metal
    resource: /lib/metal.ml
    title: Metal Shading Language kernel emitter
---

# Decision

The compiler's optimization objective is to produce optimal Metal kernels with the **fewest possible hops** (kernel dispatches) between the model input and output. Instead of emitting an interpreted sequence of 864 micro-commands with 209 barriers, the compiler will fold entire transformer subgraphs into coarse-grained megakernels.

# Motivation & Physics

In LFM2.5-350M decode ($M=1$):
* Hidden dimension $K = 1,024$; activation tensor is $1,024 \times 2\text{ bytes} = 2\text{ KB}$ FP16 ($9\text{ KB}$ inside the 4,608-wide FFN).
* Apple Silicon (M4 Pro) provides $32\text{ KB}$ of L1 threadgroup SRAM per compute unit and tens of megabytes of System Level Cache (SLC).
* In the micro-command model, after every linear, norm, slice, or activation, the $2\text{ KB}$ vector is written to global DRAM and immediately read back by the next kernel. Over 864 commands, DRAM round-trips consume excessive memory bandwidth and incur massive Objective-C command encoding latency (~3.3 ms CPU time).
* By keeping activations resident in GPU registers and threadgroup SRAM across an entire block, activations never leave on-chip memory. Only static weights stream from DRAM, enabling execution at the theoretical memory bandwidth floor ($350\text{ MB} / 200\text{ GB/s} \approx 1.75\text{ ms}$).

# Target Architecture: Whole-Block Megakernels

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Hop 0: Token Embedding Gather (1 hop)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 10 ShortConv Layers (1 hop each = 10 hops):                                 │
│   x ──► [ RMSNorm + InProj + StateShift + Conv3 + Gate + OutProj + Add ] ──► x│
├─────────────────────────────────────────────────────────────────────────────┤
│ 6 Attention Layers (2 hops each = 12 hops):                                 │
│   Hop A: x ──► [ RMSNorm + QKV Linear + RoPE + CacheWrite ] ──► (Q, KV)     │
│   Hop B: (Q, KV) ──► [ PagedAttention + OutProj + Add ] ──► x               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 16 SwiGLU FFN Layers (1 hop each = 16 hops):                                │
│   x ──► [ RMSNorm + (W1_silu * W3) + W2 Down + Add ] ──► x                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Hop 39: LM Head & Argmax (1 hop):                                           │
│   x ──► [ RMSNorm + LM_Head Q8 Linear + Threadgroup Argmax ] ──► token_id   │
└─────────────────────────────────────────────────────────────────────────────┘
Total Model Execution: ~40 hops (down from 864 commands)
```

# Parallelism & Core Saturation Contract

Earlier attempts at macro-fusion suffered because kernels were dispatched with a single threadgroup ($M=1$), leaving 47 of 48 GPU cores idle.

Every megakernel MUST satisfy:
1. **Grid Parallelization:** The grid MUST be partitioned across the output channels into $\ge 48$ threadgroups (saturating all GPU cores).
2. **SIMDgroup Reductions:** Reductions across the $K=1,024$ inner dimension use 32-thread SIMD cooperative reductions with 128-byte cache-coalesced loads (`char4` / `half4`).
3. **Register / SRAM Residence:** Intermediate activations ($W_1 \cdot x$, $W_3 \cdot x$, SiLU output) are staged in registers or threadgroup SRAM without device memory allocations.
