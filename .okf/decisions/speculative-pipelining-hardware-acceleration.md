---
type: Decision
title: 'Hardware Bitfield Extraction, Speculative Multi-Token Verification, and Queue-Coordinated Pipelining'
description: 'Break past the physical single-token memory bandwidth ceiling by combining single-cycle Apple GPU bitfield extraction (extract_bits), Split-K wide-layer reductions, and speculative draft token verification pipelined through the SRPT continuous batching serving queue.'
tags: [decision, compiler, speculative-decoding, pipelining, bitfield-extract, split-k, queue, srpt, apple-silicon, metal]
status: stable
generated: { by: 'process:antigravity', at: '2026-08-31T10:30:00+07:00' }
sources:
  - id: target-bandwidth-decision
    resource: /decisions/target-lfm25-350m-bandwidth.md
    title: Memory bandwidth physics and ERS feasibility
  - id: srpt-queue-decision
    resource: /decisions/srpt-queueing-serving-scheduler.md
    title: SRPT continuous batching queue scheduler
  - id: fewest-hops-decision
    resource: /decisions/fewest-hops-megakernel-compiler.md
    title: Fewest-Hops Whole-Block Megakernel Compiler
  - id: local-metal
    resource: /lib/metal.ml
    title: Metal Shading Language generator and megakernels
  - id: local-kernel-abi
    resource: /lib/kernel_abi.ml
    title: Shared shape-aware Linear tactic registry
  - id: local-serving-queue
    resource: /lib/serving_queue.ml
    title: Serving queue priority and batching coordinator
  - id: local-metal-runtime
    resource: /lib/metal_runtime.ml
    title: Native Metal runtime and concurrent command encoder
---

# 1. Problem Statement & Physics

Single-stream autoregressive decoding on Apple Silicon is firmly in the memory-bandwidth-bound regime ($\approx 175\text{ GB/s}$ sustained on Apple M4 Pro). Because `llmopt` already extracts $\ge 95\%$ of practical achievable memory bandwidth across `SmolLM2`, `Qwen3.5`, and `Gemma-4-E2B`, further scalar kernel tuning produces diminishing returns ($<2\%$).

To achieve a major speedup step ($2\times - 3\times$ effective tokens/sec), the architecture must address three distinct optimization levels:

1. **Microarchitectural Instruction Efficiency**: Superblock-256 dequantization (`Q4_K`, `Q5_K`, `Q6_K`) incurs substantial ALU overhead unpacking 6-bit sub-block scales and 2-bit high bits. Apple GPUs have native hardware bitfield extract instructions (`metal::extract_bits`) that can execute packed bit extraction in 1 cycle.
2. **Algorithmic Bandwidth Multiplication (Speculative Verification)**: Streaming $3.17\text{ GB}$ of Gemma weights takes $\sim 18\text{ ms}$ regardless of whether $M=1$ token or $M=4$ tokens are evaluated. By verifying $K=3-5$ candidate draft tokens in a single forward pass, the effective TPOT drops from $18\text{ ms}$ to $4.5 - 6.0\text{ ms}$ per accepted token.
3. **Queue-Coordinated Pipelining & Concurrency**: Utilizing our continuous batching queue facility (`Serving_queue`) to pipeline draft proposal generation, prefill chunk slicing, and verified decode execution concurrently across Metal queues (`MTLDispatchTypeConcurrent`).

```
   ┌──────────────────────────────────────────────────────────────┐
   │                   Serving Queue (SRPT Engine)                │
   │  - Prioritizes active decode steps over monolithic prefills  │
   │  - Coordinates speculative draft validation windows          │
   └──────────────────────────────┬───────────────────────────────┘
                                  │ Pipelined Execution
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │         Speculative Verification Megakernel (M = 2..5)       │
   │  - Hardware `extract_bits` for 1-cycle K-quant scale decode  │
   │  - Split-K parallel reduction on wide MLP layers             │
   │  - Streams 3.17 GB weights ONCE to verify K candidate tokens │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │    Selective KV-Cache Commit & Radix Rollback (M_accept <= K) │
   │  - Accepts prefix tokens matching greedy/sampled draft       │
   │  - Rolls back rejected speculative token state in 1 cycle    │
   └──────────────────────────────────────────────────────────────┘
```

---

# 2. Key Architectural Components

## 2.1. Hardware `extract_bits` Superblock Dequantizers

Superblock formats (`Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS`) encode sub-block scales and mins using non-byte-aligned bit distributions (e.g. 6-bit scales packed into 12-byte blocks).
* **Current Implementation**: Emits multiple integer shift, mask, and table-lookup instructions per 32-element sub-block.
* **Target Optimization**: Replaces manual bitwise sequences with `metal::extract_bits(src, offset, count)` in MSL, compiling directly to single-cycle Apple GPU bitfield extract hardware primitives.

## 2.2. Split-K Wide-Layer Reductions

For layers where the reduction dimension $K \ge 4096$ (such as Gemma's intermediate MLP dimension $K=6144$ or vocabulary projection $K=262,144$):
* Split the inner reduction across $P=2$ or $P=4$ threadgroups.
* Each threadgroup accumulates an independent partial dot-product directly into threadgroup memory or SIMD registers, followed by a final intra-threadgroup reduction.
* Maximizes GPU execution unit occupancy on smaller prompt batches and models with low layer counts.

## 2.3. Speculative Verification Megakernel & Tree Mask Evaluation

* The main model forward operates at batch size $M = K + 1$ (the current token plus $K$ speculative candidates).
* Attention incorporates a tree attention causal mask allowing parallel evaluation of all speculative branches in a single forward pass.
* Weights are streamed from DRAM exactly once per verification step.
* If $\bar{\alpha}$ tokens are accepted on average ($\bar{\alpha} \approx 2.5 - 3.5$), effective TPOT is reduced by:
  $$\text{TPOT}_{\text{eff}} = \frac{\text{Forward Latency}}{\bar{\alpha}} \approx \frac{18\text{ ms}}{3} = 6.0\text{ ms}$$

## 2.4. Queue-Coordinated Asynchronous Pipelining

In [`lib/serving_queue.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_queue.ml) and [`lib/serving_engine.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_engine.ml):
* **Draft/Target Pipelining**: When using a lightweight draft model or speculative head, the draft proposal forward for step $t+1$ overlaps concurrently with the target verification writeback for step $t$ using separate Metal command buffers.
* **Atomic State Commit & Rollback**: The physical KV cache allocates speculative slots optimistically and performs a single metadata pointer update committing accepted tokens and releasing rejected slots in $O(1)$ time.

---

# 3. Success Gates & Verification Criteria

1. **Bitfield Dequantization**: 100% numerical bit-exact parity with existing reference logits and argmax IDs across all GGUF K-quant layouts.
2. **Split-K Reduction**: Equal or improved latency on $K \ge 4096$ linear projections with exact numerical output.
3. **Speculative Verification**: Sustained $\ge 2.0\times$ effective decode token throughput under multi-turn generation on Gemma-4-E2B and Qwen3.5.
4. **Queue Pipelining**: Zero thread blocking, zero race conditions under concurrent client traffic, passing all strict OKF and Ninja test suites.
