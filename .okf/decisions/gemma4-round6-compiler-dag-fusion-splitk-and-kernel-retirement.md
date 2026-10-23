---
type: Decision
title: 'Gemma 4 Round 6: Compiler DAG Fusion, Split-K Parameterized GEMV, and Bespoke Kernel Retirement'
description: 'Establishes the architectural mandate to eliminate bespoke monolithic kernels, lower all linear projections through parameterized block codegen with Split-K concurrency, fuse elementwise activations and residual additions via compiler DAG passes, and absorb KV-cache packing to close the throughput gap to llama.cpp.'
status: stable
decision_status: approved
tags: [decision, compiler, dag-fusion, split-k, gemv, kv-cache, metal, gemma4]
generated: { by: antigravity/2.0, at: '2026-09-05T06:50:00+07:00' }
sources:
  - id: round5-decision
    resource: /.okf/decisions/gemma4-round5-sequence-cursor-barrier-elision-multi-simd.md
    title: Gemma 4 Round 5 Sequence Cursor, Serial Barrier Elision, and Multi-SIMDgroup M=32 Prefill GEMM
  - id: target-profile
    resource: /_artifacts/gemma4-12b-qat-round5-2026-09-05/target.json
    title: Target hardware profile for Apple M4 Pro
  - id: round5-receipt
    resource: /bench/results/uoc-gemma4-performance-round5-2026-09-05.txt
    title: Gemma 4 Round 5 performance execution receipt
---

# Context

In Rounds 2 through 5, `llmopt` improved Gemma 4 12B QAT decode performance from 141.57 ms down to 81.44 ms. However, decode latency remains far above the hardware memory roofline:
- The model weights are 6.2 GB. At the measured 228.8 GB/s memory bandwidth on Apple M4 Pro, streaming weights once per decode token requires only ~27.1 ms (matching `llama.cpp`'s ~36.5 ms / 27.37 tok/s).
- At 81.44 ms, `llmopt` achieves only ~76 GB/s (~33% of hardware bandwidth).

Simultaneously, the codebase drifted into an architectural anti-pattern:
1. Handwritten monolithic C++ kernels accumulated in `lib/metal.ml` (`llmopt_q4_0_qkv_linear_f16_x8`, `llmopt_q4_0_gated_linear_f16`, `llmopt_q4_0_linear_f16_m32_add`, `llmopt_q4_0_qkv_linear_f16_m32`).
2. Ad-hoc string-suffix routing proliferated in `lib/metal_runtime.ml` (`String.ends_with ~suffix:"_x8"`, `_m32`, etc.).
3. Decode steps execute ~670 separate dispatches per token across 48 layers (14 dispatches/layer plus 48 trailing `cache_pack` dispatches), flushing L1 and bouncing intermediate activations in and out of DRAM.
4. Decode GEMV (`_x8`) launches only 512 total threads ($4096 / 8$), severely under-occupying the 16 GPU cores of the M4 Pro and starving memory controllers.

# Decision

1. **Retire Bespoke Kernel Variants**:
   - Discontinue and remove all handcrafted monolithic kernels from `lib/metal.ml`.
   - All quantized linear projections must be emitted strictly from the schema-driven, parameterized code generator in `lib/metal_block_codegen.ml`.

2. **Split-$K$ Parameterized Decode GEMV ($M=1$)**:
   - In `lib/metal_block_codegen.ml`, introduce parameterized Split-$K$ execution ($K_{\text{split}} \in \{1, 2, 4, 8\}$) for decode GEMV.
   - Partitioning $K$ across multiple SIMDgroups per column increases GPU occupancy from 512 threads to 2,048–4,096 threads, saturating the 16 GPU cores and memory controllers to drive memory bandwidth toward the 228 GB/s roofline.

3. **Compiler DAG Epilogue & Prologue Fusion**:
   - In `lib/pass_fuse_epilogues.ml` and `lib/passes.ml`, implement generic algebraic fusion:
     - Epilogue: Fuse bias add, residual add, and SwiGLU/GELU elementwise multiplication directly into producer linear operators.
     - Prologue: Fuse RMSNorm input normalization directly into linear activation loads via cooperative threadgroup SRAM staging.
   - Reduce decode dispatches per layer from 14 down to 3–4.

4. **Direct KV-Cache Packing**:
   - Absorb the 48 trailing standalone `llmopt_cache_pack_attention_q8_pair_simd` dispatches into in-stream pipeline execution, writing Key/Value activations directly into paged $Q_8$ cache slots.

# Invariants

- **Bit-Exact Token Parity**: All 128 completion tokens must match the pinned golden reference sequence bit-for-bit (`100, 45518, 107, 14947, 135383,...`).
- **Zero Remote Downloads**: Always reuse local weights from `_artifacts/gemma4-12b-qat-primitive-schedule-2026-09-03/source/target.weights.llmopt`.
- **No Bespoke Kernel Proliferation**: Orthogonal micro-ops and compiler DAG lowerings only.
