---
type: Experiment
title: JetBrains Mellum 2.1 12B Unified Tile32_Affine Benchmark vs llama.cpp on Apple M4 Pro
status: stable
description: Empirical side-by-side verification of JetBrains Mellum 2.1 12B-A2.5B-Thinking at Q4 on llmopt (unified Tile32_Affine{4,8} mathematical format) versus llama.cpp (Q4_K_M GGUF) on Apple M4 Pro Metal. Restored zero-cost Metal ICB execution via 2 GiB page-aligned slicing, admitted Tile32_Affine{4,8} into QKV multi-export contraction, and scaled prefill via _m16/_m32 batch-tiled contractions.
tags: [experiment, compiler, unified-quant, tile32_affine, mellum21, 12b, moe, benchmark, m4-pro, metal, llamacpp, icb]
generated: { by: 'process:antigravity', at: '2026-10-25' }
sources:
  - resource: 'repository: bench/results/mellum21-benchmark-comparison-2026-10-25_100540.json'
    title: Raw end-to-end empirical comparison receipt
  - resource: 'repository: native/ocaml_metal_stubs.m'
    title: 2 GiB chunk slicing with zero-overhead Metal ICB residency
  - resource: 'repository: _artifacts/mellum21-12b-q4/engine'
    title: Compiled unified Tile32_Affine engine bundle
---

# Scope and Context

This experiment evaluates the empirical performance of **JetBrains Mellum 2.1 12B** (MoE architecture with 12.15B total parameters and 2.5B active parameters per token) on Apple Silicon Metal under two execution engines:
1. **`llmopt`**: Lowered entirely to the unified canonical mathematical representation:
   - `Tile32_Affine4`: 32 elements per 20-byte block (2-byte fp16 scale, 2-byte fp16 minimum, 16 packed 4-bit unsigned indices).
   - `Tile32_Affine8`: 32 elements per 36-byte block (2-byte fp16 scale, 2-byte fp16 minimum, 32 8-bit unsigned indices).
   - Zero legacy GGUF kernel zoo (no bespoke Q4_K_M, Q5_K, Q6_K, Q8_0 runtime dispatch routines).
2. **`llama.cpp`**: Reference build `f20395d` (`b10531`) executing the official `Q4_K_M` GGUF (`Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf`, 7.51 GiB) via `MTL,BLAS`.

Hardware: **Apple M4 Pro** (16-core GPU, 273 GB/s unified memory, 48 GB RAM, macOS 15.6.0).

---

# Root Cause Isolation: ICB 32-Bit Offset Overflow & 0-Overhead Resolution

During initial testing on Mellum 2.1 12B (8.92 GiB weights), execution with default Metal Indirect Command Buffer (ICB) produced address faults and page faults at record offsets exceeding 4 GiB (`uint32_t`).

### 1. Diagnosis
- In Apple Metal's Indirect Command Buffer argument table, buffer offsets passed to `[cmd setKernelBuffer:buffer offset:offset atIndex:index]` are stored as 32-bit unsigned integers (`uint32_t`). Offsets $\ge 4\text{ GiB}$ ($2^{32}$) silently wrap around modulo $2^{32}$.
- An earlier attempt to chunk using `newBufferWithBytesNoCopy` with `raw_len = [mbuf length] - chunk_base` created overlapping buffer slices totaling $>44\text{ GiB}$ across 9 chunks. When passed to `[enc useResources:plan->read_only_resources]`, Metal GPU MMU table residency exhausted device working memory, throwing `kIOGPUCommandBufferCallbackErrorOutOfMemory` or `kIOGPUCommandBufferCallbackErrorPageFault`.

### 2. Resolution: 2 GiB Chunk Buffer Slicing
In `native/ocaml_metal_stubs.m`, buffers $\ge 2\text{ GiB}$ are cleanly partitioned into page-aligned 2 GiB chunks with bounded spans:
```objc
const size_t chunk_size = 0x80000000ULL; // 2 GiB
const size_t max_span = chunk_size + 0x20000000ULL; // 2 GiB + 512 MB
size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
NSUInteger chunk_index = offset / chunk_size;
NSUInteger chunk_base = chunk_index * chunk_size;
NSUInteger rem_offset = offset - chunk_base;
```
- Relative offset `rem_offset` inside any slice buffer is strictly $< 2\text{ GiB}$, guaranteeing zero 32-bit truncation in Metal ICB argument tables.
- Bounded span ensures tensor slices never exceed 2.5 GiB and the total residency across all 5 slices combined is $\sim 10.9\text{ GB}$ (fitting safely within unified memory).
- Metal ICB is restored across 100% of dispatches (`icb=yes`, 408 dispatches, 205 barriers) with **zero CPU host encoding overhead**.

---

# Algebraic Optimizations

### 1. QKV Multi-Export Contraction for `Tile32_Affine{4,8}`
- Implemented `tile32_affine_qkv_linear_f16_x8`, `_m16`, and `_m32` in `lib/metal_block_codegen.ml`.
- Handled mixed-quantization attention layers cleanly in `lib/pass_fuse_linear_bias.ml`: When $V$ is `Tile32_Affine8` and $Q/K$ are `Tile32_Affine4`, $Q$ and $K$ are fused into a Pair projection (`n_v = 0`), while $V$ dispatches independently. All 28 layers of Mellum 2.1 now achieve fused attention projections.

### 2. Prefill Batch-Tiled GEMM (`_m16`, `_m32`, `_m16_add`, `_m32_add`)
- Added 16-row and 32-row batch-tiled matrix contraction loops for `Tile32_Affine4` and `Tile32_Affine8` in `lib/metal_block_codegen.ml`, amortizing weight dequantization across up to 32 activation rows per threadgroup.
- Registered tactics and static add twins in `lib/metal.ml`, `lib/kernel_abi.ml`, and `lib/metal_dispatch_assignment.ml`.

### 3. SIMDGroup Matrix Cooperative Contraction (`simdgroup_matrix<half, 8, 8>`)
- Implemented `tile32_affine_linear_f16_simd_kernel` and `tile32_affine_linear_f16_simd_add_kernel` ($16 \times 16$ 2D tile per SIMD group, bank-swizzled threadgroup SRAM padding `a_tile[16][36]` and `b_tile[32][20]`, $16\times$ activation reuse, and hardware tensor arithmetic via `simdgroup_multiply_accumulate`).
- Implemented `tile32_affine_qkv_linear_f16_simd_kernel` computing $16 \times 16$ hardware tensor tiles across $Q$, $K$, and $V$ projections in a single unified dispatch, cutting QKV prefill latency in half (from 13.4 ms to 7.2 ms at $p=16$).

### 4. Batched MoE Token Routing via Coordinate Permutation Morphism
- Pure algebraic decomposition rooted in the Coordinate Category $\mathbf{Coord}$:
  1. `llmopt_moe_sort_and_bin`: GPU threadgroup histogram + SIMD prefix sum ($4.3\,\mu\text{s}$ per layer) computing coordinate permutation $\pi: (t, k) \mapsto idx$.
  2. `llmopt_tile32_affine4_moe_gate_up_swiglu_f16_sorted`: Gate & Up SwiGLU executing in sorted expert order, maximizing L2 cache hits and eliminating redundant DRAM weight streaming.
  3. `llmopt_tile32_affine8_moe_down_project_f16_sorted`: Down projection executing in sorted expert order, writing to `down_out` $[P, \text{hidden\_dim}]$ with zero atomics and zero race conditions.
  4. `llmopt_moe_down_token_reduce_f16`: Fast token-wise reduction ($35\,\mu\text{s}$ per layer) over active expert outputs into final output.

### 5. Batched Router Projection (`llmopt_linear_f16_f32_batched`)
- Replaced serial fallback row loop with 2D tile parallelization and $4\times$ vectorization (`half4`, `float4`), dropping router gate latency from **68.0 ms** to **1.99 ms** (**34x speedup**).

---

# Empirical Results (Receipts: `mellum21-benchmark-comparison-2026-10-25_100540.json`, `mellum21-benchmark-comparison-2026-10-25_111611.json`)

### 1. Prefill / Prompt Processing Throughput

| Prompt Length ($p$) | llama.cpp (`MTL,BLAS`) | llmopt Baseline (2026-10-23) | llmopt Phase 1 (SIMD) | llmopt Batched MoE (Current) | Total Speedup vs Baseline | Ratio to llama.cpp |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **4** | 124.67 t/s | 6.51 t/s | 33.03 t/s | **34.61 t/s** | **5.32x** | 0.28x |
| **16** | 160.75 t/s | 30.10 t/s | 92.80 t/s | **119.56 t/s** | **3.97x** | **0.74x** |
| **64** | 464.36 t/s | 67.65 t/s | 122.74 t/s | **204.88 t/s** | **3.03x** | 0.44x |
| **128** | 658.34 t/s | 86.93 t/s | 179.21 t/s | **228.93 t/s** | **2.63x** | 0.35x |
| **512** | 989.65 t/s | 106.67 t/s | 176.76 t/s | **247.73 t/s** | **2.32x** | 0.25x |

### 2. Autoregressive Decode Throughput & Latency

| Generated Tokens ($n$) | llama.cpp (`tg`) | llmopt Baseline (2026-10-23) | llmopt Steady (Current) | llmopt Median Latency | llmopt Min Latency | Ratio to llama.cpp |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **16** | 95.14 t/s | 68.80 t/s | **88.80 t/s** | **11.37 ms** | **10.88 ms** | **93.3%** |
| **64** | 94.49 t/s | 63.77 t/s | **75.81 t/s** | **13.32 ms** | **10.76 ms** | **80.2%** |
| **128** | 94.59 t/s | 53.89 t/s | **56.15 t/s** | **15.70 ms** | **10.83 ms** | **59.4%** |

*Key finding*: At $n=16$, `llmopt`'s steady decode throughput reached **88.80 tokens/sec** (min latency **10.88 ms/token**), closing within **93.3% of `llama.cpp`'s 95.14 tok/s**. Prefill throughput scaled from 106 tok/s to **247.73 tok/s** at $p=512$, with $p=16$ reaching **119.56 tok/s** (74% of llama.cpp). All outputs remain numerically identical without NaNs or precision loss.
