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

---

# Empirical Results (Receipt: `mellum21-benchmark-comparison-2026-10-25_100540.json`)

### 1. Prefill / Prompt Processing Throughput

| Prompt Length ($p$) | llama.cpp (`MTL,BLAS`) | llmopt `Tile32_Affine` (2026-10-23) | llmopt `Tile32_Affine` (Current) | Speedup vs Prior | Ratio to llama.cpp |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4** | 125.30 t/s | 6.51 t/s | **33.03 t/s** | **5.07x** | 0.26x |
| **16** | 160.70 t/s | 30.10 t/s | **92.80 t/s** | **3.08x** | 0.58x |
| **64** | 462.00 t/s | 67.65 t/s | **122.74 t/s** | **1.81x** | 0.27x |
| **128** | 635.35 t/s | 86.93 t/s | **179.21 t/s** | **2.06x** | 0.28x |
| **512** | 965.26 t/s | 106.67 t/s | **176.76 t/s** | **1.66x** | 0.18x |

### 2. Autoregressive Decode Throughput & Latency

| Generated Tokens ($n$) | llama.cpp (`tg`) | llmopt Steady (2026-10-23) | llmopt Steady (Current) | llmopt Median Latency | llmopt Min Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **16** | 92.97 t/s | 68.80 t/s | **82.77 t/s** | **11.94 ms** | **10.93 ms** |
| **64** | 94.29 t/s | 63.77 t/s | **73.64 t/s** | **13.54 ms** | **11.00 ms** |
| **128** | 91.39 t/s | 53.89 t/s | **61.81 t/s** | **16.08 ms** | **11.15 ms** |

*Key finding*: At $n=16$, `llmopt`'s steady decode throughput reached **82.77 tokens/sec** with peak minimum latency of **10.93 ms/token**—closing within 89% of `llama.cpp`'s 92.97 tok/s (10.75 ms/token). Both engines generate completely valid, coherent tokens without NaNs or numerical degradation.
