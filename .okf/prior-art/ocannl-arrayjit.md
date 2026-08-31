---
type: PriorArt
title: 'Ahrefs OCANNL and the arrayjit loop-nest compiler'
description: 'Analysis of Ahrefs OCANNL (v1.0.1) and arrayjit as a loop-nest transformation, tiling, and multi-backend compiler for deep learning, evaluated as an intermediate lowering tier for llmopt.'
tags: [prior-art, ocannl, arrayjit, compiler, loop-lowering, metal, cuda, rocm]
status: stable
generated: { by: human:tung, at: '2026-08-31T13:58:00+07:00' }
sources:
  - id: ocannl-repo
    resource: https://github.com/ahrefs/ocannl
    title: OCANNL GitHub Repository
  - id: arrayjit-lib
    resource: /tmp/ocannl/arrayjit/lib/
    title: arrayjit compiler subsystem sources
---

# Prior Art: Ahrefs OCANNL and `arrayjit`

## Overview

OCANNL (*OCaml Compiles Algorithms for Neural Networks Learning*) is a compiled deep learning and autodiff framework written in pure OCaml by Ahrefs (Łukasz Stafiniak). Its optimizing backend compiler, **`arrayjit`**, lowers high-level tensor expressions into an imperative low-level representation (`Low_level.t`), performs backend-independent loop optimizations, applies modular schedule transforms (`Schedule.optop`), and emits high-performance C, Metal MSL (3.1+), NVIDIA CUDA PTX, and AMD ROCm/HIP C++ (via `hipjit`).

## Core Architecture of `arrayjit`

1. **Intermediate Representations**:
   - **Assignment IR (`Assignments.t` / `%cd`)**: High-level tensor assignments with projection indexing and reduction operators.
   - **Low-Level IR (`Low_level.t`)**: A C-like scalar language featuring `For_loop`, `Set`, `Get`, `Local_scope`, `Tile_mma`, and bitwise/arithmetic `scalar_t` expressions.
2. **Backend-Independent Optimization Pipeline**:
   - **Virtualization & Inlining (`virtual_llc`)**: Inlines intermediate tensor computations into registers and local accumulators (`Local_scope`), eliminating DRAM round-trips.
   - **Common Subexpression Elimination (`hoist_cross_statement_cse`)**: Hoists redundant scalar computations out of loops into local registers.
   - **Interval Folding & Simplification (`simplify_llc`)**: Constant folds loop bounds, branch guards, and affine indexing.
3. **Modular Schedule Transformations (`Schedule.optop`)**:
   - `Split`, `Swap`, `Pad`, `Unroll`, `Partition`: Structural loop nest manipulations.
   - `Stage`: Automatically stages memory reads into threadgroup/shared memory with XOR bank-conflict swizzling (`Swizzle_b128`) and software pipelining (depth=2).
   - `Tensorize`: Replaces nested matmul loops with hardware tensor-core instructions (`Tile_mma` lowering to Apple Silicon `simdgroup_matrix` $8\times 8\times 8$, CUDA WMMA, and HIP `rocWMMA`).
   - `Fuse_epilogue`: Folds elementwise activations, residuals, and bias additions directly into matrix store-backs.
4. **Universal Pool Allocator**: Manages device and constant memory pools across heterogeneous backends with untracked unified memory barriers.

## Comparison with `llmopt`

| Feature | `llmopt` | OCANNL `arrayjit` |
| :--- | :--- | :--- |
| **Ingest** | PyTorch Dynamo/FX Graph | OCaml eDSL (`%op`, `%cd`, einsum) |
| **Target Workload** | Ultra-low latency AOT LLM Serving | Training, Autodiff & General Tensor Execution |
| **Weight Quantization** | Bit-packed W4A16, GGUF K-quants (`Q4_K`, `Q5_0`) | Standard floats (`float16`, `bfloat16`, `float32`, `fp8`) |
| **Decode GEMV ($M=1$)** | 32-lane warp `simd_sum` + `metal::extract_bits` | Scalar/workgroup loop reduction |
| **Prefill GEMM ($M \ge 128$)**| Static heuristics / chunked loops | Full `simdgroup_matrix` + cooperative shared staging |
| **Target Backends** | Apple Silicon Metal (Monolithic string emitter) | Metal MSL, NVIDIA CUDA, AMD HIP, Native C |
| **Execution Overhead** | Zero (Prebaked Metal Indirect Command Buffers) | Dynamic host command queue dispatch |

## Key Insights for `llmopt`

1. **Multi-Backend Solution**: `arrayjit` eliminates the need for `llmopt` to hand-write thousands of lines of CUDA or HIP string templates. A single Assignment IR expression compiles to Metal, CUDA, and ROCm.
2. **Loop Optimization Engine**: `arrayjit` automates epilogue fusion, register allocation, and loop tiling, replacing custom hand-written fusion passes in `llmopt`.
3. **The Quantization Gap**: `arrayjit` does not natively feature sub-byte quantized block structures (GGUF superblocks) or warp-level `simd_sum` reductions out of the box, but its scalar and schedule IR can be extended to lower quantized math optimally.
