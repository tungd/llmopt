---
type: Decision
title: 'Adopt OCANNL arrayjit loop-nest lowering for quantized LLM kernels and multi-backend portability'
description: 'Integrate OCANNL arrayjit as an intermediate loop lowering tier between llmopt macro-fusion and device shader compilation, extending arrayjit with sub-byte dequantization and warp-level reduction transforms to automate Metal/CUDA/HIP codegen.'
tags: [decision, compiler, lowering, ocannl, arrayjit, quantization, metal, rocm, cuda, icb]
status: approved
generated: { by: human:tung, at: '2026-08-31T13:58:30+07:00' }
sources:
  - id: ocannl-prior-art
    resource: /.okf/prior-art/ocannl-arrayjit.md
    title: OCANNL arrayjit prior art analysis
  - id: gguf-ud-decision
    resource: /.okf/decisions/gguf-unsloth-dynamic-quantization.md
    title: GGUF and Unsloth Dynamic Quantization Support
  - id: backend-boundary
    resource: /.okf/decisions/backend-boundary.md
    title: Metal backend boundary
  - id: direction-md
    resource: /direction.md
    title: Project direction and ROCm multi-backend analysis
---

# Decision

`llmopt` adopts **OCANNL's `arrayjit` loop-nest compilation engine** as the intermediate lowering tier between high-level macro-fusion passes and backend shader generation. `llmopt` retains ownership of FX graph capture, GGUF parameter binding, Model Program linking, static memory planning, paged KV cache allocation, and prebaked Indirect Command Buffer (ICB) execution.

```
                      PyTorch FX Graph Capture
                                 │
                                 ▼
                     llmopt Macro-Fusion Passes
             (RMSNorm-RoPE, SwiGLU, ShortConv-GatedDelta)
                                 │
                                 ▼
              OCANNL Assignment IR (%cd / Assignments.t)
              [Math, GGUF Bit-Unpacking & Scalar Idioms]
                                 │
                                 ▼
                       OCANNL arrayjit
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
    Loop Tiling & CSE      Warp Reduction         Tile_mma / Stage
    (virtual_llc, hoist)   (simd_sum, extract)    (simdgroup_matrix)
          │                      │                      │
          ▼                      ▼                      ▼
      Metal MSL             NVIDIA CUDA              AMD HIP
          │
          ▼
   Compiled .metallib / Pipeline State Objects
          │
          ▼
    llmopt Prebaked Indirect Command Buffers (ICBs)
          │
          ▼
    llmopt Zero-JIT Native Serving Runtime
```

## Architectural Boundaries

1. **`llmopt` High-Level Authority**:
   - Captures Dynamo/FX graphs and binds GGUF quantized parameters (`UD-Q4_K_XL`, `Q8_0`, `W4A16`).
   - Executes macro-fusion across multi-layer topologies (e.g. `RMSNorm + QKV`, `SwiGLU + Down`, `Scan + Recurrence`).
   - Prepares the static memory plan (`serving_memory_plan.ml`) and encodes dispatches into zero-overhead Metal ICBs.
2. **`arrayjit` Lowering Authority**:
   - Converts macro-fused operations into Assignment IR (`Assignments.t` / `%cd`).
   - Performs backend-independent scalar optimizations (`virtual_llc`, `simplify_llc`, `hoist_cross_statement_cse`).
   - Applies schedule transforms (`Split`, `Stage`, `Tensorize`, `Fuse_epilogue`) and generates target shader source code for Metal, CUDA, and AMD HIP.
3. **Quantized Loop Lowering Extensions**:
   - **Sub-Byte Idiom Lowering**: Express block dequantization arithmetic (GGUF `Q4_K`, `Q5_0`, `W4A16`) in Assignment IR using `Byte_prec` weight tensors, `Half_prec` scale/min buffers, and bitwise operators (`land`, `lsr`, `Sub`).
   - **Hardware Bitfield Extraction**: Pattern-match bitfield extractions in `metal_backend.ml` to emit native `metal::extract_bits` intrinsics.
   - **Warp-Level Cooperative Reductions**: Introduce a `Warp_reduce` schedule transform that maps GEMV reduction loops over 32 SIMDgroup lanes and emits `simd_sum(acc)`.
   - **Compute-Bound Prefill GEMM**: Lower unquantized FP16/BF16 prefill GEMM via `Tile_mma` (`simdgroup_matrix` $8\times 8\times 8$ on Apple Silicon, WMMA on CUDA, `rocWMMA` on HIP) with cooperative staging.

# Rationale

1. **Elimination of the 400KB `metal.ml` Debt**: Replaces hand-stitched MSL string concatenations with formal OCaml data structures and verified loop optimization passes.
2. **Instant Multi-Backend Portability**: A single Assignment IR expression compiles to Metal MSL, NVIDIA CUDA PTX, and AMD HIP C++ without duplicating per-target template emitters.
3. **Preservation of Zero-JIT Serving Performance**: Emitted shader functions are compiled to `.metallib` (or device binaries) at build time and linked into `llmopt`'s prebaked ICBs, preserving single-digit microsecond dispatch latency and length-invariant paged attention.
