---
type: Decision
title: 'Constructive Primitive Assignment Lowering and GPU Queue-Aware Megakernel Compilation'
description: 'Unify high-level graph optimization and low-level loop generation by immediately lowering FX graphs to primitive assignment micro-ops, partitioning execution via GPU dependency and queue analysis, and generating multi-backend megakernels with first-class block-quantized schemas.'
tags: [decision, compiler, lowering, megakernel, arrayjit, dag-analysis, queue-model, block-quantization, metal, cuda, rocm]
status: approved
generated: { by: human:tung, at: '2026-09-01T22:56:00+07:00' }
sources:
  - id: ocannl-prior-art
    resource: /.okf/prior-art/ocannl-arrayjit.md
    title: OCANNL arrayjit loop-nest compiler analysis
  - id: fewest-hops
    resource: /.okf/decisions/fewest-hops-megakernel-compiler.md
    title: Fewest-Hops Whole-Block Megakernel Compiler
  - id: constructive-lowering
    resource: /.okf/decisions/constructive-lowering-symbolic-operator-dsl.md
    title: Constructive Lowering via Algebraic Inlining
  - id: gguf-quant
    resource: /.okf/decisions/gguf-unsloth-dynamic-quantization.md
    title: GGUF and Unsloth Dynamic Quantization Support
  - id: dag-co-scheduling
    resource: /.okf/decisions/dag-co-scheduling-optimizer-pass.md
    title: DAG Concurrency Analysis and Co-Scheduling
---

# Decision

`llmopt` adopts **Constructive Primitive Assignment Lowering with GPU Queue & Dependency Analysis**, unifying high-level graph optimization and low-level loop generation into a single mathematical compiler architecture.

This decision supersedes the dual-tier model of handwritten AST graph pattern matchers (`pass_fuse_*.ml`) and monolithic handwritten C++ template strings (`metal.ml`). 

```
                       PyTorch FX Graph Capture
                                   │
                (Immediate Operator Erasure / Decomposition)
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Primitive Assignment IR (Mathematical Micro-Ops)                         │
│    • Pointwise Math: +, -, *, /, exp, sqrt, sigmoid, silu                   │
│    • Reduction Loops: sum_k, mean_k, argmax_k                               │
│    • First-Class Block Quantization: dequant(W_physical, i, k)              │
│    • Dynamic Table Indirection: slots[token_idx]                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. GPU Queue & Dependency Partitioner (Topology-Driven Clustering)          │
│    • Affine 1-to-1 Dependencies     ──► INLINE INTO REGISTERS (virtual_llc) │
│    • Shared-Row Cooperative Ops     ──► STAGE IN SRAM (threadgroup memory)  │
│    • Cross-Threadgroup Reductions   ──► CUT CLUSTER & INSERT QUEUE BARRIER  │
│    • Dynamic Pointer Indirections   ──► CUT CLUSTER & EMIT DISPATCH BOUNDARY│
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                     Maximal Megakernel Clusters
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Block-Aware Loop & Schedule Generator (arrayjit-style)                   │
│    • Unrolls loops and inlines mathematical trees into register accumulators│
│    • Expands Block Schemas (Q4_K, Q5_0, Q8_0, W4A16) to SIMD vector loads   │
│    • Emits native bitfield intrinsics (extract_bits, __byte_perm)           │
│    • Generates multi-backend source code (Metal MSL / CUDA PTX / AMD HIP)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Zero-JIT AOT Serving Plane (llmopt Model Program Runtime)                │
│    • Compiles clusters AOT into immutable binaries (.metallib / cubin)      │
│    • Encodes dispatches and stage barriers into Prebaked Command Buffers    │
│    • Executes continuous batching (SRPT queue) and Radix prefix caching     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# Context & Post-Mortem of Previous Lowering Attempts

Previous attempts to integrate classical loop-nest compilers (such as raw `arrayjit` or naive loop inlining) into `llmopt` encountered critical friction:

1. **The Sub-Byte Modeling Gap**:
   Classical array programming compilers assume homogeneous, uniformly-strided primitive scalars ($\text{Address} = \text{Base} + i \cdot \text{stride}$). GGUF and Unsloth Dynamic quantizations use **heterogeneous compression superblocks** (`block_q4_k`, `block_q5_0`, `block_q6_k`) with non-uniform sub-block scales, min offsets, and packed 4-bit/5-bit nibbles.
   Without a first-class block-quantized abstraction, raw array compilers emit scalar bit-shifting loops, destroying SIMD vectorization and missing hardware bitfield instructions (`metal::extract_bits`).

2. **The 1D Naive Unrolling Mismatch**:
   Naive loop inlining treats computation as a single linear CPU-like instruction stream. On GPUs, unrolling an entire Transformer layer into one giant loop causes severe **register spilling to DRAM** and fails to account for **inter-threadgroup synchronization boundaries** (e.g. RMSNorm global reductions feeding parallel QKV GEMVs).

3. **Handwritten Template Debt (`metal.ml`)**:
   To hit strict empirical latency targets on tight milestones, `llmopt` authored hand-tuned C++ string templates in `metal.ml` (reaching ~10,000 lines) and bespoke pattern matchers in `pass_fuse_*.ml` (reaching ~3,000 lines). This created an exponential maintenance burden across models, quant formats, and target backends.

---

# Architecture & Formal Principles

### 1. Complete Operator Erasure
The compiler eliminates named high-level operators (`Ir.Op.Gated_linear`, `Ir.Op.Rms_rope`, `Ir.Op.Quant_qkv_linear`). All incoming operations decompose immediately into four fundamental micro-ops:

1. **Pointwise Math**: $y_i = f(x_i)$ (evaluated purely in registers).
2. **Reduction**: $y = \sum_k x_k$ (evaluated via 32-lane warp-level register shuffles `simd_sum`).
3. **Block-Quantized Accessor**: $W_{i,k} = \text{dequant}(W_{\text{physical}}[i, k/B], k \pmod B)$.
4. **Indirection**: $y_i = x[\text{table}[i]]$ (evaluated via gather/scatter addressing).

### 2. Dependency Edge to Hardware Hierarchy Mapping
The partitioner analyzes mathematical index expressions and maps dataflow edges directly to the GPU execution hierarchy:

| Dataflow Dependency Edge | Hardware Scope | Compiler Partitioning Action |
| :--- | :--- | :--- |
| **Injective Pointwise ($i \rightarrow i$)** | Single SIMD Lane / Register | **Inline into Register (`virtual_llc`)**: Zero DRAM or SRAM memory allocated. |
| **Shared-Row Cooperative ($i \rightarrow \text{col}$)** | Threadgroup SRAM | **Stage in SRAM (`Stage`)**: Cooperative load + `threadgroup_barrier`. |
| **Global Reduction ($\sum_k \rightarrow \text{broadcast}$)** | Device / Cross-Threadgroup | **Cut Cluster & Insert Queue Barrier**: Emits a stage barrier in the Prebaked ICB. |
| **Dynamic Pointer Indirection** | Dynamic Page Pool | **Cut Cluster & Emit Dispatch Boundary**: Dispatches specialized paged attention. |

### 3. First-Class Block-Quantized Schemas
Quantized weight tensors are modeled with dual-space semantics:
* **Logical Space**: Shape $[N, K]$, Dtype `Float16`.
* **Physical Space**: Buffer of structured superblocks of size $B$ (e.g., $B=256$ for $Q4\_K$, $B=32$ for $Q5\_0$).

Each quantization format is defined as a declarative, reusable OCaml schema:

```ocaml
module Block_schema = struct
  type t = {
    name : string;
    block_size : int;
    bytes_per_block : int;
    unpack_simd :
      lane:Expr.t ->
      block_ptr:Expr.t ->
      scale_ptr:Expr.t ->
      (Expr.t * Expr.t); (* Returns (q0, q1) dequantized SIMD registers *)
  }
end
```

The loop lowering engine expands `unpack_simd` directly into native target hardware intrinsics:
* **Metal MSL**: `metal::extract_bits`, `half2` vector loads, `simd_sum`.
* **NVIDIA CUDA**: `__byte_perm`, `__shfl_down_sync`, `half2`.
* **AMD ROCm/HIP**: `__builtin_amdgcn_wave_reduce_add`, `rocWMMA`.

---

# Phased Implementation Plan

### Phase 1: Micro-Op Assignment IR & Block-Quantized Schema Engine
* Implement the core mathematical AST (`Micro_op.t`) and the declarative quantization schemas (`Q4_K`, `Q5_0`, `Q8_0`, `W4A16`, `Q6_K`).
* Implement the vector loop skeleton generators for Single-Token Decode GEMV ($M=1$), Prefill GEMM ($M \ge 32$), and Epilogue Fusions.

### Phase 2: Topology-Driven GPU Queue Partitioner
* Build the DAG partitioner that computes antichains, working-set register estimates, and synchronization cuts.
* Verify that SwiGLU, RMSNorm-RoPE, and Linear-Residual blocks are clustered automatically without hand-coded pattern matchers.

### Phase 3: Multi-Backend Lowering & ICB Linking
* Connect the generated loop nests to the Ahead-of-Time compiler, emitting `.metallib` (Metal), PTX/cubin (CUDA), and HIP binaries.
* Encode the resulting dispatches and stage barriers into `llmopt`'s Prebaked Indirect Command Buffers.

### Phase 4: Progressive Deprecation of Legacy Artifacts
* Progressively route model graphs through the constructive lowering engine.
* Verify 100% bit-exact parity and sustained throughput on SmolLM2, Qwen3.5, and Gemma 4 against the llama.cpp baseline.
* Decommission `pass_fuse_*.ml` (saving ~3,000 lines) and `metal.ml` (saving ~10,000 lines).

---

# Verification & Success Criteria

1. **Zero Pattern Matcher Code**: Elimination of bespoke AST pattern matchers in favor of pure topological dependency analysis.
2. **Kernel Codebase Reduction**: Reduction of compiler shader emission code by $> 90\%$ (from ~10,000 lines of static strings to $< 500$ lines of modular schemas and loop skeletons).
3. **Multi-Backend Portability**: A single Model Program compiles and executes on Apple Silicon Metal, NVIDIA CUDA, and AMD ROCm with zero hand-written per-target kernel files.
4. **Performance Parity**: Sustained decoding throughput and TTFT remain within the established $[0.90\text{x}, 1.10\text{x}]$ performance envelope against `llama.cpp`.
