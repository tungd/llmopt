---
type: Architecture
title: 'Dynamo/FX Compiler with an OCaml Metal Serving Runtime'
description: 'A model-neutral Ahead-Of-Time (AOT) compilation and serving architecture: PyTorch Dynamo captures FX graphs, OCaml effect-driven passes lower and fuse megakernels against GGUF/UD weights, and the zero-JIT native runtime hosts continuous batching, radix prefix caching, and SIMD stochastic sampling on Apple Silicon.'
tags: [architecture, pytorch, fx, ocaml, metal, gemma, qwen, smollm, lfm, gguf, unsloth, continuous-batching, radix-cache, megakernels, aot]
status: stable
generated: { by: 'process:antigravity', at: '2026-08-31T10:15:00+07:00' }
sources:
  - id: pytorch-backend-contract
    resource: https://docs.pytorch.org/docs/2.9/torch.compiler_custom_backends.html
    title: PyTorch custom backend contract
  - id: local-python-backend
    resource: /python/llmopt_backend/__init__.py
    title: llmopt Dynamo/FX adapter
  - id: local-fx-compiler
    resource: /bin/fx_compile.ml
    title: Ahead-Of-Time compiler pipeline
  - id: local-ir
    resource: /lib/ir.ml
    title: High-level compiler intermediate representation
  - id: local-kernel-ir
    resource: /lib/kernel_ir.ml
    title: Kernel intermediate representation and stateful scan analysis
  - id: local-passes
    resource: /lib/passes.ml
    title: Modular compiler optimization pass pipeline
  - id: local-tactic-registry
    resource: /lib/kernel_abi.ml
    title: Shared shape-aware Linear tactic registry
  - id: local-target-hardware
    resource: /lib/target_hardware.ml
    title: Hardware microarchitectural discovery and analytical cost model
  - id: local-gguf
    resource: /lib/gguf.ml
    title: Native GGUF v2/v3 parser and tensor index
  - id: local-weight-archive
    resource: /lib/weight_archive.ml
    title: Versioned binary weight-archive parser and tensor index
  - id: local-serving-memory-plan
    resource: /lib/serving_memory_plan.ml
    title: Static interval-graph liveness memory planner
  - id: local-serving-package
    resource: /lib/serving_package.ml
    title: Versioned serving-package representation (ABI v25)
  - id: local-serving-schedule
    resource: /lib/serving_schedule.ml
    title: Binary typed command schedule (ABI v27)
  - id: local-metal
    resource: /lib/metal.ml
    title: Metal Shading Language (MSL) code generator and megakernels
  - id: local-metal-runtime
    resource: /lib/metal_runtime.ml
    title: Native OCaml Metal runtime and batched command dispatcher
  - id: local-metal-stubs
    resource: /native/ocaml_metal_stubs.m
    title: Objective-C Metal and ARM NEON SIMD runtime bindings
  - id: local-model-program
    resource: /lib/model_program.ml
    title: Root execution contract (ABI v2)
  - id: local-model-linker
    resource: /lib/model_program_linker.ml
    title: Model-neutral captured-package linker
  - id: local-radix-cache
    resource: /lib/radix_cache.ml
    title: OCaml compressed radix prefix cache
  - id: local-serving-cache
    resource: /lib/serving_cache.ml
    title: OCaml serving-cache coordinator
  - id: local-serving-queue
    resource: /lib/serving_queue.ml
    title: Continuous batching queue with age-weighted SRPT scheduler
  - id: local-sampling
    resource: /lib/sampling.ml
    title: Zero-allocation ARM NEON SIMD stochastic sampler
  - id: local-openai-protocol
    resource: /lib/openai_protocol.ml
    title: Typed external OpenAI compatibility edge
  - id: local-serving-server
    resource: /bin/serve.ml
    title: Persistent native OCaml HTTP/SSE streaming server
  - id: slice-31-receipt
    resource: /bench/results/compiler-generalization-slice-31-2026-08-30.json
    title: Slice 31 cross-model benchmark receipt
  - id: sglang-radix-cache
    resource: https://github.com/sgl-project/sglang/blob/d1af3c89233c475fc1bf11939d86787e6cddd58c/python/sglang/srt/mem_cache/radix_cache.py
    title: SGLang RadixCache reference revision
  - id: sglang-mamba-radix-cache
    resource: https://github.com/sgl-project/sglang/blob/d1af3c89233c475fc1bf11939d86787e6cddd58c/python/sglang/srt/mem_cache/mamba_radix_cache.py
    title: SGLang hybrid recurrent-cache reference revision
---

# 1. System Overview

`llmopt` is an Ahead-Of-Time (AOT) optimizing compiler and high-throughput zero-JIT serving runtime tailored for Apple Silicon unified memory architectures. The core engineering thesis is:

1. **Topology Authority in PyTorch/Dynamo**: Graph structure is acquired directly from standard PyTorch model code using `torch.compile(backend="llmopt")`. No architecture-specific forks or hardcoded model templates exist in the compiler pipeline.
2. **Model-Neutral AOT Solidification**: All operator fusions, memory allocations, tensor dequantizations, shape specializations, and execution schedules are computed offline. The serving engine requires zero JIT compilation, zero dynamic memory allocation on the hot path, and zero Python execution during serving.
3. **Fewest-Hops Whole-Block Megakernels**: Fusing multiple layers and operations into SRAM-resident megakernels eliminates intermediate round-trips to DRAM, matching or exceeding hand-tuned C++ engines like `llama.cpp` while retaining full compiler generalizability.
4. **Hardware-Aware Specialization**: Code generation and dispatch policies adapt to probed microarchitectural parameters (SIMD lanes, threadgroup SRAM capacity, cache line sizes, and roofline knees).

```
   ┌─────────────────────────────────────────────────────────────┐
   │             PyTorch / Dynamo Frontend (`torch.compile`)     │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Binary Transport (`graph.llmopt`, LLMOPTFX v2)
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │         OCaml AOT Compiler & Modular Pass Pipeline          │
   │  - SwiGLU / Gated Linear Fusions   - Sliced L2 Normalization│
   │  - Attention-Linked Rotary QK/RoPE - ShortConv-SiLU Fusion  │
   │  - Quantized Linear Residuals      - Recurrence/Scan Recovery│
   │  - DAG Co-Scheduling Antichains    - Shared Tactic Registry │
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Solidified Package (`package.llmopt` ABI v25, `schedule.llmopt` v27)
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             Root Model Program Contract (`model.llmopt`)    │
   │  - Prefill & Decode Schedules      - State/Cache Layouts    │
   │  - Model Profile Metadata          - Tokenizer & Chat Spec  │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                 Native Zero-JIT Serving Runtime             │
   │  - Continuous Batching (SRPT)      - Async Webs_iomux Server│
   │  - Radix KV / Recurrent Cache      - Prebaked Metal ICBs    │
   │  - Streaming NEON SIMD Sampler     - GGUF / UD Direct Mmap  │
   └─────────────────────────────────────────────────────────────┘
```

---

# 2. Subsystem Ownership & Layering

| Concern | Owning Module / Subsystem | Primary Implementation Files |
|---|---|---|
| **Graph Acquisition & Binary Transport** | Python FX Adapter | [`python/llmopt_backend/`](file:///Users/tung/Projects/std23/llmopt/python/llmopt_backend/__init__.py), [`lib/capture.ml`](file:///Users/tung/Projects/std23/llmopt/lib/capture.ml) |
| **Compiler IR & Topology Matching** | OCaml Compiler Core | [`lib/ir.ml`](file:///Users/tung/Projects/std23/llmopt/lib/ir.ml), [`lib/kernel_ir.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_ir.ml), [`lib/fusion_query.ml`](file:///Users/tung/Projects/std23/llmopt/lib/fusion_query.ml) |
| **Optimization Passes & Fusions** | Modular Pass Pipeline | [`lib/passes.ml`](file:///Users/tung/Projects/std23/llmopt/lib/passes.ml), [`lib/pass_fuse_*.ml`](file:///Users/tung/Projects/std23/llmopt/lib/pass_fuse_swiglu_ffn.ml) |
| **Linear Tactic Selection** | Shared Tactic Registry | [`lib/kernel_abi.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_abi.ml) |
| **Hardware Discovery & Cost Model** | Target Hardware | [`lib/target_hardware.ml`](file:///Users/tung/Projects/std23/llmopt/lib/target_hardware.ml), [`lib/kernel_cost_model.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_cost_model.ml) |
| **Code Generation (MSL)** | Metal Backend | [`lib/metal.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal.ml) |
| **Static Memory Liveness** | Memory Planner | [`lib/serving_memory_plan.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_memory_plan.ml) |
| **Packaging & Contracts** | Package / Schedule ABI | [`lib/serving_package.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_package.ml), [`lib/serving_schedule.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_schedule.ml) |
| **Root Execution Contract** | Model Program Linker | [`lib/model_program.ml`](file:///Users/tung/Projects/std23/llmopt/lib/model_program.ml), [`lib/model_program_linker.ml`](file:///Users/tung/Projects/std23/llmopt/lib/model_program_linker.ml) |
| **Weight Ingestion** | GGUF & Weight Archive | [`lib/gguf.ml`](file:///Users/tung/Projects/std23/llmopt/lib/gguf.ml), [`lib/weight_archive.ml`](file:///Users/tung/Projects/std23/llmopt/lib/weight_archive.ml) |
| **Metal Dispatch & FFI** | Native Metal Runtime | [`lib/metal_runtime.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_runtime.ml), [`native/ocaml_metal_stubs.m`](file:///Users/tung/Projects/std23/llmopt/native/ocaml_metal_stubs.m) |
| **Prefix & Recurrent Caching** | Serving Cache & Radix | [`lib/radix_cache.ml`](file:///Users/tung/Projects/std23/llmopt/lib/radix_cache.ml), [`lib/kv_cache.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kv_cache.ml), [`lib/serving_cache.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_cache.ml) |
| **Continuous Batching Queue** | Serving Queue (SRPT) | [`lib/serving_queue.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_queue.ml) |
| **Sampling & Generation** | NEON SIMD Sampler | [`lib/sampling.ml`](file:///Users/tung/Projects/std23/llmopt/lib/sampling.ml), [`lib/generation_core.ml`](file:///Users/tung/Projects/std23/llmopt/lib/generation_core.ml) |
| **HTTP / SSE Server** | Async Webs_iomux Server | [`bin/serve.ml`](file:///Users/tung/Projects/std23/llmopt/bin/serve.ml), [`lib/openai_protocol.ml`](file:///Users/tung/Projects/std23/llmopt/lib/openai_protocol.ml) |

---

# 3. Layer Architecture & Execution Flow

## 3.1. Frontend & Binary Transport (Python → OCaml)

The entry point is a PyTorch `torch.compile` custom backend.[^pytorch-backend-contract] When invoked, `llmopt_backend`:
1. Intercepts the FX `GraphModule` and example input tensors.
2. Extracts node targets, op kinds (`call_function`, `call_module`, `call_method`, `get_attr`, `placeholder`, `output`), and shape metadata (`val`, `tensor_meta`, or `example_value`).
3. Classifies static tensors (parameters and persistent buffers) versus dynamic request inputs.
4. Emits `graph.llmopt` using binary transport protocol `LLMOPTFX` v2. Arguments, numeric literals, shapes, dtypes, and tensor bindings are encoded with fixed-width tags and length-prefixed strings, avoiding all JSON parsing overhead and floating-point serialization inaccuracies.

## 3.2. Compiler IR, Effect Planning, & Modular Passes

The compiler pipeline ([`bin/fx_compile.ml`](file:///Users/tung/Projects/std23/llmopt/bin/fx_compile.ml)) reads `graph.llmopt` into high-level IR ([`lib/ir.ml`](file:///Users/tung/Projects/std23/llmopt/lib/ir.ml)) and executes an ordered pipeline of pure passes ([`lib/passes.ml`](file:///Users/tung/Projects/std23/llmopt/lib/passes.ml)):

1. **Topology-Selected Gated Linear & SwiGLU Fusions ([`Pass_fuse_swiglu_ffn`](file:///Users/tung/Projects/std23/llmopt/lib/pass_fuse_swiglu_ffn.ml))**:
   - Detects parallel gate and up projections sharing common inputs and feeding pointwise activation products (`silu(gate) * up`).
   - Fuses them into unified semantic `Gated_linear` operations across `Q4_K`, `Q5_K`, `Q5_0`, `IQ4_XS`, and `W4A16` layouts without materializing intermediate gate/up activation buffers in DRAM.
2. **Attention-Linked Rotary QK & RMSNorm-RoPE ([`Pass_fuse_rms_rope`](file:///Users/tung/Projects/std23/llmopt/lib/pass_fuse_rms_rope.ml))**:
   - Matches paired query/key normalization, rotary trigonometric embeddings, and transposition branches linked to a common Multi-Head/Grouped-Query Attention node.
   - Lowers them to unified layout-aware SIMD kernels (`llmopt_rms_rope_*`).
3. **RMSNorm & Residual Epilogue Fusions ([`Pass_fuse_rms_norm`](file:///Users/tung/Projects/std23/llmopt/lib/pass_fuse_rms_norm.ml))**:
   - Detects RMSNorm operations followed immediately by residual additions, folding them into `Rms_norm_add`.
   - Selects whole-threadgroup wide-row execution tactics for low-row, high-dimension tensors.
4. **Quantized Linear Residual Fusions ([`Pass_fuse_linear_bias`](file:///Users/tung/Projects/std23/llmopt/lib/pass_fuse_linear_bias.ml))**:
   - Replaces biasless block-quantized `Linear` operations and their sole-consumer same-shape Float16 `Add` with `Linear_add`.
   - Absorbs token-major value `transpose(1,2)` operations directly into the attention operand contract.
5. **Relational Short-Convolution Fusions ([`Pass_fuse_short_conv`](file:///Users/tung/Projects/std23/llmopt/lib/pass_fuse_short_conv.ml))**:
   - Fuses token-major transpose, depthwise 1D convolution, token state trim, SiLU activation, and output transposition into single `Short_conv_prefill` and `Short_conv_step` operations.
6. **Stateful Scan & Recurrence Recovery ([`Pass_fuse_gated_delta`](file:///Users/tung/Projects/std23/llmopt/lib/pass_fuse_gated_delta.ml), [`lib/kernel_ir.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_ir.ml))**:
   - Identifies unrolled recurrence loops and carried-state chains (e.g. Qwen's 63-step triangular recurrences and zero-state gated-delta expansions).
   - Reconstructs semantic `Kernel_ir.Scan` regions with explicit induction variables and carried tensors, lowering them directly to SIMD Metal kernels.
7. **DAG Concurrency & Antichain Co-Scheduling ([`Pass_co_schedule`](file:///Users/tung/Projects/std23/llmopt/lib/pass_co_schedule.ml))**:
   - Analyzes SSA dependency DAGs and computes ready antichains of independent operations (e.g., parallel Q/K/V projections or parallel layer components).
   - Groups independent kernels into concurrent execution stages bounded by `Ir.Op.Barrier_wait` and dispatched via `MTLDispatchTypeConcurrent`.

## 3.3. Tactic Registry & Microarchitectural Hardware Discovery

Rather than relying on brittle heuristic rules, kernel implementations are resolved through a unified, typed tactic registry ([`lib/kernel_abi.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_abi.ml)):
* **Shape & Layout Selection**: Automatically picks optimal SIMD tiles based on tensor dimensions $(M, N, K)$, storage format (`Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS`, `F32`), and input/output dtypes.
* **Multi-Token & Multi-Column Weight Reuse**:
  - For $M=2$ (two-token prompt or parallel decode steps), specializes kernels to decode each quantized superblock once while accumulating across both token rows simultaneously (`_m2n2` and paired-token tactics).
* **SIMD-Group Matrix Execution**: Dispatches broadcast-aware $8 \times 8$ SIMD-group matrix multiply tiles for aligned shapes with $16 \times 16$ threadgroup fallbacks.
* **Target Hardware Discovery ([`lib/target_hardware.ml`](file:///Users/tung/Projects/std23/llmopt/lib/target_hardware.ml))**: Probes GPU cores, memory bandwidth ($273\text{ GB/s}$ on M4 Pro), SIMD width (32 lanes), SRAM bank capacity, and calculates analytical prefill roofline knees:
  $$M_{\text{knee}} = \left\lceil \frac{W}{B} \times \frac{P}{2 \cdot \text{Params}} \right\rceil$$
  dynamically determining optimal chunk sizes ($M^* \in [64, 512]$) to maximize core saturation while adhering to SLA latency bounds.

## 3.4. Quantization Ingestion & Weight Architecture

`llmopt` provides first-class support for both native GGUF files and internal aligned binary archives:
* **Direct GGUF Mmap ([`lib/gguf.ml`](file:///Users/tung/Projects/std23/llmopt/lib/gguf.ml))**: Reads GGUF v2/v3 metadata, resolves tensor name aliases, and creates zero-copy 256-byte aligned `MTLBuffer` views directly from memory-mapped GGUF files without intermediate memory copies or conversions.
* **Quantization Coverage**:
  - **Block-32**: `Q4_0`, `Q8_0`, `Q5_0`
  - **Superblock-256 (K-Quants & Unsloth UD)**: `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS`
  - **Groupwise W4**: Group-64 scales with 2-complement low-nibble packing.
* **Specialized MSL Dequantizers ([`lib/metal.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal.ml))**: Vectorized compile-time dequantization unrolls 32-bit `uchar4`/`ushort2` loads directly into SIMD registers, avoiding runtime branch overhead.

## 3.5. Static Memory Liveness Planning & Runtime Dispatch

* **Liveness Memory Planner ([`lib/serving_memory_plan.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_memory_plan.ml))**:
  - Solves static interval-graph register allocation across all intermediate tensors.
  - Generates deterministic 256-byte aligned offset assignments into a single retained Metal workspace buffer.
  - Eliminates dynamic `[MTLDevice newBufferWithLength]` allocations during model execution.
* **Native Metal Runtime ([`lib/metal_runtime.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_runtime.ml))**:
  - Encodes entire schedule execution into consolidated command buffers.
  - Reuses cached `MTLComputePipelineState` objects.
  - Supports **Prebaked Indirect Command Buffers (ICB)** where pipeline states, arguments, and threadgroup launch dimensions are baked once offline, reducing CPU driver encoding time to $< 16\text{ µs}$ per decode step.

## 3.6. Serving Engine, Cache Hierarchy, & Protocols

* **Model Program ABI v2 ([`lib/model_program.ml`](file:///Users/tung/Projects/std23/llmopt/lib/model_program.ml))**:
  - Encapsulates compiled prefill schedules, decode schedules, state plans (KV cache and recurrent dimensions), tokenizer metadata, and chat templates into a self-contained root artifact (`model.llmopt`).
* **Compressed Radix Cache ([`lib/radix_cache.ml`](file:///Users/tung/Projects/std23/llmopt/lib/radix_cache.ml), [`lib/serving_cache.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_cache.ml))**:
  - Implements tree-structured prefix caching with compressed edges, LRU leaf eviction, and namespace isolation.
  - Manages physical token pools (grouped-Q8 or FP16) and recurrent checkpoint states (e.g. ShortConv buffers), reusing prefixes across multi-turn sessions without recomputation.[^sglang-radix-cache][^sglang-mamba-radix-cache]
* **Continuous Batching Queue ([`lib/serving_queue.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_queue.ml))**:
  - Implements an age-weighted Shortest Remaining Processing Time (**SRPT**) scheduling policy.
  - Prioritizes active decode steps over monolithic prefills to prevent head-of-line blocking and minimize Time-to-First-Token (TTFT) and Time-Per-Output-Token (TPOT) variance under high load.
* **Streaming NEON SIMD Sampler ([`lib/sampling.ml`](file:///Users/tung/Projects/std23/llmopt/lib/sampling.ml), [`native/ocaml_metal_stubs.m`](file:///Users/tung/Projects/std23/llmopt/native/ocaml_metal_stubs.m))**:
  - Zero-heap-allocation, single-pass streaming min-heap sampler implemented in ARM NEON assembly.
  - Supports dynamic `temperature`, `top_k`, `top_p`, `min_p`, and seeded PRNG with $< 20\text{ µs}$ per-token execution time without vocabulary sorting.
* **Async HTTP/SSE Server ([`bin/serve.ml`](file:///Users/tung/Projects/std23/llmopt/bin/serve.ml))**:
  - Embeds the non-blocking `Webs_iomux` event loop on top of `Iomux.Poll`, serving OpenAI-compatible `/v1/chat/completions` streams with low latency.

---

# 4. Probe Models & Empirical Performance

The compiler is validated against a multi-architecture probe matrix without embedding model-specific selectors into passes or runtime dispatch:

| Model | Architecture | Quantization | LLMOpt Latency (M4 Pro) | `llama.cpp` Latency | Relative Ratio | Empirical Status |
|---|---|---|---|---|---|---|
| **SmolLM2-135M-Instruct** | Standard Transformer (GQA, SwiGLU, RMSNorm, RoPE) | `Q4_K_M` | **3.2669 ms** | 3.2917 ms | **0.9925x** | Parity achieved; 60 Linear-add & 30 SwiGLU fusions. |
| **Qwen3.5-0.8B** | Hybrid / Stateful Scan (18 recurrent layers, gated-delta, ShortConv) | `UD-Q4_K_XL` | **7.9665 ms** | 7.9538 ms | **1.0016x** | Parity achieved; 18 stateful scan regions recovered & 48 Linear-add fusions. |
| **Gemma-4-E2B-it** | Deep Transformer (Wide RMSNorm, Gated MLP, h256 Attention) | `UD-Q4_K_XL` | **18.1395 ms** | 17.7819 ms | **1.0201x** | Parity achieved; wide-row RMSNorm, rotary QK, and attention layout absorption. |
| **LFM2.5-350M** | Hybrid Conv / Attention | `W4A16 / KVQ8` | **18.2 ms** (TTFT Turn 1) | 19.1 ms | **0.9529x** | Full serving probe with true suffix prefill and prefix retention. |

All benchmark runs adhere strictly to the target performance envelope ($0.90\text{x} \le \frac{\text{LLMOpt}}{\text{llama.cpp}} \le 1.10\text{x}$) while guaranteeing bit-exact deterministic execution and preserving reference argmax token predictions.[^slice-31-receipt]

---

# 5. File & ABI Version Registry

| Contract / File Format | Identifier / Magic | Current Version | Authority Module |
|---|---|---|---|
| **Binary Transport** | `LLMOPTFX` | Version 2 | [`lib/capture.ml`](file:///Users/tung/Projects/std23/llmopt/lib/capture.ml), [`python/llmopt_backend/fx_graph.py`](file:///Users/tung/Projects/std23/llmopt/python/llmopt_backend/fx_graph.py) |
| **Serving Package** | `LLMOPTPKG` | ABI Version 25 | [`lib/serving_package.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_package.ml) |
| **Serving Schedule** | `LLMOPTSCHED` | Version 27 | [`lib/serving_schedule.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_schedule.ml) |
| **Model Program** | `LLMOPTMP` | ABI Version 2 | [`lib/model_program.ml`](file:///Users/tung/Projects/std23/llmopt/lib/model_program.ml) |
| **Binary Weight Archive** | `LLMOPTWA` | Version 1 | [`lib/weight_archive.ml`](file:///Users/tung/Projects/std23/llmopt/lib/weight_archive.ml) |
| **Native GGUF Ingestion** | `GGUF` | v2 / v3 | [`lib/gguf.ml`](file:///Users/tung/Projects/std23/llmopt/lib/gguf.ml) |

---

[^pytorch-backend-contract]: PyTorch custom backend documentation.
[^sglang-radix-cache]: SGLang `RadixCache`, pinned to revision `d1af3c89233c475fc1bf11939d86787e6cddd58c`.
[^sglang-mamba-radix-cache]: SGLang `MambaRadixCache`, pinned to the same revision.
[^slice-31-receipt]: Full-model multi-campaign receipt in `bench/results/compiler-generalization-slice-31-2026-08-30.json`.
