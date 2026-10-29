---
type: Architecture
title: 'Dynamo/FX Compiler with an OCaml Metal Serving Runtime'
description: 'A model-neutral Ahead-Of-Time (AOT) compilation and serving architecture: PyTorch Dynamo captures FX graphs, OCaml effect-driven passes lower and fuse megakernels against GGUF/UD weights, and the zero-JIT native runtime hosts continuous batching, radix prefix caching, and SIMD stochastic sampling on Apple Silicon.'
tags: [architecture, pytorch, fx, ocaml, metal, gemma, qwen, smollm, lfm, gguf, unsloth, continuous-batching, radix-cache, megakernels, aot]
status: stable
generated: { by: codex/gpt-6, at: '2026-10-29T18:39:51Z' }
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
    title: Kernel intermediate representation facade and region routing
  - id: local-kernel-region
    resource: /lib/kernel_region.ml
    title: Kernel slot, primitive, effect, and region records
  - id: local-kernel-scan
    resource: /lib/kernel_scan.ml
    title: Kernel stateful scan and recurrence recovery
  - id: local-kernel-fused-tactic
    resource: /lib/kernel_fused_tactic.ml
    title: Fused tactic selection and linear epilogue folding
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
  - id: local-python-weight-archive
    resource: /python/llmopt_backend/tensor_archive.py
    title: Shared Python binary weight archive writer and inventory reader
  - id: local-linear-writeback
    resource: /lib/metal_q4_0_epilogue_codegen.ml
    title: Direct construction of linear kernel headers and writebacks
  - id: local-weight-archive
    resource: /lib/weight_archive.ml
    title: Versioned binary weight-archive parser and tensor index
  - id: local-serving-memory-plan
    resource: /lib/serving_memory_plan.ml
    title: Static interval-graph liveness memory planner
  - id: local-serving-package
    resource: /lib/serving_package.ml
    title: Versioned serving-package representation (ABI v27)
  - id: local-serving-schedule
    resource: /lib/serving_schedule.ml
    title: Binary typed command schedule facade (format v29; additive diagnostic opaque opcode)
  - id: local-serving-schedule-types
    resource: /lib/serving_schedule_types.ml
    title: Schedule command, stage, and tensor input types
  - id: local-serving-schedule-validation
    resource: /lib/serving_schedule_validation.ml
    title: Schedule and fusion command validation rules
  - id: local-serving-schedule-core
    resource: /lib/serving_schedule_core.ml
    title: Command schedule creation, validation, and barrier pruning
  - id: local-serving-schedule-codec
    resource: /lib/serving_schedule_codec.ml
    title: Binary schedule serialization and deserialization
  - id: local-serving-schedule-sequence
    resource: /lib/serving_schedule_sequence.ml
    title: Schedule sequence construction for prefill and decode
  - id: local-metal
    resource: /lib/metal.ml
    title: Metal Shading Language (MSL) code generator facade
  - id: local-metal-source-common
    resource: /lib/metal_source_common.ml
    title: Metal entry constructors and shared header definitions
  - id: local-metal-selection
    resource: /lib/metal_selection.ml
    title: Metal tactic queries, capability tests, and fusion region selection
  - id: local-metal-program
    resource: /lib/metal_program.ml
    title: Compiled Metal program structure and representations
  - id: local-metal-primary-codegen
    resource: /lib/metal_primary_codegen.ml
    title: Primary kernel representation and lower_primary
  - id: local-metal-parametric-codegen
    resource: /lib/metal_parametric_codegen.ml
    title: Parametric kernel emitters (W4A16, Q4_K, SIMDgroup GEMM)
  - id: local-metal-block-codegen
    resource: /lib/metal_block_codegen.ml
    title: Schema-driven parameterized block-quant code generation
  - id: local-metal-runtime
    resource: /lib/metal_runtime.ml
    title: Native OCaml Metal runtime facade
  - id: local-metal-execution-state
    resource: /lib/metal_execution_state.ml
    title: Runtime execution context and dispatch state
  - id: local-metal-dispatch-common
    resource: /lib/metal_dispatch_common.ml
    title: Shared dispatch helpers, buffer encoding, and grid sizing
  - id: local-metal-dispatch-assignment
    resource: /lib/metal_dispatch_assignment.ml
    title: Dispatch for assignment-based fusion kernels
  - id: local-metal-schedule-executor
    resource: /lib/metal_schedule_executor.ml
    title: Metal schedule execution traversal and workspace preparation
  - id: local-metal-stubs
    resource: /native/ocaml_metal_stubs.m
    title: Objective-C Metal and ARM NEON SIMD runtime bindings
  - id: local-model-program
    resource: /lib/model_program.ml
    title: Root execution contract (ABI v7)
  - id: local-model-linker
    resource: /lib/model_program_linker.ml
    title: Model-neutral captured-package linker
  - id: local-serving-engine
    resource: /lib/serving_engine.ml
    title: Continuous batching serving engine lifecycle owner
  - id: local-serving-engine-contract
    resource: /lib/serving_engine_contract.ml
    title: Serving engine contract, bindings, and package validation
  - id: local-serving-rope-table
    resource: /lib/serving_rope_table.ml
    title: Precomputed RoPE frequency tables and slot slicing
  - id: local-serving-speculative
    resource: /lib/serving_speculative.ml
    title: Pure speculative acceptance, MTP transactions, Medusa, and ngram proposals
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
  - id: canonical-tile32-affine
    resource: /decisions/canonical-tile32-affine-and-hadamard-attention-rotation.md
    title: Canonical Tile32_Affine Weight & KV Format and In-Register Hadamard Attention Rotation
  - id: sglang-radix-cache
    resource: https://github.com/sgl-project/sglang/blob/d1af3c89233c475fc1bf11939d86787e6cddd58c/python/sglang/srt/mem_cache/radix_cache.py
    title: SGLang RadixCache reference revision
  - id: sglang-mamba-radix-cache
    resource: https://github.com/sgl-project/sglang/blob/d1af3c89233c475fc1bf11939d86787e6cddd58c/python/sglang/srt/mem_cache/mamba_radix_cache.py
    title: SGLang hybrid recurrent-cache reference revision
---

The current weight-format direction is [EXL3](decisions/exl3-canonical-runtime.md).
The EXL3 integration adds typed archive binding, generated ordinary and indexed
expert contractions, and a bounded GGUF converter. Existing affine packages
retain their execution path. See the [Mellum integration evidence](/experiments/exp-0180-exl3-mellum-integration-2026-10-30.md).


# 1. System Overview

`llmopt` is an Ahead-Of-Time (AOT) optimizing compiler and high-throughput zero-JIT serving runtime tailored for Apple Silicon unified memory architectures. The core engineering thesis is:

1. **Topology Authority in PyTorch/Dynamo**: Graph structure is acquired directly from standard PyTorch model code using `torch.compile(backend="llmopt")`. No architecture-specific forks or hardcoded model templates exist in the compiler pipeline.
2. **Model-Neutral AOT Solidification**: All operator fusions, memory allocations, tensor dequantizations, shape specializations, and execution schedules are computed offline. The serving engine requires zero JIT compilation, zero dynamic memory allocation on the hot path, and zero Python execution during serving.
3. **Fewest-Hops Whole-Block Megakernels**: Fusing multiple layers and operations into SRAM-resident megakernels eliminates intermediate round-trips to DRAM and targets a narrow performance envelope against hand-tuned runtimes such as `llama.cpp`; scope and protocol remain explicit per measurement.
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
                                  │ Solidified Package (`package.llmopt` ABI v27, `schedule.llmopt` format v29)
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
   │  - Radix KV / Recurrent Cache      - Compiled Metal Commands│
   │  - Streaming NEON SIMD Sampler     - GGUF / UD Direct Mmap  │
   └─────────────────────────────────────────────────────────────┘
```

## 1.1 Current Product Boundary and Backend Deferral

The current product boundary is the model-neutral FX → OCaml AOT compiler and
`model.llmopt` execution contract, with Metal as the only full-model execution
backend. Metal is the current product path; Vulkan/MoltenVK is an experimental
portability probe, not a second supported serving backend. ROCm/HIP is
intentionally deferred rather than a parallel implementation target.

The compiler/backend seam remains useful: primitive semantics, fusion
strategies, and backend tactics are separate concerns, and the target profile
is explicit. That seam is not evidence of backend portability. Before a
ROCm/HIP backend is reconsidered, the current Metal path must have a stable
Model Program/package/schedule contract, a repeatable model-adoption matrix,
actionable unsupported-FX diagnostics, and matched device evidence for the
compiler, runtime, and serving boundaries. A backend port must reproduce those
contracts rather than introduce a model-specific runtime or schedule family.

# 2. Subsystem Ownership & Layering

| Concern | Owning Module / Subsystem | Primary Implementation Files |
|---|---|---|
| **Graph Acquisition & Binary Transport** | Python FX Adapter | [`python/llmopt_backend/`](file:///Users/tung/Projects/std23/llmopt/python/llmopt_backend/__init__.py), [`lib/capture.ml`](file:///Users/tung/Projects/std23/llmopt/lib/capture.ml) |
| **Compiler IR & Topology Matching** | OCaml Compiler Core | [`lib/ir.ml`](file:///Users/tung/Projects/std23/llmopt/lib/ir.ml), [`lib/kernel_ir.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_ir.ml), [`lib/kernel_region.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_region.ml), [`lib/kernel_scan.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_scan.ml), [`lib/kernel_fused_tactic.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_fused_tactic.ml), [`lib/fusion_query.ml`](file:///Users/tung/Projects/std23/llmopt/lib/fusion_query.ml) |
| **Optimization Passes & Fusions** | Modular Pass Pipeline | [`lib/passes.ml`](file:///Users/tung/Projects/std23/llmopt/lib/passes.ml), [`lib/pass_fuse_*.ml`](file:///Users/tung/Projects/std23/llmopt/lib/pass_fuse_swiglu_ffn.ml), [`lib/pass_co_schedule.ml`](file:///Users/tung/Projects/std23/llmopt/lib/pass_co_schedule.ml), [`lib/dag_analysis.ml`](file:///Users/tung/Projects/std23/llmopt/lib/dag_analysis.ml) |
| **Linear Tactic Selection** | Shared Tactic Registry | [`lib/kernel_abi.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_abi.ml) |
| **Hardware Discovery & Cost Model** | Target Hardware | [`lib/target_hardware.ml`](file:///Users/tung/Projects/std23/llmopt/lib/target_hardware.ml), [`lib/kernel_cost_model.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_cost_model.ml) |
| **Code Generation (MSL)** | Metal Backend | [`lib/metal.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal.ml), [`lib/metal_program.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_program.ml), [`lib/metal_selection.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_selection.ml), [`lib/metal_primary_codegen.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_primary_codegen.ml), [`lib/metal_parametric_codegen.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_parametric_codegen.ml), [`lib/metal_block_codegen.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_block_codegen.ml), [`lib/metal_*_source.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_source_common.ml) |
| **Static Memory Liveness** | Memory Planner | [`lib/serving_memory_plan.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_memory_plan.ml) |
| **Packaging & Contracts** | Package / Schedule ABI | [`lib/serving_package.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_package.ml), [`lib/serving_schedule.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_schedule.ml), [`lib/serving_schedule_core.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_schedule_core.ml), [`lib/serving_schedule_codec.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_schedule_codec.ml), [`lib/serving_schedule_sequence.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_schedule_sequence.ml) |
| **Root Execution Contract** | Model Program Linker | [`lib/model_program.ml`](file:///Users/tung/Projects/std23/llmopt/lib/model_program.ml), [`lib/model_program_linker.ml`](file:///Users/tung/Projects/std23/llmopt/lib/model_program_linker.ml) |
| **Weight Ingestion** | GGUF & Weight Archive | [`lib/gguf.ml`](file:///Users/tung/Projects/std23/llmopt/lib/gguf.ml), [`lib/weight_archive.ml`](file:///Users/tung/Projects/std23/llmopt/lib/weight_archive.ml) |
| **Metal Dispatch & FFI** | Native Metal Runtime | [`lib/metal_runtime.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_runtime.ml), [`lib/metal_schedule_executor.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_schedule_executor.ml), [`lib/metal_execution_state.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_execution_state.ml), [`lib/metal_dispatch_*.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_dispatch_common.ml), [`native/ocaml_metal_stubs.m`](file:///Users/tung/Projects/std23/llmopt/native/ocaml_metal_stubs.m) |
| **Serving Engine & Speculative Execution** | Serving Engine | [`lib/serving_engine.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_engine.ml), [`lib/serving_engine_contract.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_engine_contract.ml), [`lib/serving_rope_table.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_rope_table.ml), [`lib/serving_speculative.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_speculative.ml) |
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

The compiler pipeline ([`bin/fx_compile.ml`](file:///Users/tung/Projects/std23/llmopt/bin/fx_compile.ml)) reads `graph.llmopt` into high-level IR ([`lib/ir.ml`](file:///Users/tung/Projects/std23/llmopt/lib/ir.ml)) and executes an ordered pipeline of pure passes managed by [`lib/passes.ml`](file:///Users/tung/Projects/std23/llmopt/lib/passes.ml):

1. **Constructive assignment regions ([`lib/constructive_lowering.ml`](file:///Users/tung/Projects/std23/llmopt/lib/constructive_lowering.ml), [`lib/kernel_ir.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_ir.ml), [`lib/kernel_region.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_region.ml))**:
   - `Kernel_region` defines slot, primitive, effect, resource, and region models, constructing explicit bindings and validating region metadata.
   - Normalizes Linear, pointwise, reduction, movement, ShortConv, padding, batched-matmul, slice-update, and recurrence requests into one primitive assignment vocabulary.
   - Uses dependency, fan-out, effect, and resource facts to form legal components, then extracts exact executable slices for gated Linear, Linear-add, RMSNorm, RMSNorm-RoPE, ShortConv-SiLU, and gated-delta tactics via `Kernel_fused_tactic` ([`lib/kernel_fused_tactic.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_fused_tactic.ml)).
2. **Typed layout provenance ([`lib/pass_fuse_linear_bias.ml`](file:///Users/tung/Projects/std23/llmopt/lib/pass_fuse_linear_bias.ml))**:
   - Composes logical source axes, broadcast axes, and flattened axis products across movement primitives.
   - Canonicalizes grouped-query head replication, token-major attention values and outputs, and token-major gated-delta inputs without fixed AST node sequences.
3. **Stateful scan recovery ([`lib/kernel_scan.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_scan.ml))**:
   - `Kernel_scan` recovers explicit induction order and carried tensors from update dependencies, then represents triangular recurrence as a primitive operation.
   - Padded zero-state gated-delta regions execute directly from their serialized assignment bindings.
4. **Target-costed kernel selection ([`lib/kernel_abi.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_abi.ml))**:
   - Chooses shape, dtype, storage-layout, and target-specific tactics from the hardware-probe profile.
   - Q4_0 single-row decode tiles use a separate streaming bandwidth-ramp policy: for sufficiently gridded launches, register pressure affects occupancy facts but does not directly discount the memory-bandwidth ceiling; small grids retain the conservative bandwidth model.
   - Covered fused regions retain primitive source graphs and provide executable boundary values directly to Metal lowering and runtime dispatch.
5. **DAG Concurrency & Antichain Co-Scheduling ([`Pass_co_schedule`](file:///Users/tung/Projects/std23/llmopt/lib/pass_co_schedule.ml), [`lib/dag_analysis.ml`](file:///Users/tung/Projects/std23/llmopt/lib/dag_analysis.ml))**:
   - Analyzes SSA dependency DAGs and computes ready antichains of independent operations (e.g., parallel Q/K/V projections or parallel layer components).
   - Groups independent kernels into concurrent execution stages bounded by `Ir.Op.Barrier_wait` and dispatched via `MTLDispatchTypeConcurrent`.

### 3.2.1 Governing optimizing-compiler transition

The pass list above describes the current executable baseline. The governing
replacement architecture is
[Constructive Assignment Lowering and AOT GPU Command-Queue Compilation](/decisions/constructive-primitive-loop-lowering-and-gpu-queue-compilation.md).
It does not treat fusion and antichains as the same transformation. Instead,
`Compute_plan` supplies one data/state/effect dependency model from which the
compiler derives dependent fusion regions and independent concurrency stages as
separate products.

Captured operations are normalized into an SSA-like assignment graph with
explicit index domains, effects, layouts, state, aliases, resources, and FX
provenance. Target profiles from hardware probing cost legal register, SRAM,
recomputation, reduction, dispatch, and concurrency alternatives. Block-quant
schemas retain exact physical/logical semantics plus target-aware SIMD tactics;
the portable scalar form is a correctness fallback, not the optimization target.

The resulting dispatches, dependencies, and barriers form an AOT GPU command
queue which the Metal backend realizes as a precompiled direct command list.
ICB remains an optional backend tactic for a specialization whose buffers,
parameters, dimensions, and barriers can be bound correctly. This compiler
queue is distinct from the runtime SRPT serving request queue. Existing
`pass_fuse_*.ml` and `metal.ml` paths remain explicit legacy fallbacks until
generated-plan evidence shows the unified path covers their semantics.

## 3.3. Tactic Registry & Microarchitectural Hardware Discovery

Rather than relying on brittle heuristic rules, kernel implementations are resolved through a unified, typed tactic registry ([`lib/kernel_abi.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_abi.ml)):
* **Shape & Layout Selection**: Automatically picks optimal SIMD tiles based on tensor dimensions $(M, N, K)$, storage format (`Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS`, `F32`), and input/output dtypes.
* **Multi-Token & Multi-Column Weight Reuse**:
  - For $M=2$ (two-token prompt or parallel decode steps), specializes kernels to decode each quantized superblock once while accumulating across both token rows simultaneously (`_m2n2` and paired-token tactics).
* **SIMD-Group Matrix Execution**: Dispatches broadcast-aware $8 \times 8$ SIMD-group matrix multiply tiles for aligned shapes with $16 \times 16$ threadgroup fallbacks.
* **Target Hardware Discovery ([`lib/target_hardware.ml`](file:///Users/tung/Projects/std23/llmopt/lib/target_hardware.ml))**: Probes GPU cores, memory bandwidth ($273\text{ GB/s}$ on M4 Pro), SIMD width (32 lanes), SRAM bank capacity, and calculates analytical prefill roofline knees:
  $$M_{\text{knee}} = \left\lceil \frac{W}{B} \times \frac{P}{2 \cdot \text{Params}} \right\rceil$$
  dynamically determining optimal chunk sizes ($M^* \in [64, 512]$) to maximize core saturation while adhering to SLA latency bounds.

### 3.3.1 Shared linear kernel emission

`Metal_q4_0_epilogue_codegen.Writeback` constructs the kernel buffer header and
writeback directly for both Q4_0 and Tile32_Affine. Its `Bias` and `Epilogue`
variants distinguish the flat optional-bias ABI from algebraic epilogue inputs,
including an identity epilogue with no auxiliary buffers. Row-relative bias
addressing stays explicit. No generated shader text is parsed or spliced to
insert an epilogue; x16 emission retains its macro definitions and cleanup.

`Metal_q4_0_decode_codegen.Contraction_source` shares buffer declarations,
weight-row addressing, accumulator declarations, reductions, and auxiliary row
views across x8/x16 single-output and multi-export contractions. Each tactic
retains its existing load and accumulation order; tactic selection is unchanged.

## 3.4. Quantization Ingestion & Weight Architecture

The Python `llmopt_backend.tensor_archive` module owns ABI-v1 (`LLMOPTWT`)
dtype tags, deterministic sorted index layout, 256-byte payload alignment,
streaming atomic writes, and metadata-only inventory reads. `write_archive`
adapts PyTorch tensors; the Gemma builder supplies GGUF payload chunks and its
FP32 reference constants to `write_payloads`; Mellum uses `read_inventory`.
Payload failure removes the temporary file while preserving the destination.

`llmopt` provides first-class support for both native GGUF files and internal aligned binary archives:
* **Direct GGUF Mmap ([`lib/gguf.ml`](file:///Users/tung/Projects/std23/llmopt/lib/gguf.ml))**: Reads GGUF v2/v3 metadata, resolves tensor name aliases, and creates zero-copy 256-byte aligned `MTLBuffer` views directly from memory-mapped GGUF files without intermediate memory copies or conversions.
* **Quantization Coverage**:
  - **Block-32**: `Q4_0`, `Q8_0`, `Q5_0`
  - **Superblock-256 (K-Quants & Unsloth UD)**: `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS`
  - **Groupwise W4**: Group-64 scales with 2-complement low-nibble packing.
* **Hardware Bitfield Acceleration ([`lib/metal.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal.ml))**: Replaced manual shift/mask sequences in Superblock-256 unpacking with native `metal::extract_bits(src, offset, count)` hardware bitfield extraction instructions across all quantized compute and embedding paths.
* **Split-K Intra-Threadgroup Parallel Reductions ([`lib/metal.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal.ml), [`lib/kernel_abi.ml`](file:///Users/tung/Projects/std23/llmopt/lib/kernel_abi.ml))**: For wide reduction dimensions ($K \ge 4096$), partitions inner dimension across 4 SIMDgroups per column in 256-thread threadgroups, reducing partial sums across SIMD groups via threadgroup shared memory in deterministic FP32/FP16 order without global atomics.
* **Canonical `Tile32_Affine` Mathematical Reduction & In-Register Walsh-Hadamard Attention Rotation ([`lib/gguf.ml`](file:///Users/tung/Projects/std23/llmopt/lib/gguf.ml), [`lib/block_schema.ml`](file:///Users/tung/Projects/std23/llmopt/lib/block_schema.ml), [`lib/metal_block_codegen.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_block_codegen.ml), [`lib/metal_cache_source.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_cache_source.ml))**:[^canonical-tile32-affine]
  - **Lossless Offline GGUF $\to$ `Tile32_Affine` Transcoding**: Every GGUF block-32 (`Q4_0`, `Q5_0`, `Q8_0`) and superblock-256 (`Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS`) tensor is losslessly homomorphic to a 32-element SIMD-aligned affine tile $w_i = s \cdot q_i + b$ (`Tile32_Affine4` at 20 bytes / 32 elements for 4-bit formats, `Tile32_Affine8` at 36 bytes / 32 elements for 5/6/8-bit and non-linear LUT formats).
  - **Unified Parametric Contraction Inner Loop**: Eliminates bitplane stitching, 6-bit scale unpacking, and LUT indirection from GPU contractions in favor of a single parametric $B \in \{4, 8\}$ affine accumulation loop $\text{acc} \mathrel{+}= s \sum_{i=0}^{31} a_i q_i + b \sum_{i=0}^{31} a_i$ compatible with single-slot, multi-slot, and multi-export contraction groups.
  - **In-Register 32-Lane Fast Walsh-Hadamard Butterfly (`llmopt_simd_fwht32`)**: Applies a 5-step `simd_shuffle_xor` orthogonal and self-inverse butterfly ($\tilde{H}_{32} = \frac{1}{\sqrt{32}} H_{32}$) in registers prior to `Tile32_Affine` KV-cache quantization, dispersing channel outliers across all 32 SIMD lanes while preserving inner products ($\langle \tilde{H}_{32} q, \tilde{H}_{32} k \rangle = \langle q, k \rangle$).

## 3.5. Static Memory Liveness Planning & Runtime Dispatch

* **Liveness Memory Planner ([`lib/serving_memory_plan.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_memory_plan.ml))**:
  - Solves static interval-graph register allocation across all intermediate tensors.
  - Generates deterministic 256-byte aligned offset assignments into a single retained Metal workspace buffer.
  - Eliminates dynamic `[MTLDevice newBufferWithLength]` allocations during model execution.
* **Native Metal Runtime ([`lib/metal_runtime.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_runtime.ml))**:
  - Acts as the operational facade for GPU execution, coordinating compilation, caching, buffer tracking, and dispatch submission.
  - **Schedule Executor ([`lib/metal_schedule_executor.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_schedule_executor.ml))**: Traverses the solidified command schedule, prepares workspace buffers, handles barrier synchronization, and routes commands through execution state context ([`lib/metal_execution_state.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_execution_state.ml)).
  - **Modular Dispatch Subsystem**: Operation-specific dispatch logic is partitioned into dedicated domain handlers with shared grid and parameter encoding ([`lib/metal_dispatch_common.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_dispatch_common.ml)):
    - `Metal_dispatch_tensor`: Dense tensor movement, casts, embedding, pointwise, and reduction operations.
    - `Metal_dispatch_normalization`: RMSNorm and rotary position embedding kernels.
    - `Metal_dispatch_recurrent`: ShortConv, triangular recurrence, and gated-delta recurrence steps.
    - `Metal_dispatch_attention`: Standard, wide-head, and paged Q8 attention dispatchers.
    - `Metal_dispatch_linear`: Dense and quantized matrix-vector / GEMM operations.
    - `Metal_dispatch_assignment`: Dynamic argument assembly and dispatch for constructive fused assignment regions.
  - Encodes entire schedule execution into consolidated command buffers.
  - Reuses cached `MTLComputePipelineState` objects with automatic tactic candidate fallback.
  - Supports precompiled direct dispatch lists whose dynamic paged-attention and checkpoint parameters are patched for each execution. The earlier ICB realization was removed after it produced a deterministic long-decode collapse despite explicit ICB barriers; the AOT schedule and serving-request queue remain separate contracts.

## 3.6. Serving Engine, Cache Hierarchy, and Speculative Primitives

* **Model Program ABI v7 ([`lib/model_program.ml`](file:///Users/tung/Projects/std23/llmopt/lib/model_program.ml))**:
  - Encapsulates compiled prefill schedules, decode schedules, state plans, tokenizer metadata, generation metadata, optional MTP metadata, and optional prompt-lookup verification metadata into a self-contained root artifact (`model.llmopt`). ABI v7 is the current writer/reader contract; earlier ABI references in historical experiment receipts remain historical.
  - The Q8 physical cache stores heterogeneous key/value regions through explicit logical offsets. The cache pack/unpack and paged-attention kernels consume those offsets, so 64-, 256-, and 512-wide attention heads do not alter the generic cache contract.
* **Serving Engine & Contract Validation ([`lib/serving_engine.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_engine.ml), [`lib/serving_engine_contract.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_engine_contract.ml))**:
  - `Serving_engine_contract` owns head, attention, and recurrent bindings, input/output metadata checks, and package validation for prefill and decode phases.
  - `Serving_engine` remains the unified operational coordinator managing continuous execution steps, logical and physical cache lease/release transactions, and `Prompt_lookup.verify_and_commit`.
* **Rotary Position Tables ([`lib/serving_rope_table.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_rope_table.ml))**:
  - Precomputes and caches FP16 cosine/sine tables from inverse-frequency tensors; provides efficient sliced buffer views and single-slot copies for decode steps.
* **Speculative Decoding Algorithms ([`lib/serving_speculative.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_speculative.ml))**:
  - Isolates pure draft proposal and speculative verification logic independent of runtime lifecycle:
    - `Speculative_acceptance`: Pure greedy $K+1$ draft token matching and acceptance.
    - `Target_coupled_mtp`: Transaction coordinator for target-coupled multi-token prediction.
    - `Medusa`: Tree attention masks, tree generation, and speculative tree candidate evaluation.
    - `Prompt_lookup`: History ngram candidate search and linear/tree proposal generation.
* **Speculative Tree-Mask Attention ([`lib/metal.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal.ml), [`lib/metal_runtime.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_runtime.ml))**:
  - Registers fused tree-attention megakernels (`llmopt_attention_tree_speculative_f16`) for compact bitmask causal adjacency topologies. No current target Model Program invokes them for end-to-end verification.
* **Compressed Radix Cache & Optimistic Rollback ([`lib/radix_cache.ml`](file:///Users/tung/Projects/std23/llmopt/lib/radix_cache.ml), [`lib/serving_cache.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_cache.ml))**:
  - Implements tree-structured prefix caching with compressed edges, LRU leaf eviction, and namespace isolation.[^sglang-radix-cache][^sglang-mamba-radix-cache]
  - Supports tested speculative slot reservation, partial commit, and rollback metadata. These operations are not yet bound to physical Gemma verification writes.
* **Continuous Batching Queue and Acceptance Semantics ([`lib/serving_queue.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_queue.ml), [`lib/serving_engine.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_engine.ml))**:
  - The queue retains its measured prefill/decode SRPT model with no speculative rate multiplier.
  - Pure speculative acceptance and proposal algorithms delegate to `Serving_speculative`. Target-coupled proposal and verification execution remain open under the Gemma MTP runtime contract.
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
| **SmolLM2-135M-Instruct** | Standard Transformer (GQA, SwiGLU, RMSNorm, RoPE) | `Q4_K_M` | **3.4214 ms** | 3.1178 ms | **1.0974x** | Parity achieved; 60 Linear-add & 30 SwiGLU fusions, bitfield unpack. |
| **Qwen3.5-0.8B** | Hybrid / Stateful Scan (18 recurrent layers, gated-delta, ShortConv) | `UD-Q4_K_XL` | **8.1384 ms** | 8.0930 ms | **1.0056x** | Parity achieved; 18 stateful scan regions recovered & 48 Linear-add fusions. |
| **Gemma-4-E2B-it** | Deep Transformer (Wide RMSNorm, Gated MLP, h256 Attention) | `UD-Q4_K_XL` | **19.4005 ms** | 18.8907 ms | **1.0270x** | Parity achieved; wide-row RMSNorm, rotary QK, Split-K parallel reductions. |
| **LFM2.5-350M** | Hybrid Conv / Attention | `W4A16 / KVQ8` | **18.2 ms** (TTFT Turn 1) | 19.1 ms | **0.9529x** | Full serving probe with true suffix prefill and prefix retention. |

These rows are scoped observations from their cited probe artifacts. Performance
ratios, deterministic replay, byte equality, and reference argmax parity remain
separate claims; no row generalizes those properties to every benchmark run.[^slice-31-receipt]

---

# 5. File & ABI Version Registry

| Contract / File Format | Identifier / Magic | Current Version | Authority Module |
|---|---|---|---|
| **Binary Transport** | `LLMOPTFX` | Version 2 | [`lib/capture.ml`](file:///Users/tung/Projects/std23/llmopt/lib/capture.ml), [`python/llmopt_backend/fx_graph.py`](file:///Users/tung/Projects/std23/llmopt/python/llmopt_backend/fx_graph.py) |
| **Serving Package** | `LLMOPTPK` | ABI Version 27 | [`lib/serving_package.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_package.ml) |
| **Serving Schedule** | `LLMOSCH` | Format Version 29 | [`lib/serving_schedule_codec.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_schedule_codec.ml) (facade: [`lib/serving_schedule.ml`](file:///Users/tung/Projects/std23/llmopt/lib/serving_schedule.ml)) |
| **Model Program** | `LLMOPT-MODEL` | ABI Version 7 | [`lib/model_program.ml`](file:///Users/tung/Projects/std23/llmopt/lib/model_program.ml) |
| **Binary Weight Archive** | `LLMOPTWT` | Version 1 | [`lib/weight_archive.ml`](file:///Users/tung/Projects/std23/llmopt/lib/weight_archive.ml) |
| **Native GGUF Ingestion** | `GGUF` | v2 / v3 | [`lib/gguf.ml`](file:///Users/tung/Projects/std23/llmopt/lib/gguf.ml) |

---

[^pytorch-backend-contract]: PyTorch custom backend documentation.
[^sglang-radix-cache]: SGLang `RadixCache`, pinned to revision `d1af3c89233c475fc1bf11939d86787e6cddd58c`.
[^sglang-mamba-radix-cache]: SGLang `MambaRadixCache`, pinned to the same revision.
[^slice-31-receipt]: Full-model multi-campaign receipt in `bench/results/compiler-generalization-slice-31-2026-08-30.json`.
[^canonical-tile32-affine]: Canonical `Tile32_Affine` Weight & KV Format and In-Register Hadamard Attention Rotation decision (`decisions/canonical-tile32-affine-and-hadamard-attention-rotation.md`).
