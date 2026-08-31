---
type: Decision
title: 'Ahead-Of-Time Decode Solidification and Zero-JIT Serving Architecture'
description: 'Move all graph specialization, memory planning, and shape binding into the offline AOT compiler pipeline so the serving runtime is a model-agnostic, zero-JIT execution engine capable of hosting any PyTorch model trivially.'
tags: [decision, compiler, aot, serving, zero-jit, paged-attention, fusions, portability]
status: stable
generated: { by: 'process:antigravity', at: '2026-08-28T02:00:00+07:00' }
sources:
  - id: user-vision
    resource: scope:conversation:2026-08-28-aot-serving-vision
    title: PyTorch model to serving binary AOT vision
    author: human:tung
  - id: compiler
    resource: /bin/fx_compile.ml
    title: AOT FX compilation pipeline
  - id: passes
    resource: /lib/passes.ml
    title: Compiler optimization passes
  - id: package
    resource: /lib/serving_package.ml
    title: Solidified serving package format
  - id: engine
    resource: /lib/serving_engine.ml
    title: Zero-JIT serving engine execution
---

# Decision

All graph transformations, memory planning, buffer offset assignments, and operator fusions MUST execute **Ahead-Of-Time (AOT)** in the compiler pipeline (`llmopt-fx` / `llmopt-compile`). 

The output of the compiler is a **solidified, fully specialized `package.llmopt`** and an accompanying `kernel.metallib`. 

The serving runtime (`llmopt-serve`) MUST be a **pure, zero-JIT execution engine**. It is prohibited from performing:
1. Runtime schedule mutations (no `Serving_schedule.Lfm25.specialize_decode_paged_q8`).
2. Runtime IR graph rewrites or node replacements.
3. Runtime memory planning or workspace recalculations (`Serving_memory_plan.create`).
4. Maintaining per-sequence-length schedule caches (`decode_schedules : Hashtbl.t`).

---

# Problem Statement & Historical Context

The original architecture of `llmopt` was conceived as a compiler:
$$\text{PyTorch Model} \xrightarrow{\text{torch.compile / FX}} \text{Optimizing Graph Passes} \xrightarrow{\text{Solidified Package}} \text{Native Metal Serving Binary}$$

However, during initial KV-cache and recurrent state integration, runtime shortcuts crept into the codebase:
- PyTorch FX captured standard unpaged `Attention` with dynamic concatenation and full causal masks.
- Instead of compiling this into a finalized serving schedule AOT, a runtime module (`Serving_schedule.Lfm25`) was introduced.
- At runtime, on every new prompt length, `Serving_engine` invoked `specialize_decode_paged_q8` to mutate IR node shapes, recalculate memory plans, and precompile new dispatch batches.

This introduced three severe regressions:
1. **Model Lock-In:** The serving engine became coupled to model-specific layout assumptions (LFM2.5 layer topologies, hardcoded recurrent bindings). Supporting Gemma, Qwen, Mistral, or Llama would require duplicating bespoke runtime specialization code.
2. **Cold-Start Latency Jitter:** The first decode step of each conversation turn paid a ~4.8 ms CPU penalty re-specializing schedules and re-allocating plans.
3. **Dispatch Bloat:** Unnecessary ops from FX tracing (e.g. 8 causal mask arithmetic kernels and 6 attention transposes in single-token decode) survived in the runtime graph because dead-code elimination was omitted.

---

# Architecture: The Clean Boundary

```
OFFLINE / AOT COMPILER (llmopt-fx)
  PyTorch FX Capture (TorchDynamo)
           │
           ▼
  Generic Optimizing Graph Passes (lib/passes.ml)
     ├── Applicative Horizontal Linear Fusions ([W_gate; W_up]x, [W_q; W_k; W_v]x)
     ├── Paged Attention Lowering (Direct pool/slots references)
     ├── Single-Token Autoregressive Causal Mask Pruning (DCE for Q=1)
     ├── Identity Movement & Transpose Elimination (Zero-cost views for S=1)
     └── Epilogue Linear-Residual-Norm Fusions (RMSNorm(x + W_down y))
           │
           ▼
  Static Memory Planning & Workspace Assignment (Serving_memory_plan)
           │
           ▼
  Solidified Serving Artifacts (_artifacts/<model>/decode/)
     ├── package.llmopt     <── Final schedule, memory plan, static buffer contracts
     └── kernel.metallib    <── Compiled Metal Shading Language binary
══════════════════════════════════════════════════════════════════════════════
ONLINE SERVING RUNTIME (llmopt-serve)
  1. Memory-map package.llmopt, kernel.metallib, and weights.llmopt
  2. Allocate static physical buffers declared in package manifest
  3. Pre-encode dispatches into persistent MTLCommandBuffer
  4. Loop:
       write_uniforms(token, past_tokens, checkpoint);
       commit_and_wait();
       yield_token();
```

---

# The Generic AOT Passes

All transformations operate purely on the generic `Ir.Graph` representation. None contain model-specific branch conditions.

### 1. AOT Paged Attention Lowering
- **Input:** `Ir.Op.Primitive (Ir.Primitive.Attention)` consuming query, key, value, and mask.
- **Transform:** Detects the dynamic KV concatenation chain, bypasses historical concatenation nodes, and rewrites the op to `Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8)`.
- **Buffer Invariant:** References the static `pool` and `slots_scratch` buffers declared in the engine cache contract. Sequence position $T$ is passed as a dynamic runtime scalar uniform, avoiding graph re-specialization.

### 2. Single-Token Autoregressive Causal Mask Pruning
- **Property:** In autoregressive decode ($Q = 1$), all attended keys stored in the KV cache reside at positions $K \le T$. All attended positions are causal.
- **Transform:** Causal mask inputs to `Paged_attention_q8` are pruned. Dead-code elimination (DCE) recursively removes all upstream mask construction nodes (`arange`, `add`, `index`, `expand`, `le`), eliminating 8 kernel dispatches per token.

### 3. Identity Movement & Transpose Elimination
- **Property:** Transposition or reshaping of single-token activations (e.g. `[1, 16, 1, 64] -> [1, 1, 16, 64]`) preserves contiguous row-major memory stride ($1 \times 16 \times 1 \times 64 = 1024$ contiguous elements).
- **Transform:** Rewrites transpositions and contiguous copies into zero-cost identity view reshapes, eliminating 6 kernel dispatches per token.

### 4. Applicative Horizontal Linear Fusion ($f \otimes g$)
- **Property:** Parallel linear projections that consume the identical input tensor $x$ are horizontally fused into multi-matrix GEMV kernels:
  $$\begin{pmatrix} y_1 \\ y_2 \end{pmatrix} = \begin{pmatrix} W_1 \\ W_2 \end{pmatrix} x$$
- **Coverage:** SwiGLU Gate + Up projections (16 layers) and Attention Q, K, V projections (6 layers). Halves DRAM activation read traffic across all transformer and hybrid models.

### 5. Epilogue Linear-Residual-Norm Fusion
- **Property:** Linear down-projections feed residual addition and post-normalization:
  $$x_{l+1} = \text{RMSNorm}(x_l + W_{\text{down}} h)$$
- **Transform:** Evaluates GEMV accumulation, residual addition, and RMS variance reduction in threadgroup registers, writing normalized FP16 activations directly to DRAM. Eliminates intermediate un-normalized residual round-trips.

---

# Verification & Parity Gates

1. **Zero Runtime Schedule Specialization:**
   - `Serving_engine` MUST NOT invoke any schedule mutation functions.
   - `decode_schedules` and `suffix_prefill_schedules` hashtables are removed; `Serving_package.schedule` is consumed directly.
2. **Bit-Exact Token Parity:**
   - Exact match with greedy decoding reference (`518, 509, 7, 708, 2` on smoke test input `1, 2, 3, 4, 5, 6`).
   - Zero output token mismatches across multi-turn server benchmark against `llama.cpp`.
3. **Serving Latency Gate:**
   - Decode TPOT within $\le 95\%$ of `llama.cpp` (~2.0–2.2 ms target).
   - Zero cold rebuild spikes on Token 1.
4. **Model Portability:**
   - Exporting and serving a standard open weights model (Gemma, Qwen, or Llama) requires only `python3 export.py -> llmopt-fx -> llmopt-serve`, with zero modifications to engine source code.
