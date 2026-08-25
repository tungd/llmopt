---
type: Decision
title: 'DAG Concurrency Analysis and Complementary Co-Scheduling Optimizer Pass'
description: 'An optimizing compiler pass that analyzes captured FX/SSA graph dependencies, identifies parallel antichains, pairs memory-bound and compute-bound kernels, and generates a staged concurrent Metal execution plan.'
tags: [decision, compiler, dag, co-scheduling, metal, queueing, concurrency, passes]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T01:25:00Z' }
sources:
  - id: local-passes
    resource: /lib/passes.ml
    title: Optimizer passes and pattern rewrites
  - id: local-ir
    resource: /lib/ir.ml
    title: SSA IR and Graph structures
  - id: local-schedule
    resource: /lib/serving_schedule.ml
    title: Binary typed command schedule
  - id: local-memory-plan
    resource: /lib/serving_memory_plan.ml
    title: Workspace liveness analysis and offset allocator
  - id: local-metal-stubs
    resource: /native/ocaml_metal_stubs.m
    title: Metal command encoding and dispatch bindings
---

# Problem Statement

Currently, `llmopt` lowers an optimized `Ir.Graph.t` into a **strictly serial sequence of commands** in [`lib/serving_schedule.ml`](../../lib/serving_schedule.ml). At runtime, [`lib/metal_runtime.ml`](../../lib/metal_runtime.ml) encodes each kernel sequentially into a single `MTLComputeCommandEncoder`.

### Bottlenecks of Purely Serial Execution:
1. **GPU Under-Occupancy:** Small memory-bound operations (e.g. 1-row RMSNorm, RoPE, ShortConv 1D state shifts, and KV cache packing) utilize less than 1% of the GPU compute cores on Apple Silicon (M-series GPUs with 16 to 40+ cores).
2. **Resource Starvation:** During memory-bound kernels, the GPU ALUs sit idle; during compute-bound GEMMs, memory bus bandwidth is under-saturated.
3. **Unnecessary Synchronization:** Independent branches in the computational DAG (e.g., ShortConv state update vs. Attention QKV projection vs. RoPE table preparation) are artificially serialized, creating artificial critical paths.

---

# Architectural Concept: The Co-Scheduling Optimizer Pass

We introduce an optimizing compiler pass: **`Passes.co_schedule`** (and an execution stage planner) that transforms the SSA IR graph into a **Staged Concurrent Execution Plan**.

```
                        [ Captured SSA Graph (Ir.Graph.t) ]
                                         │
                                         ▼
                        [ 1. Macro-Fusion Pass (Passes.optimize) ]
                          - Fused RMSNorm-RoPE
                          - Fused SwiGLU Dual Projections (W1 & W3)
                          - Fused Q8 Linear-SiLU / Add / Mul-Add
                                         │
                                         ▼
                        [ 2. DAG Antichain & Critical Path Analysis ]
                          - Build Dependency Graph G = (V, E)
                          - Compute Transitive Reduction & Slack
                          - Identify Independent Antichains A_t
                                         │
                                         ▼
                        [ 3. Resource Classification & Complementary Pairing ]
                          - Class A: Memory-Bound (RMSNorm, ShortConv, KV Pack)
                          - Class B: Compute-Bound (Q8 Linear, GEMM, FFN)
                          - Pair (Class A ⊗ Class B) in Concurrent Stages
                                         │
                                         ▼
                        [ 4. Hazard-Free Workspace Memory Layout ]
                          - 2D Liveness & Memory Interval Packing
                          - Guarantee Disjoint Offsets for Concurrent Nodes
                                         │
                                         ▼
                        [ 5. Staged Concurrent Serving Schedule ]
                          - Emits Staged Schedule (Sequential | Concurrent)
                          - Metal Runtime: MTLDispatchTypeConcurrent
```

---

# Detailed Algorithm Design

## Step 1: DAG Dependency & Antichain Extraction
For an SSA graph $G = (V, E)$:
* An edge $(u, v) \in E$ exists if node $v$ consumes the output tensor of node $u$.
* For each time step $t$, the **Ready Antichain** $\mathcal{A}_t \subset V$ is the set of unexecuted nodes whose predecessors have all completed:
  $$\mathcal{A}_t = \{ v \in V \setminus \text{Scheduled} \mid \forall u \in \text{Pred}(v), u \in \text{Scheduled} \}$$

## Step 2: Hardware Bottleneck Classification
Each IR node $v$ is classified by its arithmetic intensity:

$$\text{Intensity}(v) = \frac{\text{FLOPs}(v)}{\text{Bytes Transferred}(v)}$$

* **$\text{Class}_{\text{Mem}}$ (Memory-Bound / Low Occupancy):** $\text{Intensity}(v) < \theta_{\text{threshold}}$ (e.g. RMSNorm, RoPE, ShortConv state update, Pointwise, KV pack/unpack).
* **$\text{Class}_{\text{Comp}}$ (Compute-Bound / High Occupancy):** $\text{Intensity}(v) \ge \theta_{\text{threshold}}$ (e.g. Q8 Linear, GEMM, FFN projections).

## Step 3: Queue-Theoretic Complementary Pairing
When $|\mathcal{A}_t| > 1$, the scheduler forms **Concurrent Dispatch Stages**:
* If $\mathcal{A}_t$ contains both $u \in \text{Class}_{\text{Mem}}$ and $w \in \text{Class}_{\text{Comp}}$:
  * Check for memory aliasing between $\text{Outputs}(u) \cup \text{Inputs}(u)$ and $\text{Outputs}(w) \cup \text{Inputs}(w)$.
  * If disjoint, emit a **Concurrent Dispatch Block**:
    $$\text{Stage}_t = \text{Concurrent } [u; w]$$
* Otherwise, emit the node with the **minimum slack on the Critical Path (CPM)** first.

## Step 4: Disjoint Workspace Memory Planning
[`lib/serving_memory_plan.ml`](../../lib/serving_memory_plan.ml) is extended to support concurrent execution:
* For any two nodes $u, w$ in the same `Concurrent` stage:
  $$\text{Interval}(u) \cap \text{Interval}(w) = \emptyset \implies \text{Offset}(u) + \text{Bytes}(u) \le \text{Offset}(w) \lor \text{Offset}(w) + \text{Bytes}(w) \le \text{Offset}(u)$$
* The memory planner places their intermediate buffers into non-overlapping offsets within the single retained Metal workspace buffer.

---

# Data Structures & Interface Specifications

## 1. Staged Schedule Representation (`lib/serving_schedule.mli`)

```ocaml
module Stage : sig
  type t =
    | Sequential of Command.t
    | Concurrent of Command.t list
    | Barrier

  val commands : t -> Command.t list
end

type t = {
  stages : Stage.t list;
  runtime_inputs : Tensor_input.t list;
  workspace_bytes : int;
}

val of_graph_concurrent : Ir.Graph.t -> (t, string) result
```

## 2. Metal Runtime Concurrent Dispatch (`native/ocaml_metal_stubs.m`)

Apple Metal supports concurrent kernel execution within a command buffer using `MTLDispatchTypeConcurrent`:

```objc
// In ocaml_metal_stubs.m:
CAMLprim value caml_llmopt_metal_batch_dispatch_concurrent(value arguments) {
  // Sets compute command encoder with MTLDispatchTypeConcurrent
  // Dispatches multiple compute pipelines without waiting for previous completion
  // Metal hardware scheduler distributes threadgroups across idle GPU Execution Units
}
```

---

# Concrete Application to LFM2.5-350M Architecture

Across each of the 16 layers of LFM2.5-350M:

| Layer Stage | Independent Branches | Co-Scheduled Pairing | Expected Acceleration |
|---|---|---|---|
| **Layer Input** | 1. Input RMSNorm<br>2. Recurrent Conv State Shift | Co-dispatch RMSNorm + ShortConv state shift | Overlaps state update latency |
| **Attention Block** | 1. QKV Projections (Compute)<br>2. RoPE Cosine/Sine Table Gather (Mem) | Co-dispatch QKV GEMM + RoPE Gather | Fills memory bubbles during GEMM |
| **FFN Block** | 1. $w_1$ (Gate) Projection<br>2. $w_3$ (Up) Projection | Fused Dual-Linear Q8 GEMV Kernel ($[W_1; W_3] x$) | 2× reduction in memory bandwidth |
| **KV Cache Store** | 1. KV Attention Page Pack<br>2. Next Layer RMSNorm | Co-dispatch KV Cache Pack + Layer Norm | Zero-overhead KV cache writes |

---

# Verification & Success Gates

1. **Numerical Bitwise Parity:**
   * Output tensors of co-scheduled executions must match the CPU reference (`lib/cpu.ml`) with max absolute difference $< 10^{-4}$ in FP16.
2. **GPU Core Occupancy & Latency Measurement:**
   * Measure kernel duration using Metal GPU counter sample buffers (`MTLCounterSampleBuffer`).
   * Target $\ge 20\%$ reduction in single-token decode latency ($\text{TPOT}$) and $\ge 15\%$ reduction in prefill latency on Apple Silicon M-series GPUs.
