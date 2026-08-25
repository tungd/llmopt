---
type: Decision
title: 'XGBoost / GBDT Micro-Level Kernel Cost Model and Tile Optimizer'
description: 'Train an offline XGBoost decision tree ensemble on Apple Silicon kernel profiling sweeps and transpile the model to standalone OCaml code to dynamically select optimal Metal tile geometries (Tm, Tn, Tk), SIMD group layouts, and GEMV vs GEMM dispatch modes at compile/plan time.'
tags: [decision, compiler, cost-model, xgboost, gbdt, tiling, metal, gemm, gemv]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T02:31:00Z' }
sources:
  - id: local-metal
    resource: /lib/metal.ml
    title: Metal Shading Language emitter and kernel templates
  - id: local-schedule
    resource: /lib/serving_schedule.ml
    title: Serving schedule generator and dispatch grid sizing
  - id: local-metal-runtime
    resource: /lib/metal_runtime.ml
    title: Standalone Metal runtime and kernel loader
  - id: local-bench
    resource: /bench/README.md
    title: Benchmark suite and profiling harness
---

# Problem Statement & Context

Currently, `lib/metal.ml` and `lib/serving_schedule.ml` use **static heuristics** to choose kernel variants and dispatch parameters:
1. Multi-row Q8 linear operations statically hardcode $16 \times 16$ threadgroup tiles with 64-wide reductions across all matrix dimensions ($M, N, K$).
2. Decode linear operations statically switch between single-channel 32-lane GEMV and paired 2-channel SIMD GEMV using hardcoded shape checks.
3. Apple Silicon GPUs vary significantly across generations and tiers (M1 to M4; Base 8-core, Pro 16-core, Max 40-core, Ultra 80-core; memory bandwidth from 100 GB/s to 800+ GB/s). A tile size that saturates an M4 Pro can underutilize an M3 Max or exceed register pressure limits on an M1 base chip.

---

# Architecture & ML Pipeline Design

```
+-----------------------------------------------------------------------------------+
| 1. OFFLINE PROFILING SWEEP (bench/bench_kernel_sweep.py)                          |
|    - Microbenchmark parameter grid: (M, N, K, Tm, Tn, Tk, SIMD_lanes, Unroll)    |
|    - Measure GPU execution latency (μs) on Apple Silicon Metal                    |
+-----------------------------------------┬-----------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
| 2. XGBOOST MODEL TRAINING (python/llmopt_backend/cost_model/train.py)             |
|    - Features: Shape (M, N, K), Device (Cores, Bandwidth), Tile (Tm, Tn, Tk)     |
|    - Target: Latency (μs) or Optimal Class (1-row GEMV, Paired GEMV, Tiled GEMM)  |
|    - Loss: Mean Squared Logarithmic Error (MSLE) or Multi-class Log-loss          |
+-----------------------------------------┬-----------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
| 3. OCAML MODEL TRANSPILER (python/llmopt_backend/cost_model/transpile_ocaml.py)   |
|    - Converts trained XGBoost decision trees into pure OCaml (match/if AST)       |
|    - Emits lib/kernel_cost_model.ml & lib/kernel_cost_model.mli                   |
|    - Zero external C/Python dependencies; evaluation latency < 0.5 μs at plan time |
+-----------------------------------------┬-----------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
| 4. COMPILER INTEGRATION (lib/metal.ml & lib/serving_schedule.ml)                  |
|    - Serving_schedule queries Kernel_cost_model.select_tile(M, N, K, Device)      |
|    - Selects parameterized MSL templates & dispatch grid at compilation           |
+-----------------------------------------------------------------------------------+
```

---

# Mathematical Formulation & Feature Engineering

### 1. Cost Model Formulation
Let $\mathcal{C} = \{ (T_m, T_n, T_k, \text{mode}) \}$ be the discrete set of candidate tile configurations for a given linear/matmul operation.
The cost model predicts:

$$\hat{y} = f_{\text{XGB}}(M, N, K, \text{CoreCount}, \text{BandwidthGBps}, T_m, T_n, T_k, \text{mode})$$

At compile time, the planner evaluates:
$$(T_m^*, T_n^*, T_k^*, \text{mode}^*) = \arg\min_{c \in \mathcal{C}} f_{\text{XGB}}(\mathbf{x}_{\text{shape}}, \mathbf{x}_{\text{device}}, c)$$

### 2. Feature Vectors
* **Shape Features:** $\log_2(M)$, $\log_2(N)$, $\log_2(K)$, $M \cdot N$, $M \cdot K$, aspect ratio $M / N$.
* **Hardware Features:** GPU core count ($c \in [8, 80]$), theoretical memory bandwidth (GB/s), maximum threads per threadgroup.
* **Tiling Candidate Features:** $T_m \in \{1, 8, 16, 32\}$, $T_n \in \{16, 32, 64, 128\}$, $T_k \in \{32, 64, 128, 256\}$, threadgroup size $T_m \cdot T_n / 4$, unroll depth.

---

# Standalone OCaml Transpilation

To ensure `llmopt`'s compiler remains a self-contained, high-performance OCaml binary with **zero Python or C library runtime dependencies**, the trained XGBoost trees are transpiled to native OCaml nested conditionals:

```ocaml
(* Generated in lib/kernel_cost_model.ml *)
module Tree_0 = struct
  let eval (m : int) (n : int) (k : int) (cores : int) : float =
    if m <= 1 then
      if n <= 1024 then -0.842 else -0.125
    else if m <= 16 then
      if k <= 4096 then 0.231 else 0.654
    else 1.102
end
```

Evaluation of the full 20-tree ensemble takes **less than 1 microsecond** in OCaml native code.

---

# Verification & Success Gates

1. **Model Prediction Accuracy:**
   * $R^2 \ge 0.92$ on held-out Apple Silicon microbenchmark test sets.
   * Top-1 tile selection accuracy $\ge 95\%$ compared to exhaustive grid sweeps.
2. **Transpilation Integrity:**
   * OCaml transpiled tree outputs match Python XGBoost predictions within floating-point tolerance ($< 10^{-5}$).
3. **Execution Latency Gains:**
   * $\ge 15\%$ speedup in Q8 linear execution for intermediate prompt lengths ($M \in [2, 64]$) where static heuristics fail to balance GEMV and GEMM.
