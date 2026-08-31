---
type: Decision
title: 'Hardware-Aware AOT Compilation and Microarchitectural Discovery'
description: 'Formalize hardware discovery (SIMD lanes, SRAM banks, bank width, SRAM capacity, cache line sizes, dispatch latency) as a first-class compiler phase in llmopt-fx, making graph fusion and scheduling decisions generalizable across models and hardware geometries (similar to GCC -O3 -march=native).'
tags: [decision, compiler, aot, hardware-discovery, simd, sram-banks, bank-conflicts, cost-model, xgboost]
status: stable
generated: { by: 'process:antigravity', at: '2026-08-28T09:59:00Z' }
sources:
  - id: local-cost-model
    resource: /lib/kernel_cost_model.ml
    title: XGBoost cost model and Device profile
  - id: local-passes
    resource: /lib/passes.ml
    title: Optimization passes pipeline
  - id: local-metal
    resource: /lib/metal.ml
    title: Metal Shading Language code generator
  - id: local-fx-compile
    resource: /bin/fx_compile.ml
    title: Ahead-Of-Time compiler CLI
---

# Problem Statement & Context

`llmopt`'s core vision is a generalizable compiler pipeline:
$$\text{PyTorch FX Graph} \xrightarrow{\text{AOT Compilation (-O3)}} \text{Native Serving Binary (\texttt{package.llmopt})}$$

Adopting any new model architecture (Transformers, Liquid Foundation Models, RNNs/SSMs, Hybrids) must be seamless without manual per-kernel patching. However, optimizations that appear beneficial in isolation can be degraded by microarchitectural constraints:
1. **Threadgroup SRAM Bank Conflicts**: Apple Silicon GPUs feature 32 memory banks (4 bytes wide) in threadgroup shared memory. When 32 SIMD lanes read with strides that align to bank multiples (e.g. FP16 elements with stride 2), multi-way bank conflicts serialize SRAM accesses, degrading throughput by $2\times$ or $4\times$.
2. **SRAM Redundancy vs L1 Cache Broadcast**: Computing an intermediate reduction (e.g. RMSNorm) inside each threadgroup across 576 threadgroups repeats work 576 times. An isolated 1-threadgroup kernel that writes once to the GPU L1 texture/constant cache (which broadcasts 128-byte cache lines without bank conflicts) is physically faster.
3. **Dispatch Overhead vs Kernel Concurrency**: At ~6.5 $\mu$s per Metal dispatch, 134 dispatches incur ~0.9 ms of CPU-GPU queue overhead. Fusions must balance dispatch elimination against memory hierarchy efficiency.

To achieve GCC `-O3 -march=native`-level optimization that is universally applicable, microarchitectural parameters (**SIMD lane count, SRAM banks, bank width, SRAM capacity, L1 cache line size, dispatch overhead**) must be part of a formal **Hardware Discovery & Target Profile** phase.

---

# Architecture & Design

```
+-----------------------------------------------------------------------------+
| 1. HARDWARE DISCOVERY PHASE (Target_hardware.discover)                      |
|    - Queries runtime device: Apple M4 Pro / M3 Max / Generic Metal / CUDA   |
|    - Detects: simd_lanes (32), sram_banks (32), bank_width (4B),            |
|               sram_capacity (32KB), l1_line (128B), bandwidth, cores        |
+--------------------------------------┬--------------------------------------+
                                       │
                                       ▼
+-----------------------------------------------------------------------------+
| 2. AOT COMPILER PASSES (Passes.optimize ~target graph)                      |
|    - Bank Conflict Analyzer: bank(lane) = (offset * elem_size / 4) % 32     |
|    - SRAM Fusion vs L1 Broadcast Cost Model:                                |
|        T_sram = N_tgs * (bytes/BW_sram * C_bank) + T_barrier               |
|        T_l1   = T_dispatch + (bytes/BW_l1)                                  |
|      Fuses into SRAM only when T_sram < T_l1.                               |
|    - Algebraic graph fusions (QKV, SwiGLU, Down-Add, RoPE)                  |
+--------------------------------------┬--------------------------------------+
                                       │
                                       ▼
+-----------------------------------------------------------------------------+
| 3. LEARNED & ANALYTICAL COST MODEL (Kernel_cost_model)                      |
|    - Features: (M, N, K, Cores, Bandwidth, SIMD_Lanes, SRAM_Banks, C_bank)  |
|    - Dynamically selects tile geometries (Tm, Tn, Tk) and reduction paths   |
+--------------------------------------┬--------------------------------------+
                                       │
                                       ▼
+-----------------------------------------------------------------------------+
| 4. SOLIDIFIED ARTIFACT EMISSION (Serving_package.create)                    |
|    - Emits package.llmopt with target metadata and minimal MSL kernels       |
|    - Serving engine executes pre-baked command buffer with zero JIT         |
+-----------------------------------------------------------------------------+
```

---

# Mathematical Formulation

### 1. SRAM Bank Conflict Degree
For a SIMD group of width $W = \text{simd\_lanes}$, accessing shared memory with element size $S$ (bytes) and stride $\Delta$ (elements):

$$\text{bank}(\text{lane}) = \left( \left\lfloor \frac{\text{base} + \text{lane} \cdot \Delta \cdot S}{\text{bank\_width\_bytes}} \right\rfloor \right) \bmod \text{sram\_banks}$$

The collision degree $C_{\text{bank}}$ is the maximum number of simultaneous requests to any single bank:

$$C_{\text{bank}} = \max_{b \in [0, \text{sram\_banks}-1]} \sum_{\text{lane}=0}^{W - 1} \mathbb{I}(\text{bank}(\text{lane}) == b)$$

- $C_{\text{bank}} = 1$: Ideal conflict-free access (full bandwidth).
- $C_{\text{bank}} \ge 2$: Access is serialized over $C_{\text{bank}}$ clock cycles.

### 2. Analytical Fusion Decision Rule
When considering fusing an operator into threadgroup SRAM across $N_{\text{tgs}}$ threadgroups versus executing as a separate kernel broadcasting through L1 cache:

$$\Delta T = T_{\text{SRAM\_fused}} - T_{\text{separate\_L1}}$$

$$T_{\text{SRAM\_fused}} = N_{\text{tgs}} \cdot \left( \frac{\text{Bytes}_{\text{work}}}{\text{BW}_{\text{SRAM}}} \cdot C_{\text{bank}} \right) + N_{\text{barriers}} \cdot T_{\text{barrier}}$$

$$T_{\text{separate\_L1}} = T_{\text{dispatch}} + \frac{\text{Bytes}_{\text{write}}}{\text{BW}_{\text{DRAM}}} + \frac{\text{Bytes}_{\text{read}}}{\text{BW}_{\text{L1}}}$$

**Decision Invariant**: The compiler fuses into threadgroup SRAM if and only if $\Delta T < 0$ and $\text{Footprint}_{\text{SRAM}} \le \text{SRAM}_{\text{capacity}}$.

---

# Generalizing the Feature Contract in the Learned Cost Model

In `python/llmopt_backend/cost_model/train.py` and `lib/kernel_cost_model.ml`, the feature vector is generalized from pure tensor dimensions to include hardware geometry:

```python
FEATURE_NAMES = (
    "m", "n", "k",
    "gpu_core_count",
    "memory_bandwidth_gbps",
    "simd_lanes",
    "sram_banks",
    "sram_capacity_bytes",
    "bank_conflict_factor",
    "tile_m", "tile_n", "tile_k",
    "threadgroup_size",
    "mode_code",
)
```

This guarantees that decision trees trained on one chip tier (e.g. M4 Pro) correctly predict performance on other tiers (M1, M2, M3 Max, Ultra) by conditioning on microarchitectural features rather than chip names.
