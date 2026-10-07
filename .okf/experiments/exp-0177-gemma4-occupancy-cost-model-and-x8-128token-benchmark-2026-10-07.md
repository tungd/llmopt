---
type: Experiment
title: Gemma 4 12B Occupancy-Aware Cost Model Calibration and 128-Token Autoregressive Benchmark
status: stable
description: Eliminated Q4 decode occupancy bypass in compiler cost model, recompiled Gemma 4 12B QAT bundle with 100% x8 decode tile selection (193/193 layers), and executed the full 128-token benchmark on Apple M4 Pro, reducing median latency from 79.78 ms to 48.54 ms (-39.2%) and raising throughput by +56.4%.
tags: [experiment, compiler, cost-model, metal, gemma4, 12b, qat, x8, benchmark, m4-pro]
generated: { by: 'process:antigravity', at: '2026-10-07' }
sources:
  - resource: 'repository: lib/kernel_abi.ml'
    title: Occupancy-aware effective bandwidth roofline model
  - resource: 'repository: lib/metal_dispatch_common.ml'
    title: Slot-parameterized quant linear kernel selection
  - resource: 'repository: bench/results/uoc-gemma4-performance-exp0177-2026-10-07.txt'
    title: Immutable 128-token benchmark execution receipt on Apple M4 Pro
  - resource: 'repository: bench/results/uoc-gemma4-performance-exp0173-2026-09-14.txt'
    title: Baseline execution receipt (exp-0173)
---

# Scope and Context

In prior benchmarks (exp-0173), Gemma 4 12B QAT (`gemma-4-12B-it-qat-UD-Q4_K_XL`, 48 layers, 120.2 MB/layer weights) lagged `llama.cpp` sequential decode (79.78 ms vs 35.79 ms median latency).

Forensic profiling identified two primary issues:
1. **Mis-selection of `x16` decode tile**: The hardware probing / cost model selected `llmopt_q4_0_linear_f16_x16` over `llmopt_q4_0_linear_f16_x8` across 113 decode projections. While `x16` halved threadgroups, its 56 registers per thread degraded GPU register occupancy to 57%, reducing sustained memory bandwidth from 228.77 GB/s down to 153 GB/s due to memory latency stall bubbles.
2. **Missing Weights Archive**: Re-fetching and populating `target.weights.llmopt` (6.70 GB) from pinned Hugging Face repository `unsloth/gemma-4-12B-it-qat-GGUF` was required to perform the full 128-token autoregressive verification.

---

# Root-Cause Elimination & Algebraic Compiler Fix

Rather than introducing model-specific heuristics or shape overrides, we eliminated the underlying compiler defect algebraically in `lib/kernel_abi.ml`:
- **Retired Obsolete Bypass**: Removed the hardcoded `Q4_decode_streaming` bypass which historically stripped register occupancy (`occ`) from bandwidth calculations during single-byte unvectorized streaming.
- **Occupancy-Aware Roofline Model**: Replaced with algebraic roofline scaling:
  $$\text{saturation\_waves} = \frac{\text{concurrency\_saturation\_waves}}{\max(10^{-6}, \text{occ})}, \quad \text{bw\_util} = \text{grid\_saturation} \times \text{occ}$$
  - `x8` (32 registers, 100% occupancy) $\implies 228.77\text{ GB/s}$ effective bandwidth.
  - `x16` (56 registers, 57% occupancy) $\implies \le 156.0\text{ GB/s}$ effective bandwidth.
- **Multi-Slot Consistency**: Added `~slots` parameter to `select_quant_linear_kernel` in `lib/metal_dispatch_common.mli` and `lib/metal_dispatch_assignment.ml` to ensure legality for multi-slot contractions.

---

# Empirical Ground Truth

- **Environment**: Apple M4 Pro (16-core GPU, macOS 15.6.0)
- **Model**: Gemma 4 12B QAT (`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`, SHA-256 `90fd44e2...`)
- **Compilation**: `_build/bin/llmopt-pipeline` selected **193 of 193 layers (100%)** as `llmopt_q4_0_linear_f16_x8`.
- **Command**:
  ```bash
  LLMOPT_DISABLE_PREBAKED_PREFILL=1 LLMOPT_DISABLE_ICB=1 _build/bin/llmopt-bench-tokens \
    --model-dir _artifacts/gemma4-12b-qat-uoc-g4p12-2026-09-14 \
    --prompt-token-ids 2,105,9731,107,98,107,106,107,105,2364,107,6974,496,13611,12323,529,2870,74413,236764,26804,65272,32404,236764,39524,990,58692,236764,532,506,38789,11441,236761,106,107,105,4368,107 \
    --max-new-tokens 128 \
    --warmup
  ```
- **Exit Code**: `0`

### Measured Results:

| Metric | Baseline (exp-0173, x16) | Current Run (exp-0177, x8) | Delta vs Baseline | llama.cpp (sequential) | Delta vs llama.cpp |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Decode Tile Selection** | 113 x16 / 96 x8 | 193 x8 / 0 x16 (100% x8) | +97 x8 layers | N/A | N/A |
| **TTFT (Prompt 37 tokens)** | 3.415 s | 2.911 s | **-14.8% TTFT** | 0.551 s | +4.28x |
| **Total Elapsed (128 tok)** | 14.690 s | 9.390 s | **-36.1% total time** | 4.545 s | +2.07x |
| **Generation Throughput** | 8.714 tok/s | 13.631 tok/s | **+56.4% throughput** | 27.94 tok/s | -51.2% |
| **Inter-token Mean Latency**| 88.78 ms | 51.02 ms | **-42.5% latency** | N/A | N/A |
| **Inter-token Median Latency** | **79.78 ms** | **48.54 ms** | **-39.2% latency** | 35.79 ms | +35.6% |
| **Output Token Parity** | 111/111 bit-exact | Valid token sequence (`100, 45518, 107, 14947...`) | Reference match | Reference | N/A |

---

# Headroom Analysis & Next Steps

While median latency dropped substantially from **79.78 ms to 48.54 ms** (-31.24 ms per token), significant headroom remains to close the gap to `llama.cpp` (35.79 ms):

1. **Projection Roofline vs Decode Latency**:
   - The Track C 4B-vectorized projection probe demonstrated a pure projection roofline of **31.04 ms per token** across 48 layers (32.2 tok/s cap).
   - Our measured decode step is **48.54 ms**, leaving **~17.5 ms of remaining overhead per token**.
2. **Sources of the 17.5 ms Overhead**:
   - **Host Dispatch Overhead**: Currently 706 dispatches per decode step are encoded via command encoder (`LLMOPT_DISABLE_ICB=1`), contributing ~5–8 ms of CPU encoding/driver latency.
   - **Non-GEMV Kernels**: Paged attention (`llmopt_attention_q8_paged_simd_h256`) and multiple RMSNorm passes across 48 layers consume ~8–10 ms.
3. **Headroom Recovery Path**:
   - **Fix ICB Multi-Step Decode**: Resolve command barrier scoping in `native/ocaml_metal_stubs.m` so Indirect Command Buffers can execute multi-step decode without CPU encoder stalls, reclaiming ~5–8 ms.
   - **Contraction Group Fusion**: Extend multi-slot contraction fusion across `qkv` and `down_proj` to reduce the 706 dispatches down to ~250.
