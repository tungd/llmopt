---
type: Experiment
title: 'Expanded M=16 prefill tiling and threadgroup SRAM activation staging for Gemma 4'
description: 'Implements M_tile=16 prefill GEMM for Q4_0 linear, QKV, and gated operations reducing weight passes from 10 to 3 for prompt-37, adds cooperative threadgroup SRAM activation staging in decode GEMV eliminating redundant DRAM traffic, and preserves 100% bit-exact 128-token parity.'
tags: [experiment, compiler, prefill, gemm, sram-staging, metal, gemma4]
status: stable
generated: { by: antigravity/2.0, at: '2026-09-04T18:25:00+07:00' }
sources:
  - id: decision-round2
    resource: /.okf/decisions/gemma4-round2-sram-staging-and-m32-tiling.md
    title: Gemma 4 Round 2 SRAM Activation Staging and M=16/32 Prefill Tiling
  - id: decision-batched-prefill
    resource: /.okf/decisions/gemma4-batched-prefill-and-pipelined-queue.md
    title: Batched Prefill GEMM and Pipelined Queue Execution for Gemma 4
  - id: block-codegen
    resource: /lib/metal_block_codegen.ml
    title: M=16 batched Q4_0 linear prefill GEMM micro-ops and SRAM decode staging
  - id: metal-emitter
    resource: /lib/metal.ml
    title: Metal code generation for M=16 prefill QKV and gated linear templates
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: M=16 prefill dispatch, grid sizing, and tactic selection
  - id: prior-device-run
    resource: /.okf/experiments/exp-0146-gemma4-performance-parity-2026-09-04.md
    title: Round 1 batched prefill GEMM and pipelined queue decode receipt
  - id: device-run-receipt
    resource: /bench/results/uoc-gemma4-performance-round2-2026-09-04.txt
    title: Gemma 4 Round 2 performance execution receipt
---

# Result

Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal advances toward `llama.cpp` throughput parity while strictly avoiding monolithic handwritten kernel proliferation:

1. **Expanded Prefill GEMM Tiling ($M_{\text{tile}} = 16$)**:
   - Rather than splitting the 37-token prompt into 10 passes ($M=4$) streaming 60.8 GB of weights from DRAM, the parameterized $M=16$ prefill templates (`llmopt_q4_0_linear_f16_m16_add`, `llmopt_q4_0_qkv_linear_f16_m16`, `llmopt_q4_0_gated_linear_f16_m16`) stream weights only $\lceil 37/16 \rceil = 3$ times (18.24 GB DRAM traffic).
   - Time-to-first-token (TTFT) drops to `1.409 s` (a further 15.5% reduction vs exp-0146 `1.668 s`, and 4.6x faster than exp-0145 naive baseline `6.536 s`).

2. **Threadgroup SRAM Activation Staging in Decode GEMV**:
   - In decode GEMV ($M=1$), the cooperative threadgroup SRAM staging pattern loads the activation vector cooperatively across all threads into `threadgroup half s_activation[4096]` followed by `threadgroup_barrier(mem_flags::mem_threadgroup)`.
   - SIMD groups stream activations directly from on-chip SRAM instead of repeatedly reloading from device DRAM, eliminating ~3.6 GB of DRAM reload traffic per decode token across the 48 layers.

3. **Bit-Exact Token Parity**:
   - 100% bit-exact sequence parity maintained across all 128 generated tokens on the pinned prompt:
     `100, 45518, 107, 14947, 135383, 568, 11124, 568, 147505, 236789, ...`.

# Static Validation

`ninja -f ninja.build all test` passes cleanly with all 79 Python unit tests and all canonical OCaml test suites passing. Clean orthogonal dequant micro-ops and parameterized templates preserved throughout.

# Device Execution Receipt

Full 128-token autoregressive generation on Apple Metal succeeded
(`_artifacts/gemma4-12b-qat-round2-2026-09-04`):
- **TTFT**: `1.409365 s` (1.18x faster than exp-0146 `1.668463 s`; 4.6x faster than exp-0145 `6.536144 s`).
- **Total Elapsed**: `12.386513 s`.
- **Generation Throughput**: `10.333820 tok/s`.
- **Inter-token Mean Latency**: `86.434240 ms`.
- **Inter-token Median Latency**: `85.938931 ms`.
- **Bit-Exact Parity**: All 128 tokens match the golden sequence from exp-0145 and exp-0146 with 100% bit-exact fidelity.

Evidence: `bench/results/uoc-gemma4-performance-round2-2026-09-04.txt` records the measured 128-token Metal execution receipt.
