---
type: Experiment
title: 'Batched prefill GEMM and pipelined queue decode execution for Gemma 4'
description: 'Implements M-tiled prefill GEMM (M >= 4) for Q4_0 linear and gated operations to stream weights once per batch, implements depth-2 pipelined decode execution in Ring_queue with GPU-resident token chaining, and achieves 10.36 tok/s full generation throughput with bit-exact 128-token parity.'
tags: [experiment, compiler, prefill, gemm, pipelining, ring-queue, metal, gemma4]
status: stable
generated: { by: antigravity/2.0, at: '2026-09-04T17:45:00+07:00' }
sources:
  - id: decision-batched-prefill
    resource: /.okf/decisions/gemma4-batched-prefill-and-pipelined-queue.md
    title: Batched Prefill GEMM and Pipelined Queue Execution for Gemma 4
  - id: block-codegen
    resource: /lib/metal_block_codegen.ml
    title: Batched Q4_0 linear prefill GEMM micro-ops
  - id: metal-emitter
    resource: /lib/metal.ml
    title: Metal code generation for batched prefill GEMM and chain token micro-op
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Batched linear and QKV prefill dispatch and queue configuration
  - id: native-queue
    resource: /native/ocaml_metal_stubs.m
    title: Pipelined depth-2 ring queue worker loop
  - id: engine
    resource: /lib/serving_engine.ml
    title: Asynchronous ring queue static decode orchestration
  - id: prior-device-run
    resource: /experiments/exp-0145-static-decode-and-schema-metal-2026-09-03.md
    title: Static decode and schema Metal baseline
  - id: device-run-receipt
    resource: /bench/results/uoc-gemma4-performance-parity-2026-09-04.txt
    title: Gemma 4 performance parity execution receipt
---

# Result

Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal achieves a 2.0x end-to-end throughput increase and 3.9x faster time-to-first-token (TTFT) through two orthogonal compiler-level optimizations that adhere strictly to the principle of rejecting bespoke monolithic kernel zoos:

1. **Batched Prefill GEMM ($M \ge 4$)**:
   - Rather than streaming the 6.08 GB model weights 37 separate times from DRAM for a 37-token prompt (~225 GB DRAM traffic), quantized linear, QKV, and gated linear operations are tiled across $M$ tokens concurrently (`llmopt_q4_0_linear_f16_m4_add`, `llmopt_q4_0_qkv_linear_f16_m4`, and `llmopt_q4_0_gated_linear_f16_m4`).
   - Weights are streamed contiguous once across the batch tile, slashing DRAM bandwidth demand and dropping TTFT from `6.54 s` (exp-0145) to `1.67 s` (a 74.5% latency reduction).

2. **Pipelined Queue Decode Execution (`Ring_queue` Depth $D=2$)**:
   - `llmopt_worker_loop` in `native/ocaml_metal_stubs.m` implements depth-2 pipelining where command buffer $t+1$ is encoded and committed to Metal's hardware queue while step $t$ executes on the GPU.
   - The GPU-resident token chaining micro-kernel `llmopt_chain_token_id` casts the on-device argmax output directly into the token buffer on the GPU, avoiding host-device synchronization bubbles.
   - Decode latency drops from `141.57 ms` (mean) / `126.96 ms` (median) in exp-0145 to `84.18 ms` (mean) / `82.73 ms` (median) in exp-0146 (a 1.7x speedup).

# Static Validation

`ninja -f ninja.build all test` passes cleanly with all 79 Python unit tests and all canonical OCaml test suites passing. Codebase preserves clean abstractions with zero handwritten monolithic kernels.

# Device Execution Receipt

Full 128-token autoregressive generation on Apple Metal succeeded
(`_artifacts/gemma4-12b-qat-performance-parity-2026-09-04`):
- **TTFT**: `1.668463 s` (3.9x faster than exp-0145 `6.536144 s`).
- **Total Elapsed**: `12.358919 s` (2.0x faster than exp-0145 `24.514963 s`).
- **Generation Throughput**: `10.356893 tok/s` (2.0x higher than exp-0145 `5.221301 tok/s`).
- **Inter-token Mean Latency**: `84.176816 ms` (1.7x faster than exp-0145 `141.565505 ms`).
- **Inter-token Median Latency**: `82.733870 ms` (1.5x faster than exp-0145 `126.964092 ms`).
- **Bit-Exact Parity**: All 128 tokens match the golden sequence from exp-0145 with 100% bit-exact fidelity (`100, 45518, 107, 14947, 135383,...`).

Evidence: `bench/results/uoc-gemma4-performance-parity-2026-09-04.txt` records the measured 128-token Metal execution receipt.
