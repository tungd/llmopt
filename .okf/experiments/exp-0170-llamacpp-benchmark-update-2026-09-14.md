---
type: Experiment
title: 'llama.cpp benchmark update on Apple M4 Pro (LFM2.5-350M Q4_0 and Q8_0)'
description: 'Refreshed native llama-bench throughput measurements for LiquidAI/LFM2.5-350M-GGUF across Q4_0 and Q8_0 on Apple M4 Pro (Metal build 10531). Q4_0 achieved 8,122.25 prompt tok/s and 501.82 generation tok/s (1.99 ms/tok); Q8_0 achieved 7,965.82 prompt tok/s and 362.46 generation tok/s (2.76 ms/tok).'
tags: [experiment, benchmark, llama.cpp, llama-bench, lfm2.5, q4_0, q8_0, throughput, metal, m4-pro]
status: stable
generated: { by: 'process:antigravity', at: '2026-09-14T15:52:00+07:00' }
sources:
  - id: llama-bench-runner
    resource: /bench/llama_cpp_bench.py
    title: native llama-bench wrapper and JSON receipt writer
  - id: receipt-q4
    resource: /bench/results/lfm25-350m-llama-cpp-q4-2026-09-14.json
    title: llama-bench receipt for LFM2.5-350M Q4_0 (2026-09-14)
  - id: receipt-q8
    resource: /bench/results/lfm25-350m-llama-cpp-q8-2026-09-14.json
    title: llama-bench receipt for LFM2.5-350M Q8_0 (2026-09-14)
  - id: historical-q8
    resource: /bench/results/lfm25-350m-llama-cpp-q8-2026-08-26.json
    title: historical llama-bench receipt for LFM2.5-350M Q8_0 (2026-08-26)
  - id: benchmark-doc
    resource: /.okf/benchmarks/llama-cpp.md
    title: LFM2.5-350M llama.cpp comparison and ERS side benchmark protocol
---

# Result

Refreshed native `llama-bench` throughput numbers on Apple M4 Pro for the `LiquidAI/LFM2.5-350M-GGUF` model target across both `Q4_0` and `Q8_0` quantizations.

## Hardware & Environment

- **Host**: Apple M4 Pro (macOS 15.6.0, Darwin 25.6.0)
- **GPU**: Apple M4 Pro 16-core GPU, unified memory architecture (MTL, BLAS)
- **Engine**: `llama.cpp` version 0.1.2-dev (build 10531, commit `f20395d`)
- **Metal runtime**: embedded metal library loaded in 0.008 s, residency sets enabled

## Benchmark Configuration

Following the standard repository benchmark protocol ([.okf/benchmarks/llama-cpp.md](file:///Users/tung/Projects/std23/llmopt/.okf/benchmarks/llama-cpp.md)):
- Prompt tokens: 512
- Generation tokens: 128
- Repetitions: 5
- Batch size: 2,048
- Micro-batch size (`ubatch`): 512
- GPU layers: 99 (all layers offloaded to GPU)
- Warmup: enabled (1 warmup run prior to timed samples)

## Measured Throughput (2026-09-14)

### 1. Q4_0 (`LiquidAI/LFM2.5-350M-GGUF:Q4_0`)
- **Model size**: 216.9 MB (354,483,968 parameters)
- **Prompt processing (512 tokens)**:
  - Average latency: **63.09 ms** (stddev: 2.00 ms)
  - Throughput: **8,122.25 tokens/s** (stddev: 254.33 tok/s)
  - Samples (ms): `[62.75, 62.38, 66.17, 60.68, 63.45]`
- **Autoregressive generation (128 tokens)**:
  - Average latency: **255.85 ms** (stddev: 15.87 ms)
  - Per-token latency (TPOT): **1.999 ms / token**
  - Throughput: **501.82 tokens/s** (stddev: 30.71 tok/s)
  - Samples (tok/s): `[495.39, 537.41, 525.59, 491.12, 459.58]`
- **Receipt**: `bench/results/lfm25-350m-llama-cpp-q4-2026-09-14.json`

### 2. Q8_0 (`LiquidAI/LFM2.5-350M-GGUF:Q8_0`)
- **Model size**: 376.8 MB (354,483,968 parameters)
- **Prompt processing (512 tokens)**:
  - Average latency: **64.31 ms** (stddev: 1.57 ms)
  - Throughput: **7,965.82 tokens/s** (stddev: 194.31 tok/s)
  - Samples (ms): `[62.29, 66.06, 63.98, 63.51, 65.69]`
- **Autoregressive generation (128 tokens)**:
  - Average latency: **354.00 ms** (stddev: 19.97 ms)
  - Per-token latency (TPOT): **2.766 ms / token**
  - Throughput: **362.46 tokens/s** (stddev: 19.57 tok/s)
  - Samples (tok/s): `[366.53, 358.80, 330.97, 374.39, 381.63]`
- **Receipt**: `bench/results/lfm25-350m-llama-cpp-q8-2026-09-14.json`

## Historical Comparison (Q8_0 Baseline)

Comparing current measurements against the initial August 26, 2026 baseline (`bench/results/lfm25-350m-llama-cpp-q8-2026-08-26.json`):

| Metric | Baseline (2026-08-26) | Current (2026-09-14) | Delta |
|---|---|---|---|
| Q8_0 Prompt Throughput | 8,162.33 tok/s (62.79 ms) | 7,965.82 tok/s (64.31 ms) | -2.4% (within run-to-run noise) |
| Q8_0 Generation Throughput | 367.86 tok/s (347.99 ms) | 362.46 tok/s (354.00 ms) | -1.5% (within run-to-run noise) |
| Q4_0 Prompt Throughput | N/A | 8,122.25 tok/s (63.09 ms) | Established current Q4_0 baseline |
| Q4_0 Generation Throughput | N/A | 501.82 tok/s (255.85 ms) | Established current Q4_0 baseline |

The numbers confirm that the underlying hardware and `llama.cpp` runtime exhibit negligible drift (< 2.5% variation across weeks), providing a solid and verified target ceiling for our compiler and serving optimizations.
