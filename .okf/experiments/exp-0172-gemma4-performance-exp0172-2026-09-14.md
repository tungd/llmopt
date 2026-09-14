---
type: Experiment
title: 'Gemma 4 12B QAT on-device Metal execution breakthrough (8.87 tok/s, 79.59 ms median latency, 100% bit-exact parity)'
description: 'Executed full-model Gemma 4 (12B QAT UD-Q4_K_XL) on host Apple Silicon Metal (Apple M4 Pro) with the unified optimizing compiler: 128/128 completion tokens strictly matched the golden reference sequence bit-for-bit (100% parity). Measured generation throughput reached 8.87 tok/s (up from 3.44 tok/s in Round 7, a 2.58x speedup), TTFT dropped from 8.12 s to 3.48 s (2.33x faster), and median inter-token latency reached 79.59 ms (2.59x faster).'
tags: [experiment, benchmark, gemma4, 12b, qat, metal, on-device, throughput, latency, bit-exact-parity, m4-pro, apple-silicon]
status: stable
generated: { by: 'process:antigravity', at: '2026-09-14T18:05:00+07:00' }
sources:
  - id: receipt-round7
    resource: /bench/results/uoc-gemma4-performance-round7-2026-09-05.txt
    title: Gemma 4 Round 7 performance execution receipt
  - id: receipt-exp0172
    resource: /bench/results/uoc-gemma4-performance-exp0172-2026-09-14.txt
    title: Gemma 4 exp-0172 performance execution receipt (2026-09-14)
  - id: exp-0152
    resource: /.okf/experiments/exp-0152-gemma4-performance-round7-2026-09-05.md
    title: Gemma 4 Round 7 performance
  - id: exp-0171
    resource: /.okf/experiments/exp-0171-multi-export-schedule-and-metal-execution-2026-09-14.md
    title: Multi-export contraction group schedule and Metal execution
---

# Result

Following the on-device verification of multi-export contraction groups (`exp-0171`), we executed full-model autoregressive generation for **Gemma 4 (12B QAT UD-Q4_K_XL)** on Apple Silicon Metal (Apple M4 Pro GPU) using `_build/bin/llmopt-bench-tokens`.

The benchmark demonstrated a major performance breakthrough across all metrics while preserving **100% bit-exact numerical parity** across all 128 generated completion tokens.

---

## 1. Hardware & Execution Configuration

- **Host**: Apple M4 Pro (macOS 15.6.0, Darwin 25.6.0)
- **GPU**: Apple M4 Pro 16-core GPU, unified memory architecture
- **Model**: Gemma 4 (12B QAT UD-Q4_K_XL, 48 attention layers, vocabulary 262,144)
- **Artifact**: `_artifacts/gemma4-12b-qat-round7-2026-09-05`
  - Model Program SHA-256: `9856d3a011c1c348c6be4391e08ada362efbdf292e98d9ad4c4d91e6b408965a`
  - Prefill Package SHA-256: `9a51f827c9d0664a114abd25003cdef88f78e812f97c9b5707c8d1046720a229` (4,569 commands, 106 kernels)
  - Decode Package SHA-256: `438decddcf36ee85e6bd7a066c3c4b7b2274811d60f685b8b2d86877a0e816a5` (4,617 commands, 106 kernels)
- **Execution Command**:
  ```bash
  _build/bin/llmopt-bench-tokens \
    --model-dir _artifacts/gemma4-12b-qat-round7-2026-09-05 \
    --prompt-token-ids 2,105,9731,107,98,107,106,107,105,2364,107,6974,496,13611,12323,529,2870,74413,236764,26804,65272,32404,236764,39524,990,58692,236764,532,506,38789,11441,236761,106,107,105,4368,107 \
    --max-new-tokens 128 \
    --warmup
  ```

---

## 2. Benchmark Comparison (Round 7 vs Current exp-0172)

| Metric | Round 7 (2026-09-05) | Current (2026-09-14, exp-0172) | Delta / Speedup |
|---|---|---|---|
| **TTFT (Prompt 37 tokens)** | 8.121 s | **3.479 s** | **2.33x faster** (-57.2% latency) |
| **Total Elapsed (128 tokens)** | 37.169 s | **14.432 s** | **2.58x faster** (-61.2% elapsed) |
| **Generation Throughput** | 3.444 tok/s | **8.869 tok/s** | **2.58x higher** (+157.5% throughput) |
| **Inter-token Mean Latency** | 228.72 ms | **86.24 ms** | **2.65x faster** (-62.3% latency) |
| **Inter-token Median Latency** | 206.15 ms | **79.59 ms** | **2.59x faster** (-61.4% latency) |
| **128-Token Parity** | 128/128 exact | **128/128 exact** | **100% bit-exact parity preserved** |

---

## 3. Bit-Exact Token Sequence Verification

All 128 generated completion token IDs strictly match the golden reference sequence bit-for-bit:
```
100,45518,107,14947,135383,568,11124,568,147505,236789,236751,236761,107,14947,135383,236761,107,4225,552,236761,107,4225,552,6795,32404,236764,65272,32404,236764,65272,32404,236764,39524,990,58692,236764,990,58692,236764,990,58692,236764,38789,11441,236764,990,58692,236764,38789,11441,236764,990,58692,236764,38789,11441,236764,32404,236764,38789,11441,236764,32404,236764,990,58692,236764,32404,236764,38789,11441,236764,32404,236764,38789,11441,236764,32404,236764,32404,236764,38789,236764,990,58692,236764,32404,236764,38789,236764,38789,236764,32404,236764,38789,236764,4044,236761,107,140,236829,107,140,236829,107,140,236829,139,236772,58991,236764,4044,236761,107,140,236829,139,236772,58991,236764,4044,236761,107,140,236829,139,236772,58991
```

Receipt saved at: `bench/results/uoc-gemma4-performance-exp0172-2026-09-14.txt`.
