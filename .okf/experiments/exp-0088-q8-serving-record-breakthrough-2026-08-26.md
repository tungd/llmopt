---
type: Experiment
title: 'Q8 serving queue decode-first scheduling and record-breaking ERS 0.429 on Apple M4 Pro'
description: 'Resolve serving queue priority inversion by giving active decodes strict priority over monolithic prefills, achieving record-breaking single-stream ERS of 0.4290 (TPOT 6.81 ms) and concurrent trace wall time of 0.202s.'
tags: [experiment, compiler, ocaml, metal, q8, serving, scheduler, latency, record, tpot]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T06:15:00Z' }
sources:
  - id: receipt
    resource: /bench/results/lfm25-350m-q8-record-breakthrough-2026-08-26.txt
    title: Record breakthrough execution receipt
  - id: serving_loop
    resource: /bin/lfm_serve.ml
    title: Serving queue decode-first scheduling event loop
---

# Finding

Under concurrent load (`num_conversations = 2`), the serving event loop suffered from priority inversion: `Serving_queue.pop_next_batch` popped both active decodes and incoming monolithic prefills, but executed the monolithic prefill (~60-100 ms) before active decodes. This starved in-flight decoding requests and inflated TPOT from 12.1 ms to 60.2 ms.

# Resolution

1. **Decode-First Event Loop:** In `bin/lfm_serve.ml`, active decodes are now given execution priority. If active decodes are dispatched, any new prefill candidate is deferred until decode steps advance.
2. **High-Throughput Coalesced Kernels:** Serving verified with coalesced SIMD-pair kernels (`llmopt_q8_gemv_pair_simd`, `llmopt_q8_gemv_silu_pair_simd`, `llmopt_q8_gemv_add_pair_simd`, `llmopt_q8_gemv_mul_add_pair_simd`), yielding 5.92 ms decode step latency.

# Execution Results (Apple M4 Pro)

1. **Single-Stream Scored Smoke Trace (`--max-workers 1`):**
   * 4/4 requests successful (100%), 0 token mismatches.
   * **Overall ERS: 0.429039** (Crushed previous repository record of 0.39089).
   * **Median TPOT: 6.809 ms** (Mean: 6.753 ms, p95: 7.036 ms).
   * **Median TTFT: 75.428 ms** (Mean: 68.665 ms, p95: 95.573 ms).
   * **Wall Clock: 0.356 s**.

2. **Concurrent Multi-Conversation Trace (`trace.num_conversations = 2`):**
   * 4/4 requests successful (100%), 0 token mismatches.
   * **Overall ERS: 0.391994**.
   * **Median TPOT: 8.284 ms** (Down from 60.2 ms under prior priority inversion).
   * **Median TTFT: 60.160 ms**.
   * **Wall Clock: 0.202 s** (5x faster completion of concurrent workload).

# Conclusion

The serving queue flaw has been eliminated, unlocking sub-7ms TPOT on Apple Silicon and setting an all-time repository record ERS score of 0.4290.
