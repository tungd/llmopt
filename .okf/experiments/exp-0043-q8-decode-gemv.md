---
type: Experiment
title: 'Q8 decode-specialized Metal GEMV'
description: 'Select a vectorized one-row Q8 kernel for decode while retaining tiled Q8 GEMM for prefill, then record exact parity and the matched ERS observation.'
tags: [experiment, ocaml, metal, q8, gemv, decode, serving, ers, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-24T03:12:00Z' }
sources:
  - id: emitter
    resource: /lib/metal.ml
    title: Q8 tiled GEMM and decode GEMV emitters
  - id: dispatcher
    resource: /lib/metal_runtime.ml
    title: Shape-selected Q8 dispatch
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-native-gemv-2026-08-24.txt
    title: Exact device and matched HTTP observation
---

# Optimization boundary

The 16 by 16 Q8 kernel is appropriate for multi-row prefill but decode has
`m = 1`. It previously launched sixteen thread rows, staged sixteen input rows,
and executed the reduction for all of them even though only one output row was
valid.

# Implementation

The Metal emitter now declares float16 and float32 Q8 GEMV entries with 256
threads per threadgroup. One thread owns each output channel, loads activations
as `half4` or `float4`, loads Q8 weights as `char4`, and accumulates in ascending
K order. Unaligned weight rows use the scalar path. The runtime selects GEMV
only when `m = 1`; larger Q8 operations retain the tiled kernel.

The binary-input offline replan emits 48 prefill and 46 decode entries, zero
opaque commands, and 241 validated tensor bindings against the shared
422,137,216-byte binary tensor archive.

# Correctness evidence

At 58% free memory with no model process, one 30-second-supervised Apple M4 Pro
probe selected `llmopt_q8_gemv` and retained 40/40 exact outputs across 135
commands and 39 dispatched kernels. The generated fixture used a 9,984-byte
workspace.

The one matched model attempt completed 4/4 requests, retained the same four
eager-Q8 token sequences, and retained 80/194 cached prompt tokens.

# Matched observation

| Metric | Batched tiled Q8 | Decode GEMV | GEMV minus tiled |
|---|---:|---:|---:|
| ERS | 0.11058587181748172 | 0.10860341576307225 | -0.0019824560544094705 |
| Median TTFT | 1095.193854504032 ms | 1007.2229375073221 ms | -87.97091699671 ms |
| Median TPOT | 106.2433541713593 ms | 96.3275695006208 ms | -9.915784670738503 ms |
| Cached prompt tokens | 80/194 | 80/194 | 0 |
| Exact eager-Q8 sequences | 4/4 | 4/4 | 0 mismatches |

All four per-request TPOT measurements fell by 9.12 to 10.71 ms. Under the
adopted Racebench formula, TPOT at or above 10 ms contributes zero, so those
reductions did not change ERS. The two first-turn TTFT values rose by 0.39 and
4.10 ms in this four-request observation; the second-turn TTFT values fell by
176.33 and 196.84 ms.

# Next measured boundary

Cache pack and unpack dispatches outside each generated schedule still commit
and wait individually. Prompt suffix replay also invokes one full decode
schedule per uncached token, so both submission grouping and replay planning
remain distinct optimization surfaces.
