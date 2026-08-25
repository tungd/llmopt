---
type: Experiment
title: 'Native cache-submission batching'
description: 'Group physical KV and recurrent cache conversions into ordered command buffers around each generated schedule and record exact cache, token, and ERS evidence.'
tags: [experiment, ocaml, metal, q8, kv-cache, batching, serving, ers, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T03:43:49Z' }
sources:
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Typed cache command-buffer batches
  - id: engine
    resource: /lib/serving_engine.ml
    title: Batched prefill and decode cache phases
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-native-cache-batching-2026-08-25.txt
    title: Exact cache and matched HTTP observation
---

# Optimization boundary

Each LFM2.5-350M decode has six attention layers and ten recurrent layers. The
runtime previously committed and waited after each of 22 unpack kernels and 22
pack kernels even though each phase is ordered and independent of the CPU
until the phase completes.

# Implementation

`Metal_runtime.Cache.with_batch` owns one exception-safe Metal batch, retains
temporary slot buffers through completion, commits once when at least one
cache kernel was encoded, and aborts on a typed or exceptional failure. The
serving engine groups all unpack operations before the generated schedule and
all pack operations after it. A decode changes from 45 synchronous submissions
to three; prefill changes from 23 to two. The package ABI, tensor archive, KV
formats, and radix ownership do not change.

# Correctness evidence

At 44% free memory with no model process, one device probe encoded six
Q8-group-64 cache conversions in one command buffer and six FP16 conversions
in a second. Attention key/value and recurrent checkpoint bytes all round-trip
exactly.

The one matched model attempt completed 4/4 requests, retained the same four
eager-Q8 token sequences, and retained 80/194 cached prompt tokens.

# Matched observation

| Metric | Decode GEMV | Cache batching | Batched minus GEMV |
|---|---:|---:|---:|
| ERS | 0.10860341576307225 | 0.11381808711306604 | +0.005214671349993788 |
| Median TTFT | 1007.2229375073221 ms | 1014.6957290126011 ms | +7.4727915052790195 ms |
| Median TPOT | 96.3275695006208 ms | 91.1463333274393 ms | -5.181236173181503 ms |
| Cached prompt tokens | 80/194 | 80/194 | 0 |
| Exact eager-Q8 sequences | 4/4 | 4/4 | 0 mismatches |

All four TPOT values fall by 2.82 to 5.45 ms. Both score-contributing
first-turn TTFT values fall by 10.66 and 1.87 ms; the two second-turn changes
are +50.64 and -65.76 ms and remain above the 400 ms zero-score ceiling.

# Next measured boundary

Prompt suffix replay still runs one complete decode transaction per uncached
token. A replay plan that keeps the matched cache state on device across the
entire suffix is now the dominant explicit serving boundary before deeper
kernel fusion.
