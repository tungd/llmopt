---
type: Experiment
title: 'Vectorized Q8 cache-unpack model measurement'
description: 'Measure vec4 Q8 cache restoration on the bounded LFM2.5-350M trace and long-context needle matrix.'
tags: [experiment, compiler, ocaml, metal, q8, kv-cache, benchmark, ers, needle, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T10:28:14Z' }
sources:
  - id: measurement
    resource: /bench/results/lfm25-350m-q8-vector-cache-unpack-measurement-2026-08-25.txt
    title: Vector Q8 cache-unpack native observations
  - id: compiler
    resource: /bench/results/lfm25-350m-q8-vector-cache-unpack-compiler-2026-08-25.txt
    title: Vector Q8 cache-unpack compiler evidence
  - id: previous-q8
    resource: /bench/results/lfm25-350m-q8-simd-cache-pack-measurement-2026-08-25.txt
    title: Previous SIMD-pack Q8 observation
  - id: previous-needle
    resource: /bench/results/lfm25-350m-q8-paired-simd-measurement-2026-08-25.txt
    title: Previous paired long-context observation
---

# Short trace

The vec4-unpack package completes 4/4 warmup and 4/4 scored requests, preserves
every established full-Q8 eager token sequence, and reports the same 80/194
radix reuse. ERS is `0.41665989463124997`, median TTFT is
`68.59033298678696 ms`, and median TPOT is `6.89978466834873 ms`.

Against the immediately preceding SIMD-pack package, ERS changes by
`+0.014504803224463791`, median TTFT by `-4.54179200460203 ms`, and median
TPOT by `-0.4078611673321575 ms`. Mean TTFT/TPOT changes by
`-1.7378024786012247/-0.2664375788299367 ms`; request deltas are mixed.

# Long-context matrix

The vec4 package completes all six 2,048/4,096-token prompts with 6/6 retrieval
and 6/6 exact 12-token parity. Exact-only formatting remains 0/6. Median
TTFT/TPOT is `1229.480/34.831 ms` at 2,048 and `2935.000/64.035 ms` at 4,096
tokens.

Relative to the preceding paired matrix, median TTFT/TPOT changes by
`+69.007/+1.101 ms` at 2,048 and `+233.848/+1.843 ms` at 4,096 tokens. Median
end-to-end latency changes by `+77.575/+264.551 ms`. The six distinct prompts
contain 18,432 prompt tokens and intentionally produce no cache hits.

# Evidence boundary

The short and long comparisons are separate, non-interleaved single
observations. The short trace observes lower medians, while the long-context
matrix observes higher medians. No eager process ran; parity uses established
full-Q8 eager IDs.
