---
type: Experiment
title: 'Paired SIMD-group Q8 model measurement'
description: 'Measure the paired-channel Q8 decode layout on the bounded 350M ERS trace and fixed-output needle matrix.'
tags: [experiment, compiler, ocaml, metal, q8, gemv, benchmark, ers, needle, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T09:57:58Z' }
sources:
  - id: measurement
    resource: /bench/results/lfm25-350m-q8-paired-simd-measurement-2026-08-25.txt
    title: Paired native and needle observations
  - id: compiler
    resource: /bench/results/lfm25-350m-q8-paired-simd-compiler-2026-08-25.txt
    title: Paired SIMD compiler evidence
  - id: previous
    resource: /bench/results/lfm25-350m-q8-lm-head-measurement-2026-08-25.txt
    title: Previous single-channel full-Q8 observations
---

# Short trace

The paired package completes 4/4 warmup and 4/4 scored requests, preserves all
established full-Q8 eager IDs, and reports the same 80/194 radix reuse. ERS is
`0.40701575836615456`, median TTFT is `75.22493749274872 ms`, and median TPOT
is `6.936652838097264 ms`.

Against the preceding single-channel full-Q8 observation, ERS changes by
`+0.016119526259391448`, median TTFT by `+3.7482914922293276 ms`, and median
TPOT by `-0.37391649675555616 ms`. Mean TTFT/TPOT changes by
`-0.8084479923127219/-0.629916665881562 ms`. The four request-level deltas are
mixed, and both reports are separate single observations.

# Long-context matrix

The paired package completes all six 2,048/4,096-token prompts with 6/6
retrieval and 6/6 exact 12-token parity. Exact-only formatting remains 0/6.
Median TTFT/TPOT is `1160.473/33.729 ms` at 2,048 tokens and
`2701.152/62.192 ms` at 4,096 tokens.

Relative to the preceding single-channel matrix, median TTFT changes by
`-3.925/-4.867 ms`, while median TPOT changes by `+0.722/+1.326 ms`. Median
end-to-end latency changes by `+4.023/-1.774 ms`. The six distinct prompts
contain 18,432 total prompt tokens and intentionally produce no cache hits.

# Supervision note

The short runner completed both JSON reports, but the enclosing shell returned
143 because `set -e` propagated the intentionally terminated server's `wait`
status. The server and port were released and no retry occurred. The needle
runner handled that expected server status separately and exited 0. Memory
recovered after both attempts.

# Evidence boundary

No new eager model process was loaded. Token parity uses the established
full-Q8 eager short and fixed-12 references. The observations are not
interleaved repetitions.
