---
type: Experiment
title: 'Packed SIMD Q8 GEMV model measurement'
description: 'One memory-bounded LFM2.5-350M run exercises packed Q8 decode loads with exact eager-Q8 tokens, unchanged radix reuse, and a new ERS observation.'
tags: [experiment, compiler, ocaml, metal, q8, gemv, simdgroup, benchmark, ers, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T08:22:49Z' }
sources:
  - id: package
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi11-packed-gemv-v1-2026-08-25
    title: Packed ABI-v11 package pair
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-packed-simd-gemv-measurement-2026-08-25.txt
    title: Exact bounded measurement
  - id: prior
    resource: /bench/results/lfm25-350m-q8-native-optimized-stack-2026-08-25.txt
    title: Previous native optimized-stack observation
---

# One bounded 350M attempt

At 52% free system memory with no resident model or native-server process, one
240-second supervised attempt loaded the packed ABI-v11 package pair and ran
four serial warmup plus four serial scored requests. The Q8 runtime used 512
token slots and 128 recurrent checkpoints. It exited cleanly with 52% memory
free and released port 18085.

# Correctness and cache behavior

All eight requests completed with pinned four-token outputs. The four scored
sequences match both the trace expectations and the separately recorded eager
Q8 sequences exactly. Scored second turns reuse 42/61 and 38/59 prompt tokens,
preserving the previous 80/194 total radix reuse.

# Measurement

Scored ERS is `0.3253700872862615`, median TTFT is
`95.60127052827738 ms`, and median TPOT is `7.93296533326308 ms`. All four
scored TPOT observations are between `7.271249991996835` and
`9.664375005134692 ms`.

Against the preceding native optimized-stack observation, ERS changes by
`+0.08881494606510171`, median TTFT by `-41.14252148428932 ms`, and median
TPOT by `-6.870736009053265 ms`. The separately recorded eager-Q8 observation
remains ERS `0.36872784102635947`, median TTFT `62.557083496358246 ms`, and
median TPOT `44.406860998909295 ms`.

# Attribution boundary

The current and preceding native packages retain the same 808/862-command
schedule, 61/59 entries, 241-tensor archive, Q8 cache policy, and request
trace. The generated packed GEMV loop is the changed Metal boundary. Because
the reports are single, non-interleaved observations, the measured delta is
not assigned wholly to that code change. No needle or exact-logit request ran.
