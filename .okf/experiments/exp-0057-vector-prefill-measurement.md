---
type: Experiment
title: 'Vector-staged Q8 prefill model measurement'
description: 'One memory-bounded LFM2.5-350M run exercises the 64-wide prefill stage with exact eager-Q8 tokens, unchanged radix reuse, and a new ERS observation.'
tags: [experiment, compiler, ocaml, metal, q8, gemm, prefill, benchmark, ers, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T08:40:41Z' }
sources:
  - id: package
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi11-vector-prefill-v1-2026-08-25
    title: Vector-prefill ABI-v11 package pair
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-vector-prefill-measurement-2026-08-25.txt
    title: Exact bounded measurement
  - id: prior
    resource: /bench/results/lfm25-350m-q8-packed-simd-gemv-measurement-2026-08-25.txt
    title: Previous packed-decode native observation
---

# One bounded 350M attempt

At 47% free system memory with no resident model or native-server process, one
240-second supervised attempt loaded the vector-prefill ABI-v11 pair and ran
four serial warmup plus four serial scored requests. The Q8 runtime used 512
token slots and 128 recurrent checkpoints. It exited cleanly with 49% memory
free and released port 18086.

# Correctness and cache behavior

All eight requests completed with pinned four-token outputs. The four scored
sequences match both trace expectations and the separately recorded eager-Q8
sequences exactly. Scored second turns reuse 42/61 and 38/59 prompt tokens,
preserving 80/194 total radix reuse.

# Measurement

Scored ERS is `0.3377415731686302`, median TTFT is `93.155520997243 ms`, and
median TPOT is `7.948180340463296 ms`. All scored TPOT values are between
`7.3890276641274495` and `8.616208331659436 ms`.

Against the preceding packed-decode native observation, ERS changes by
`+0.012371485882368694`, median TTFT by `-2.44574953103438 ms`, and median
TPOT by `+0.015215007200215958 ms`. The separately recorded eager-Q8
observation remains ERS `0.36872784102635947`, median TTFT
`62.557083496358246 ms`, and median TPOT `44.406860998909295 ms`.

# Attribution boundary

The current and preceding packages retain the same 808/862-command schedule,
61/59 entries, 241-tensor archive, Q8 cache policy, and request traces. The
64-wide vector-staged prefill kernel is the changed Metal boundary. The reports
are single, non-interleaved observations and per-request deltas are mixed, so
the aggregate difference is not assigned wholly to the kernel change. No
needle or exact-logit request ran.
