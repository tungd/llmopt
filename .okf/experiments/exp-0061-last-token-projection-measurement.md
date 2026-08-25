---
type: Experiment
title: 'Last-token vocabulary projection model measurement'
description: 'One bounded LFM2.5-350M run measures the serving-only final-row LM-head specialization with exact eager-Q8 tokens and radix reuse.'
tags: [experiment, compiler, ocaml, metal, q8, prefill, benchmark, ers, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T09:22:08Z' }
sources:
  - id: implementation
    resource: /lib/serving_schedule.ml
    title: Final-row prefill specialization
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-last-token-projection-measurement-2026-08-25.txt
    title: Exact bounded measurement
  - id: prior
    resource: /bench/results/lfm25-350m-q8-vector-prefill-measurement-2026-08-25.txt
    title: Previous native observation
---

# One bounded 350M attempt

At 44% free system memory with no resident model/native process, one
240-second supervised attempt loaded the existing ABI-v11 package and ran four
serial warmup plus four serial scored requests. The Q8 runtime used 512 token
slots and 128 recurrent checkpoints. Memory was 39% free after warmup and 42%
after exit; the sampled server RSS was 103,520 KiB and port 18088 was released.

# Correctness and cache behavior

All eight requests completed with pinned four-token outputs. The four scored
sequences match trace expectations and the separate eager-Q8 observation
exactly. Scored second turns reuse 42/61 and 38/59 prompt tokens, preserving
80/194 total radix reuse.

# Measurement

Scored ERS is `0.3588470515801844`, median TTFT is
`79.15747948572971 ms`, and median TPOT is `8.299840342563886 ms`.
Against the preceding native observation, ERS changes by
`+0.02110547841155419`, median TTFT by `-13.998041511513293 ms`, and median
TPOT by `+0.3516600021005907 ms`.

The two uncached first-turn TTFT values change by `-28.874541050754488 ms` and
`-26.257833058480173 ms`. The two cached second-turn values change by
`+0.8784580277279019 ms` and `+2.180499956011772 ms`.

The separate eager-Q8 observation remains ERS `0.36872784102635947`, median
TTFT `62.557083496358246 ms`, and median TPOT `44.406860998909295 ms`.

# Attribution boundary

The captured package, Metal libraries, Q8 policy, capacities, and traces are
unchanged; the OCaml prefill schedule specialization is the changed compiler
boundary. Current, prior, and eager reports are non-interleaved single
observations, so the measured deltas are recorded without assigning all timing
variance to the pass. No needle workload or new eager model process ran.
