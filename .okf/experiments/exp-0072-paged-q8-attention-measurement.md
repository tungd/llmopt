---
type: Experiment
title: 'Direct paged-Q8 attention model measurement'
description: 'One bounded LFM2.5-350M trace and one needle matrix preserve exact tokens while recording mixed short and long-context latency deltas.'
tags: [experiment, compiler, ocaml, metal, q8, attention, radix-cache, benchmark, ers, needle, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T10:51:18Z' }
sources:
  - id: implementation
    resource: /bench/results/lfm25-350m-q8-paged-attention-compiler-2026-08-25.txt
    title: Direct paged-Q8 compiler boundary
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-paged-attention-measurement-2026-08-25.txt
    title: Bounded short and long-context measurement
  - id: prior
    resource: /bench/results/lfm25-350m-q8-vector-cache-unpack-measurement-2026-08-25.txt
    title: Previous materialized vector-unpack observation
---

# Bounded 350M execution

One 240-second supervised LFM2.5-350M attempt started at 50% system-wide free
memory with no resident model or native server. It ran four serial warmup and
four serial scored requests through the direct paged-Q8 decode schedule. All
requests completed, all four scored token sequences match the established
full-Q8 eager IDs, and radix reuse remains 80/194 prompt tokens. Memory was 45%
free after warmup and 46% after process exit; the sampled server RSS was 65,456
KiB and port 18097 was released.

Scored ERS is `0.38326789681891504`, median TTFT is
`72.54981249570847 ms`, and median TPOT is `8.034680504351854 ms`. Against the
previous vector-unpack observation, those values change by
`-0.03339199781233493`, `+3.959479508921504 ms`, and
`+1.1348958360031247 ms`.

# Long-context execution

A separate 900-second supervised attempt started at 51% free memory and ran
the six fixed-output 2,048/4,096-token needle requests. Retrieval is 6/6 and
all six 12-token sequences match the eager-Q8 reference exactly. Exact-only
text remains 0/6 because the pinned output is `RAVEN-4271Lottery`.

At 2,048 tokens, median TTFT/TPOT/latency is
`1,239.5158750005066/24.747768909120087/1,509.9365829955786 ms`; the deltas
against vector unpack are `+10.036125022452325/-10.082787908190351/
-99.1467090207152 ms`. At 4,096 tokens, medians are
`2,903.4347910201177/41.48384854620831/3,353.581291041337 ms`, changing by
`-31.565208977553993/-22.55124990849501/-289.8096669232473 ms`.

# Evidence boundary

The short and long reports each contain one non-interleaved observation. Exact
tokens and radix accounting are unchanged. Short-trace TTFT/TPOT are higher
and ERS is lower; long-context TPOT and total latency are lower at both tested
lengths. No eager process ran in either attempt. The model target was
`LiquidAI/LFM2.5-350M`; 2.6B was not loaded.
