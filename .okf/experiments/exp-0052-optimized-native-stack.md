---
type: Experiment
title: 'Optimized native stack measurement'
description: 'One memory-bounded 350M run validates corrected suffix replay and the fused/SIMD Metal stack with exact eager-Q8 token parity and a new ERS observation.'
tags: [experiment, compiler, ocaml, metal, q8, simdgroup, radix-cache, benchmark, ers, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T07:32:19Z' }
sources:
  - id: package
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi11-simd-attention-v1-2026-08-25
    title: Optimized ABI-v11 package pair
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-native-optimized-stack-2026-08-25.txt
    title: Exact aggregate measurement
  - id: prior
    resource: /bench/results/lfm25-350m-q8-native-cache-batching-2026-08-25.txt
    title: Previous native observation
---

# One bounded attempt

At 52% free system memory with no resident Torch, LFM, or native-server
process, one 240-second supervised native-Q8 attempt ran four serial warmup and
four serial scored requests. The server used 512 token slots, 128 recurrent
checkpoints, and the current 808/862-command ABI-v11 package pair. It exited
cleanly, released its port, and left 50% memory free with no model process.

# Correctness and cache behavior

All eight requests succeeded. Warmup and scored token IDs each match the
separately saved eager-Q8 reports 4/4. The scored trace retains the exact prior
cache accounting: 42/61 and 38/59 second-turn tokens, or 80/194 overall. This
is the first model evidence for the corrected dependent cached-suffix batch and
for the combined graph-fusion/SIMD kernel stack.

# Aggregate measurement

Native ERS is `0.23655514122115978`, median TTFT is
`136.7437920125667 ms`, and median TPOT is `14.803701342316344 ms`. Against the
earlier native cache-batched observation, ERS changes by
`+0.12273705410809374`, median TTFT by `-877.9519370000344 ms`, and median TPOT
by `-76.34263198512295 ms` with unchanged token IDs and cache counts.

The separately recorded eager-Q8 observation remains ERS
`0.36872784102635947`, median TTFT `62.557083496358246 ms`, and median TPOT
`44.406860998909295 ms`. Thus this one native observation has lower median
TPOT, higher median TTFT, and lower ERS than that earlier eager trace.

# Attribution boundary

The measured binary includes corrected suffix replay, three Q8 graph fusions,
SIMD-group Q8 GEMV, SIMD-group RMSNorm, and single-pass online-softmax
attention. Because they ran together and the comparison reports are earlier
single observations, the delta is aggregate and cannot be assigned to one
pass. No long-context needle or exact-logit request ran; the optimized stack's
fixed-12-token needle matrix remains open.
