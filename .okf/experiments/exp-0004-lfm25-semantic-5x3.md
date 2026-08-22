---
type: Experiment
title: 'LFM2.5 isolated semantic 5x3 MPS comparison'
description: 'A shape-matched long-context comparison with 15 requests per candidate, exact token parity, and natural needle validation.'
tags: [experiment, benchmark, ERS, long-context, needle, lfm2.5, mps]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-20T13:09:53Z' }
sources:
  - id: suite
    resource: /bench/lfm25_benchsuite.py
    title: isolated benchsuite implementation
  - id: profile
    resource: /bench/racebench/profiles.py
    title: deterministic semantic 5x3 profile
  - id: result
    resource: /bench/results/lfm25-benchsuite-semantic-5x3-2026-08-20.json
    title: compact recorded result
  - id: artifact
    resource: /_artifacts/lfm25-benchsuite/result.json
    title: full combined report from the local probe
---

# Procedure

```sh
ninja -f ninja.build bench-suite
```

The comparison used the deterministic 5-conversation x 3-turn profile: 15
requests per candidate, 300 pinned completion tokens, about 1,000 rendered
shared-prefix tokens, about 1,000 rendered conversation-prefix tokens, and
127–131 new user tokens per turn. Eager and llmopt ran in separate child
processes. The needle probe used 2,048 and 4,096 prompt tokens at 10/50/90
percent placement.

# Observation

Both candidates completed all 15 warmup and all 15 scored requests. Eager and
llmopt each recorded ERS `0.0`; all 30 scored requests received zero because
long-context TTFT exceeded the adopted 400 ms reference ceiling. The useful
raw scored medians were:

| Candidate | TTFT median | TPOT median | ERS |
|---|---:|---:|---:|
| eager | 3396.53 ms | 34.32 ms | 0.0 |
| llmopt | 3002.44 ms | 33.07 ms | 0.0 |

The llmopt observation was `-394.09 ms` TTFT median and `-1.25 ms` TPOT median
relative to eager, but this is one isolated execution per candidate with no
repeated or counterbalanced samples and is not recorded as a speed claim.

Correctness was exact: all 15 scored generated token sequences and all 15
warmup sequences matched by token ID, and the fixed forward tensor digest
matched at shape `[1, 15, 128000]`. The needle probe returned `0/6` for both
candidates; its failures are recorded separately from the engine pass.

# Limits

This result demonstrates that the adopted racebench ERS has no useful dynamic
range for this long-context LFM2.5 MPS workload. Subsequent compiler work
should use raw prefill/decode latency and exact parity for local iteration,
while retaining ERS as a compatibility field. Repeated/counterbalanced
sampling remains an open measurement design question.
