---
type: Experiment
title: 'Native fixed-12-token long-context needle parity'
description: 'Run the corrected 2,048/4,096-token needle matrix through the optimized OCaml Metal server and compare every generated token with eager Q8.'
tags: [experiment, ocaml, metal, serving, needle, long-context, q8, parity, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T09:01:53Z' }
sources:
  - id: runner
    resource: /bench/lfm25_http_needle.py
    title: Fixed-output native HTTP needle runner
  - id: result
    resource: /bench/results/lfm25-350m-q8-native-needle-fixed12-2026-08-25.txt
    title: Bounded fixed-output matrix
  - id: prior
    resource: /bench/results/lfm25-350m-q8-native-needle-stop-eos-2026-08-24.txt
    title: Earlier normal-EOS native matrix
  - id: eager-reference
    resource: /bench/results/lfm25-350m-q8-racebench-baseline.json
    title: Eager-Q8 fixed-output IDs
---

# Probe

At 51% free memory with no resident model/native process, one supervised
LFM2.5-350M Q8 server-and-runner attempt exercised exact 2,048/4,096-token
prompts at 10/50/90 percent needle placement. The request contract pinned 12
tokens through EOS. The optimized ABI-v11 package contained 808/862 commands,
61/59 kernels, and zero opaque operations.

The attempt exited 0 in about 36 seconds. A sample after five requests showed
34% free memory; memory recovered to 48% after both processes exited, the port
was released, and no model process remained resident.

# Correctness

All six requests succeeded and retrieved `RAVEN-4271`. Every native output
matched the complete eager-Q8 sequence
`8832,563,2880,522,31429,526,7,2,1,553,849,18149`. Exact token parity is 6/6.
Exact-only text formatting is 0/6 because fixed generation continues through
the message-end token and decodes as `RAVEN-4271Lottery`.

| Prompt length | Retrieval | Token parity | Exact text | Median TTFT | Median TPOT | Median latency |
|---|---:|---:|---:|---:|---:|---:|
| 2,048 | 3/3 | 3/3 | 0/3 | 2,859.034 ms | 34.252 ms | 3,259.540 ms |
| 4,096 | 3/3 | 3/3 | 0/3 | 6,540.756 ms | 65.629 ms | 7,255.438 ms |

# Comparison boundary

The earlier normal-EOS matrix emitted seven tokens and used an older package.
Its median TTFT/TPOT was 15,768.789/3,932.246 ms at 2,048 tokens and
52,162.024/8,007.024 ms at 4,096 tokens. The current values are observations
under a different output policy and package stack, so the deltas are not
assigned to one compiler pass. This matrix produces no ERS score and its six
distinct prompts report zero cached prompt tokens.
