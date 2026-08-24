---
type: Experiment
title: 'Native long-context needle retrieval with EOS boundary'
description: 'Run the 2,048/4,096-token natural needle matrix through the native OCaml server, record exact retrieval and long-context latency, and distinguish normal EOS stopping from the benchmark fixed-output contract.'
tags: [experiment, ocaml, metal, serving, needle, long-context, q8, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-24T02:01:26Z' }
sources:
  - id: runner
    resource: /bench/lfm25_http_needle.py
    title: Native HTTP natural-needle runner
  - id: prompt
    resource: /bench/racebench/needle.py
    title: Exact-length natural archive prompt and grader
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-native-needle-stop-eos-2026-08-24.txt
    title: Native long-context retrieval and latency record
  - id: eager-reference
    resource: /bench/results/lfm25-350m-q8-racebench-baseline.json
    title: Existing eager-Q8 fixed-output needle IDs
---

# Probe

The HTTP needle runner reuses the existing deterministic archive corpus,
tokenizer-derived exact-length fitter, 10/50/90 percent placement matrix, and
retrieval-versus-format grader. It sends each prompt to the persistent native
OCaml endpoint and records token IDs, SHA-256, prompt/cache counts, TTFT, TPOT,
latency, and text.

Before launch, macOS reported 59% free unified memory and no resident model
process. The one supervised process used a 32,768-token Q8 KV pool, 512 Q8
recurrent checkpoints, and the previously planned 759,300,096-byte 4,096-token
workspace. Memory recovered to 56% after the process exited.

# Observation

All six exact 2,048/4,096-token prompts returned exactly `RAVEN-4271` across
all placements. Every output had IDs
`8832,563,2880,522,31429,526,7` and the same digest. Those IDs exactly match
the first seven IDs from every existing eager-Q8 needle result; token `7` is
the LFM message-end token.

| Prompt length | Retrieval | Exact text | Median TTFT | Median TPOT | Median latency |
|---|---:|---:|---:|---:|---:|
| 2,048 | 3/3 | 3/3 | 15,768.789 ms | 3,932.246 ms | 39,362.278 ms |
| 4,096 | 3/3 | 3/3 | 52,162.024 ms | 8,007.024 ms | 100,204.188 ms |

# Contract correction

This run used normal EOS termination. The existing in-process needle benchmark
pins 12 output tokens and disables EOS stopping, so eager continues after token
`7` with `2,1,553,849,18149`, producing `Lottery` after the retrieved code.
The long observation therefore proves 6/6 retrieval and seven-token eager
prefix parity, but it is not a 12-token parity comparison.

After the observation, `lfm25_http_needle.py` was corrected to pin
`min_tokens=max_tokens` and send `ignore_eos=true` by default; `--allow-eos`
selects the behavior measured here. The corrected 12-token long matrix was not
launched because the probe policy permits one attempt only.
