---
type: Experiment
title: 'Pre-integration ERS and needle-grader correction'
description: 'Record the post-radix-implementation benchmark before the OCaml cache enters execution, and separate retrieval from exact response formatting.'
tags: [experiment, ers, needle, q8, mps, radix-cache]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T16:28:23Z' }
sources:
  - id: result
    resource: /_artifacts/lfm25-benchsuite-q8-radix-e3d0d15/result.json
    title: Isolated eager and llmopt result
  - id: suite
    resource: /bench/lfm25_benchsuite.py
    title: Benchmark and needle evaluation path
  - id: grader
    resource: /bench/racebench/needle.py
    title: Corrected needle response evaluator
  - id: cache-commit
    resource: git:e3d0d15
    title: OCaml radix and KV cache commit
---

# Run

After `memory_pressure -Q` reported 58% system-wide free memory, one isolated
LFM2.5-350M Q8 semantic-5x3 comparison ran with generated exact Metal mode,
shape-matched warmup, MPS cache release, and six 2,048/4,096-token needle
prompts. Sampled free memory did not fall below 43% and recovered to 52%.

| Candidate | ERS | Scored requests | Median TTFT | Median TPOT |
|---|---:|---:|---:|---:|
| eager | 0.0 | 15/15 | 541.31 ms | 43.99 ms |
| llmopt | 0.0 | 15/15 | 523.32 ms | 44.29 ms |

Fixed-forward tensor digests were exact and all 15 warmup plus 15 scored token
sequences matched. This was one execution per candidate, so the raw latency
deltas are recorded without a relative speed claim.

# Cache boundary

Both candidate reports recorded `cached_prompt_tokens: 0`. The new OCaml radix
and KV structures were not connected to request execution, so this run is a
pre-integration baseline and does not measure prefix-cache impact.

# Needle correction

Every eager and llmopt needle response was `RAVEN-4271Lottery`. The previous
grader compared the entire normalized output to `RAVEN-4271` and labeled all
rows incorrect. The saved output proves two different observations:

| Observation | eager | llmopt |
|---|---:|---:|
| Control-code retrieval | 6/6 | 6/6 |
| Exact only-the-code response | 0/6 | 0/6 |

The grader now extracts the expected control code without accepting a different
numeric suffix and records exact response formatting separately. This
correction is deterministic over the saved text and does not require another
model launch.
