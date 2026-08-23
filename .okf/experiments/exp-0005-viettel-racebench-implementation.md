---
type: Experiment
title: 'Viettel AI Race benchsuite implementation'
description: 'Adopt the adjacent racebench trace, streaming, warmup, and ERS contracts for the local LFM2.5-350M PyTorch MPS target.'
tags: [experiment, benchmark, racebench, ERS, mps, lfm2.5]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-20T13:30:00Z' }
sources:
  - id: local-suite
    resource: /bench/lfm25_benchsuite.py
    title: local MPS adapter and result coordinator
  - id: local-http
    resource: /bench/racebench/http.py
    title: reference-style streaming HTTP runner
  - id: local-cli
    resource: /bench/racebench/cli.py
    title: trace validation and score CLI
  - id: local-profile
    resource: /bench/racebench/profiles.py
    title: deterministic 5x3 and full 70x6 profiles
  - id: adjacent-score
    resource: /Users/tung/Projects/tungd/viettel-ai-race/src/racebench/score.py
    title: authoritative ERS implementation
  - id: adjacent-benchmark
    resource: /Users/tung/Projects/tungd/viettel-ai-race/src/racebench/benchmark.py
    title: authoritative closed-loop HTTP benchmark
  - id: current-result
    resource: /bench/results/lfm25-350m-q8-racebench-baseline.json
    title: authoritative 350M engine-pass and baseline result
---

# Implementation

The local benchsuite now carries the adjacent runner's pure ERS functions,
shape-matched byte-distinct warmup validation, pinned completion counts,
concurrent-conversation/serial-turn HTTP execution, streaming TTFT/TPOT
measurement, and trace/report CLI surface. The local PyTorch MPS adapter uses
the same request contract and exact token-ID/fixed-forward observations, with a
serialized device execution boundary because concurrent MPS generation aborted
in Metal on this host.

The local target is LFM2.5-350M, while the adjacent contest trace is authored
for LFM2.5-1.2B. The implementation therefore retains the adjacent 5x3 sample
and 70x6/420-request shape and seeded arrivals, but generates deterministic
target-local prompt content rather than claiming byte identity with the
1.2B trace.

# Verification

The following non-device checks passed:

```sh
PYTHONPATH=python:bench python3.13 -m unittest discover -s python/tests -v
PYTHONPATH=python:bench python3.13 -m py_compile bench/lfm25_benchsuite.py bench/racebench/*.py
PYTHONPATH=python:bench python3.13 -m racebench.cli validate-trace \
  /Users/tung/Projects/tungd/viettel-ai-race/traces/official-shape-poisson70-v1.json \
  --warmup-trace /Users/tung/Projects/tungd/viettel-ai-race/traces/official-shape-poisson70-disjoint-warmup-v1.json \
  --require-shape-matched
```

# Current authoritative result

After the safety configuration was corrected (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8`, `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7`), the target completed with:

```sh
ninja -f ninja.build bench-suite
```

The compact result is [bench/results/lfm25-350m-q8-racebench-baseline.json](../../bench/results/lfm25-350m-q8-racebench-baseline.json).
It records `engine_pass: true`, exit code `0`, and 15/15 successful warmup and
scored requests for both eager and llmopt. Warmup and scored generated token IDs match exactly
for all 15 requests, and the fixed-forward output digests are exact across
the two isolated candidate processes.

The raw whole-string grader recorded `0/6`, but every saved response contains
`RAVEN-4271Lottery`: control-code retrieval is 6/6 and exact-only formatting is
0/6 for both candidates. The result keeps needle retrieval independent from engine pass. It
also marks the relative speed comparison invalid because this is one execution
per candidate rather than repeated or counterbalanced sampling; the persisted
baseline is an observation, not an optimization claim.
