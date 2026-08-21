---
type: Experiment
title: 'LFM2.5-350M smaller-model benchsuite probe'
description: 'Run the adopted Viettel AI Race benchsuite against the official smaller instruction checkpoint without conflating it with the 2.6B target baseline.'
tags: [experiment, benchmark, racebench, ERS, mps, lfm2.5, 350m]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-20T15:00:00Z' }
sources:
  - id: ninja-target
    resource: /ninja.build
    title: separate bench-suite-350m target
  - id: local-runner
    resource: /bench/lfm25_benchsuite.py
    title: local MPS adapter and result coordinator
  - id: model
    resource: https://huggingface.co/LiquidAI/LFM2.5-350M
    title: official instruction checkpoint
  - id: score
    resource: /Users/tung/Projects/tungd/viettel-ai-race/src/racebench/score.py
    title: authoritative ERS implementation
  - id: result
    resource: /bench/results/lfm25-350m-racebench-baseline.json
    title: persisted 350M benchsuite result and baseline
---

# Rationale

The 2.6B probe drove unified-memory free percentage from 69% to 36% before
the first request completed. The local Hugging Face cache did not contain a
350M checkpoint, so the official `LiquidAI/LFM2.5-350M` snapshot was downloaded
and is now addressed by an explicit Ninja target.

# Protocol

```sh
ninja -f ninja.build bench-suite-350m
```

The target retains the semantic 5x3 shape, shape-matched byte-distinct
warmup, serialized MPS execution, exact token-output comparison, and 2,048 /
4,096-token needle matrix at 10 / 50 / 90 percent placement. Its result is
written to `_artifacts/lfm25-benchsuite-350m-racebench-safe/result.json` and
the compact baseline record to
`bench/results/lfm25-350m-racebench-baseline.json`.

# Result

The single isolated run completed with Ninja exit code `0` at
`2026-08-20T13:48:07Z`.

* `engine_pass`: `true`; both candidates completed 15/15 warmup and 15/15
  scored requests with no request failures.
* Eager baseline ERS: `0.0003597708408867709` across 15 scored requests.
* llmopt ERS: `0.0` across 15 scored requests.
* Warmup and scored generated token IDs matched exactly for all 15 compared
  requests; the fixed-forward tensor digests were also exact.
* Needle retrieval was `0/6` for each candidate (`RAVEN-4271Lottery` was
  generated instead of the exact expected response); needle validation was
  recorded separately and was not required for the engine pass.
* The recorded MPS values were 708,969,984 allocated bytes and 1,091,010,560
  driver-allocated bytes, with watermark ratios `0.8` / `0.7`.

This proves the smaller-model benchsuite path and its eager baseline record.
It remains separate from the LFM2.5-2.6B target; the authoritative 2.6B
result is now recorded in [exp-0005](exp-0005-viettel-racebench-implementation.md).
