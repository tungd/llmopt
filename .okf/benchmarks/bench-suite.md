---
type: Benchmark Suite
title: 'LFM2.5 MPS ERS benchsuite'
description: 'Racebench-compatible local MPS runner with the reference HTTP contract, shape-matched warmup, full workload shape, and natural needle-in-a-haystack validation.'
tags: [benchmark, ERS, trace, needle, lfm2.5, mps]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-20T12:20:00Z' }
sources:
  - id: score
    resource: /bench/racebench/score.py
    title: local ERS score functions
  - id: trace
    resource: /bench/racebench/trace.py
    title: local trace and warmup contract
  - id: profile
    resource: /bench/racebench/profiles.py
    title: deterministic semantic 5x3 workload profile
  - id: suite
    resource: /bench/lfm25_benchsuite.py
    title: isolated MPS benchsuite and target adapter
  - id: http
    resource: /bench/racebench/http.py
    title: adjacent-compatible streaming HTTP runner
  - id: cli
    resource: /bench/racebench/cli.py
    title: reference-style trace and score CLI
  - id: needle
    resource: /bench/racebench/needle.py
    title: natural needle prompt construction
  - id: result
    resource: /bench/results/lfm25-benchsuite-semantic-5x3-2026-08-20.json
    title: isolated semantic 5x3 comparison result
  - id: model-350m
    resource: https://huggingface.co/LiquidAI/LFM2.5-350M
    title: official LFM2.5-350M instruction checkpoint
  - id: result-350m
    resource: /bench/results/lfm25-350m-racebench-baseline.json
    title: persisted 350M engine-pass and baseline result
  - id: result-26b
    resource: /bench/results/lfm25-racebench-baseline.json
    title: persisted 2.6B engine-pass and baseline result
---

# Scope

The suite compares eager PyTorch MPS with the OCaml-planned llmopt backend on
the same LFM2.5-2.6B model. It reports compiler/runtime work separately and
keeps a warmup trace distinct from the scored trace. The default comparison
profile follows the adjacent 5-conversation x 3-turn semantic shape: 15
requests, 300 pinned output tokens, and roughly 1,000-token shared and
conversation prefixes. `official-shape-70x6` exposes the full 70-conversation
x 6-turn / 420-request shape with the adjacent seeded arrival process. Local
prompt content is deterministic and target-specific because the adjacent
trace is generated for LFM2.5-1.2B; the execution shape and scoring contract
are retained for LFM2.5-2.6B.

The comparison command launches one child process per candidate. This removes
the shared in-process MPS and compilation state. The MPS adapter serializes
active conversations by default because concurrent generation hit a Metal
command-buffer assertion on this host; the reference HTTP adapter retains the
adjacent runner's concurrent-conversation/serial-turn behavior.

The default `bench-suite` target remains the LFM2.5-2.6B comparison. The
separate `bench-suite-350m` target uses the official instruction-tuned
`LiquidAI/LFM2.5-350M` checkpoint, writes distinct artifacts and a distinct
baseline record, and is a smaller-model probe rather than a replacement for
the 2.6B result.

# Scoring

Each request records TTFT, TPOT, completion-token count, success, and the
derived request score. The score uses the reference constants: TTFT 10/400 ms,
TPOT 1/10 ms, squared clamped components, and equal TTFT/TPOT weighting. ERS
is the arithmetic mean of request scores, including failed requests as zero.

# Correctness

The suite runs a fixed-input exact-logit comparison and a natural
needle-in-a-haystack matrix. The needle matrix constructs exact tokenizer-sized
prompts with the control record at 10, 50, and 90 percent of the archive, then
expects the exact `RAVEN-4271` response. Needle output is recorded separately
from the engine exit status unless `--require-needle` is supplied.

# Commands

```sh
ninja -f ninja.build bench-suite
ninja -f ninja.build bench-suite-350m
ninja -f ninja.build bench-smoke
PYTHONPATH=python:bench python3.13 bench/lfm25_benchsuite.py \
  --needle-lengths 7500,9000,16000,30000 \
  --needle-positions 10,50,90
```

The Ninja comparison profile uses 2,048/4,096-token needle prompts at
10/50/90 percent. The full natural matrix is available through the CLI. The
short smoke keeps the old 2 x 2 trace and 128/256-token needle prompts.
Per-candidate `warmup.json` and `report.json` files are written under
`_artifacts/lfm25-benchsuite-racebench/`. A completed Ninja run additionally
writes `bench/results/lfm25-racebench-baseline.json`; no such record was
written by the earlier interrupted MPS probe on 2026-08-20. The later
authoritative 2.6B run did write it: `engine_pass: true`, exit code `0`, 15/15
successful warmup and scored requests per candidate, eager baseline ERS
`0.0`, exact generated-token parity, and exact fixed-forward digests. Its
needle matrix was `0/6` for both candidates and remains a separate validation
observation. The one-run relative speed comparison is explicitly invalid.

The 350M target writes under `_artifacts/lfm25-benchsuite-350m-racebench-safe/`
and records `bench/results/lfm25-350m-racebench-baseline.json`.

The recorded 350M result has `engine_pass: true`, eager baseline ERS
`0.0003597708408867709` over 15 scored requests, exact warmup/scored token
parity, exact fixed-forward digests, and `0/6` needle retrieval for each
candidate. Its one-run candidate comparison is retained as an observation;
the result explicitly marks relative speed claims invalid without repeated or
counterbalanced samples.
