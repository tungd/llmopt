---
type: Benchmark Suite
title: 'LFM2.5 MPS ERS benchsuite'
description: 'Racebench-compatible local MPS runner with the reference HTTP contract, shape-matched warmup, full workload shape, and natural needle-in-a-haystack validation.'
tags: [benchmark, ERS, trace, needle, lfm2.5, mps]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T16:28:23Z' }
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
  - id: native-needle-runner
    resource: /bench/lfm25_http_needle.py
    title: native endpoint natural-needle runner
  - id: cli
    resource: /bench/racebench/cli.py
    title: reference-style trace and score CLI
  - id: needle
    resource: /bench/racebench/needle.py
    title: natural needle prompt construction
  - id: model-350m
    resource: https://huggingface.co/LiquidAI/LFM2.5-350M
    title: official LFM2.5-350M instruction checkpoint
  - id: result-350m-q8
    resource: /bench/results/lfm25-350m-q8-racebench-baseline.json
    title: persisted 350M Q8 engine-pass and baseline result
  - id: result-350m-fp16
    resource: /bench/results/lfm25-350m-racebench-baseline.json
    title: persisted 350M FP16 engine-pass and baseline result
  - id: result-native-http
    resource: /bench/results/lfm25-350m-q8-native-http-2026-08-24.txt
    title: native HTTP token parity and ERS observation
  - id: result-native-needle
    resource: /bench/results/lfm25-350m-q8-native-needle-stop-eos-2026-08-24.txt
    title: native long-context retrieval observation
  - id: result-native-batched
    resource: /bench/results/lfm25-350m-q8-native-batched-command-2026-08-24.txt
    title: native command-buffer optimization result
  - id: result-preintegration
    resource: /_artifacts/lfm25-benchsuite-q8-radix-e3d0d15/result.json
    title: post-cache-implementation pre-integration observation
---

# Scope

The suite compares eager PyTorch MPS with the OCaml-planned llmopt backend on
the LFM2.5-350M model. It reports compiler/runtime work separately and
keeps a warmup trace distinct from the scored trace. The default comparison
profile follows the adjacent 5-conversation x 3-turn semantic shape: 15
requests, 300 pinned output tokens, and roughly 1,000-token shared and
conversation prefixes. `official-shape-70x6` exposes the full 70-conversation
x 6-turn / 420-request shape with the adjacent seeded arrival process. Local
prompt content is deterministic and target-specific; the execution shape and
scoring contract are retained for LFM2.5-350M.

The comparison command launches one child process per candidate. This removes
the shared in-process MPS and compilation state. The MPS adapter serializes
active conversations by default because concurrent generation hit a Metal
command-buffer assertion on this host; the reference HTTP adapter retains the
adjacent runner's concurrent-conversation/serial-turn behavior.

The default `bench-suite` target runs the LFM2.5-350M comparison using the
official instruction-tuned `LiquidAI/LFM2.5-350M` checkpoint and records
its baseline under `bench/results/lfm25-350m-q8-racebench-baseline.json`.

The same HTTP runner now drives the persistent native OCaml server. Native SSE
events carry exact token IDs in addition to decoded text so special tokens and
split UTF-8 scalars retain token-level timing. The first warmed serial smoke
records 4/4 eager-Q8 token parity, 80/194 cached prompt tokens, native ERS
`0.06169548638841863`, and eager ERS `0.36872784102635947`. A native
stop-on-EOS long matrix retrieves 6/6 and matches the first seven eager-Q8 IDs;
the corrected fixed-12-token matrix and semantic 5x3 profile remain separate
open measurements.

Schedule-wide command-buffer batching preserves those four eager-Q8 sequences
and 80/194 cached tokens. On the identical warmed serial trace, native ERS
increases to `0.11058587181748172`, median TTFT decreases by `716.914 ms`, and
median TPOT decreases by `71.567 ms` relative to the initial native endpoint.

# Scoring

Each request records TTFT, TPOT, completion-token count, success, and the
derived request score. The score uses the reference constants: TTFT 10/400 ms,
TPOT 1/10 ms, squared clamped components, and equal TTFT/TPOT weighting. ERS
is the arithmetic mean of request scores, including failed requests as zero.

# Correctness

The suite runs a fixed-input exact-logit comparison and a natural
needle-in-a-haystack matrix. The needle matrix constructs exact tokenizer-sized
prompts with the control record at 10, 50, and 90 percent of the archive, then
checks whether the response retrieves `RAVEN-4271` and separately records
whether the response contains only that code. Needle retrieval is recorded
separately from the engine exit status unless `--require-needle` is supplied.

# Commands

```sh
ninja -f ninja.build bench-suite
ninja -f ninja.build bench-smoke
PYTHONPATH=python:bench python3.13 bench/lfm25_benchsuite.py \
  --needle-lengths 7500,9000,16000,30000 \
  --needle-positions 10,50,90
```

The Ninja comparison profile uses 2,048/4,096-token needle prompts at
10/50/90 percent. The full natural matrix is available through the CLI. The
short smoke keeps the old 2 x 2 trace and 128/256-token needle prompts.
Per-candidate `warmup.json` and `report.json` files are written under
`_artifacts/lfm25-benchsuite-q8-racebench-safe/`.

The saved 350M outputs contain `RAVEN-4271Lottery` in all six positions for
both candidates. The old whole-string grader recorded `0/6`; corrected
semantics are 6/6 control-code retrieval and 0/6 exact-only formatting. The
post-radix-implementation run retains exact token/digest parity but records
zero cached prompt tokens because the OCaml serving cache is not yet connected.
