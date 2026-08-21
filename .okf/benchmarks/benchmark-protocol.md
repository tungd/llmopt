---
type: Benchmark Protocol
title: 'LFM2.5-2.6B PyTorch MPS comparison protocol'
description: 'Record compiler and runtime measurements separately when comparing llmopt with eager PyTorch MPS on the same host.'
tags: [benchmark, lfm2.5, pytorch, mps, apple-silicon]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-20T11:24:21Z' }
sources:
  - id: local-protocol
    resource: /bench/README.md
    title: repository benchmark setup
  - id: result
    resource: /bench/results/lfm25-mps-2026-08-20.json
    title: recorded LFM2.5 MPS measurement
  - id: semantic-result
    resource: /bench/results/lfm25-benchsuite-semantic-5x3-2026-08-20.json
    title: recorded isolated semantic comparison
  - id: 350m-result
    resource: /bench/results/lfm25-350m-racebench-baseline.json
    title: recorded LFM2.5-350M engine-pass and baseline result
  - id: 2.6b-result
    resource: /bench/results/lfm25-racebench-baseline.json
    title: recorded LFM2.5-2.6B engine-pass and baseline result
  - id: racebench-runner
    resource: /bench/racebench/http.py
    title: reference-style concurrent HTTP runner
---

# Scope

Compare eager PyTorch MPS with the OCaml-planned llmopt FX GraphModule on the
same LFM2.5-2.6B checkpoint and Apple Silicon host. Keep graph
capture/planning, model loading, and tensor execution as separately reported
measurements.

# Run record

Record model revision, format or quantization, host chip, macOS and Xcode
versions, Python/PyTorch versions, llmopt revision, prompt corpus, seed, and
warm/cold state.

# Measurements

Record FX capture and OCaml planning time, prompt processing throughput,
time-to-first-token, single-stream decode throughput, selected concurrency
results, peak unified-memory use, model load time, and fixed-input exact output
comparisons against eager PyTorch MPS.

# ERS contract

For each generated request, record TTFT, mean TPOT, completion tokens, success,
and the request score. The request score is zero for failed or zero-token
requests; otherwise it is `0.5 * ttft_score + 0.5 * tpot_score`, with TTFT
floor/ceiling `10/400 ms`, TPOT floor/ceiling `1/10 ms`, and exponent `2`.
ERS is the arithmetic mean over request scores.

# Trace and correctness contract

Keep warmup and scored traces byte-distinct. The suite preserves closed-loop
turn order, pins output-token counts, and writes per-request reports. It also
runs the natural needle matrix at explicit prompt lengths and placements and
records exact response text plus hashes. The local reference probe uses
7,500/9,000/16,000/30,000 tokens and 10/50/90 percent; the short Ninja smoke
uses 128/256 tokens. Generation must route through the selected forward
callable, rather than calling `generate` on a `torch.compile` wrapper directly.

The default local comparison uses a deterministic 5-conversation x 3-turn
profile shaped like the adjacent semantic sample, with 15 requests, 300
completion tokens, and tokenizer-derived long prefixes. Eager and llmopt run
in separate child processes; generated token IDs are compared exactly and the
fixed forward is compared by tensor digest across processes. A single isolated
execution is still recorded as an observation, not as a repeated speed claim.

The local implementation now carries the adjacent runner's trace, request,
warmup, streaming, and score contracts. The in-process MPS adapter serializes
active conversations because the host aborted a concurrent Metal command
encoder; this is recorded as a target adaptation rather than silently treated
as equivalent serving concurrency. The full-shape profile is 70 conversations
x 6 turns = 420 requests with the adjacent seeded 70-rps initial-arrival
sequence.

The first racebench-aligned serialized-MPS launch was stopped after live
system-wide free memory fell to 23%; the host recovered to 67% after clean
termination. No baseline result was written, so the baseline field remains
pending a safe measurement window. The protocol records measurements for
comparison. It introduces no unstated performance threshold or release
decision.

A later launch loaded weights but did not reach inference because PyTorch
rejected the configured high watermark `0.8` alongside its default low
watermark `1.4`. The Ninja rule now sets low watermark `0.7`; that attempt
also produced no request or score.

The separate `bench-suite-350m` target then completed the same semantic 5x3
protocol with 15/15 successful warmup and scored requests for both candidates,
`engine_pass: true`, exact token/digest parity, and eager baseline ERS
`0.0003597708408867709`. Needle retrieval was `0/6` for both candidates and
was recorded separately from engine pass. This is evidence for the 350M
probe only; the 2.6B comparison is recorded separately below.

# Current 2.6B result

The corrected authoritative target completed with `ninja -f ninja.build
bench-suite` and wrote `bench/results/lfm25-racebench-baseline.json`. The
result has `engine_pass: true`, exit code `0`, 15/15 successful warmup and
scored requests for eager and llmopt, eager baseline ERS `0.0`, exact generated
token-ID parity, and exact fixed-forward tensor-digest parity. Needle retrieval
was `0/6` for each candidate and is not part of the engine-pass status. Since
each candidate ran once in an isolated process, the result marks relative
speed claims invalid and retains the eager ERS as the recorded baseline only.
