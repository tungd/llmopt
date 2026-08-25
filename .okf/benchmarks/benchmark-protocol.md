---
type: Benchmark Protocol
title: 'LFM2.5-350M PyTorch MPS comparison protocol'
description: 'Record compiler and runtime measurements separately when comparing llmopt with eager PyTorch MPS on the same host.'
tags: [benchmark, lfm2.5, pytorch, mps, apple-silicon]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T09:57:58Z' }
sources:
  - id: local-protocol
    resource: /bench/README.md
    title: repository benchmark setup
  - id: 350m-q8-result
    resource: /bench/results/lfm25-350m-q8-racebench-baseline.json
    title: recorded LFM2.5-350M Q8 engine-pass and baseline result
  - id: 350m-fp16-result
    resource: /bench/results/lfm25-350m-racebench-baseline.json
    title: recorded LFM2.5-350M FP16 engine-pass and baseline result
  - id: racebench-runner
    resource: /bench/racebench/http.py
    title: reference-style concurrent HTTP runner
  - id: current-native-result
    resource: /bench/results/lfm25-350m-q8-paired-simd-measurement-2026-08-25.txt
    title: Current native OCaml paired full-Q8 observation
---

# Scope

Compare eager PyTorch MPS, the OCaml-planned llmopt FX GraphModule, and the
native OCaml generated-Metal server on the same LFM2.5-350M checkpoint and
Apple Silicon host. Keep graph capture/planning, model loading, HTTP edge, and
tensor execution as separately reported measurements.

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

The local implementation carries the adjacent runner's trace, request,
warmup, streaming, and score contracts. The in-process MPS adapter serializes
active conversations because the host aborted a concurrent Metal command
encoder; this is recorded as a target adaptation rather than silently treated
as equivalent serving concurrency. The full-shape profile is 70 conversations
x 6 turns = 420 requests with the adjacent seeded 70-rps initial-arrival
sequence.

# Current 350M result

The authoritative target completed with `ninja -f ninja.build bench-suite`
and wrote `bench/results/lfm25-350m-q8-racebench-baseline.json`. The
result has `engine_pass: true`, exit code `0`, 15/15 successful warmup and
scored requests for eager and llmopt, exact generated token-ID parity, and
exact fixed-forward tensor-digest parity. The historical whole-string field is
`0/6`, but the saved outputs establish 6/6 control-code retrieval and 0/6
exact-only formatting for each candidate; needle retrieval is not part of the
engine-pass status. Since each candidate ran
once in an isolated process, the result marks relative speed claims invalid
and retains the eager ERS as the recorded baseline only.

# Native HTTP smoke result

The first warmed serial native endpoint observation completed 4/4 scored
requests, reported 80/194 cached prompt tokens, and matched every eager-Q8
token sequence and digest. Native/eager ERS was
`0.06169548638841863/0.36872784102635947`, median TTFT was
`1812.1075005328748/62.557083496358246` ms, and median TPOT was
`177.81014566814218/44.406860998909295` ms.

The native SSE extension reports every generated token ID, including
empty-text special tokens, so token timing does not collapse to visible text
events. JSON is used only for the external HTTP/SSE compatibility boundary and
generated reports; native model data remains binary or typed. The exact
per-request record is
[`/bench/results/lfm25-350m-q8-native-http-2026-08-24.txt`](/bench/results/lfm25-350m-q8-native-http-2026-08-24.txt).

Batching each generated schedule into one ordered Metal command buffer retains
4/4 exact token parity and 80/194 cached prompt tokens. The identical warmed
serial native trace changes ERS from `0.06169548638841863` to
`0.11058587181748172`, median TTFT by `-716.9136460288428 ms`, and median TPOT
by `-71.56679149678288 ms`. The matched record is
[`/bench/results/lfm25-350m-q8-native-batched-command-2026-08-24.txt`](/bench/results/lfm25-350m-q8-native-batched-command-2026-08-24.txt).

Selecting the one-row Q8 GEMV entry preserves 4/4 exact token parity and
80/194 cached prompt tokens. Relative to the batched tiled-Q8 trace, median
TTFT changes by `-87.97091699671 ms`, median TPOT by
`-9.915784670738503 ms`, and ERS by `-0.0019824560544094705`. The four TPOT
values all improve but remain at or above the formula's 10 ms zero-score
ceiling; the exact record is
[`/bench/results/lfm25-350m-q8-native-gemv-2026-08-24.txt`](/bench/results/lfm25-350m-q8-native-gemv-2026-08-24.txt).

Grouping each physical-cache unpack and pack phase retains 4/4 exact token
parity and 80/194 cached prompt tokens. Relative to the GEMV trace, median TTFT
changes by `+7.4727915052790195 ms`, median TPOT by
`-5.181236173181503 ms`, and ERS by `+0.005214671349993788`. The exact record
is
[`/bench/results/lfm25-350m-q8-native-cache-batching-2026-08-25.txt`](/bench/results/lfm25-350m-q8-native-cache-batching-2026-08-25.txt).

# Current native paired full-Q8 observation

The current bounded native trace uses the paired-channel package with all 93
linears, including `lm_head`, quantized to Q8. It completes 4/4 warmup and 4/4
scored requests, matches all established full-Q8 eager token IDs, and reuses
80/194 prompt tokens. ERS is `0.40701575836615456`, median TTFT is
`75.22493749274872 ms`, and median TPOT is `6.936652838097264 ms`. Relative to
the preceding single-channel full-Q8 observation, those values change by
`+0.016119526259391448`, `+3.7482914922293276 ms`, and
`-0.37391649675555616 ms`. The reports are separate non-interleaved single
observations. The exact record is
[`/bench/results/lfm25-350m-q8-paired-simd-measurement-2026-08-25.txt`](/bench/results/lfm25-350m-q8-paired-simd-measurement-2026-08-25.txt).

# Native long-context retrieval

The native endpoint completed the six 2,048/4,096-token natural prompts with
6/6 retrieval and exact answer text under normal EOS stopping. Its seven IDs
match the first seven eager-Q8 IDs in every case. The existing benchmark forces
12 outputs through EOS; the native runner now does the same by default. The
historical normal-EOS observation is recorded in
[`/bench/results/lfm25-350m-q8-native-needle-stop-eos-2026-08-24.txt`](/bench/results/lfm25-350m-q8-native-needle-stop-eos-2026-08-24.txt).

The current paired full-Q8 matrix completes 6/6 requests, retrieves the code
in 6/6, and matches all 12 established eager-Q8 IDs in every case. Exact-only
text is 0/6 because the fixed continuation is `RAVEN-4271Lottery`. Median TTFT/TPOT is
`1160.473/33.729 ms` for 2,048-token prompts and `2701.152/62.192 ms` for
4,096-token prompts. The observation is recorded in
[`/bench/results/lfm25-350m-q8-paired-simd-measurement-2026-08-25.txt`](/bench/results/lfm25-350m-q8-paired-simd-measurement-2026-08-25.txt).
