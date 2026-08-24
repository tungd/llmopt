---
type: Experiment
title: 'Native OCaml HTTP serving and token-level ERS'
description: 'Expose the persistent Q8 Metal generation engine through the benchmark HTTP/SSE contract, prove cross-turn radix reuse and exact eager token parity, and record corrected token-level latency.'
tags: [experiment, ocaml, metal, serving, http, sse, radix-cache, q8, ers, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-24T01:46:58Z' }
sources:
  - id: server
    resource: /bin/lfm_serve.ml
    title: Persistent native OCaml HTTP server
  - id: protocol
    resource: /lib/openai_protocol.ml
    title: Typed OpenAI-compatible edge protocol
  - id: runner
    resource: /bench/racebench/http.py
    title: Token-instrumented HTTP benchmark runner
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-native-http-2026-08-24.txt
    title: Native and eager Q8 scored observation
---

# Runtime boundary

`llmopt-serve` loads `LLMOPTTK`, the ABI-v8 prefill/decode packages, their
generated metallibs, and the shared 241-tensor Q8 archive once. It owns one
`Generation.t`, so physical Metal KV/checkpoint pools and the compressed radix
tree persist across HTTP requests. Requests are accepted serially to retain one
owner for mutable Metal and cache state without duplicating model memory.

The OpenAI-compatible boundary accepts typed `messages`, greedy
`temperature=0`, a positive `max_tokens`, `min_tokens`, and `ignore_eos`, then
streams content, finish reason, usage, and
`prompt_tokens_details.cached_tokens`. Q8-group-64 KV is the default and FP16
remains selectable through `--kv fp16`.

JSON is confined to this external compatibility edge and generated benchmark
reports. Graph capture, package loading, tensor data, tokenizer state, command
schedules, radix/KV ownership, and Metal dispatch use their binary or typed
representations.

# Token-level streaming correction

One generated token does not necessarily produce visible text: special tokens
decode to an empty string and UTF-8 scalars can span token boundaries. Timing
visible SSE content events therefore produced a false `0 ms` TPOT when four
generated tokens collapsed to one visible event.

The incremental OCaml tokenizer decoder now buffers only incomplete UTF-8
bytes. Every SSE token frame carries `x_llmopt_token_id`, including empty-text
special tokens. The HTTP runner records those token IDs and timestamps while
remaining compatible with ordinary content-only OpenAI endpoints. Reports now
contain exact output-token sequences and their SHA-256 digests.

# Observation

The preflight reported 54% free unified memory, no resident model process, and
an unused server port. One 240-second-supervised process ran the byte-distinct
four-request warmup and then the four-request scored smoke serially. A separate
memory-checked eager-Q8 process used the same traces and fixed four-token
generation contract.

| Metric | Native OCaml Q8 | Eager PyTorch MPS Q8 | Native minus eager |
|---|---:|---:|---:|
| Successful scored requests | 4/4 | 4/4 | 0 |
| ERS | 0.06169548638841863 | 0.36872784102635947 | -0.30703235463794084 |
| Median TTFT | 1812.1075005328748 ms | 62.557083496358246 ms | +1749.5504170365166 ms |
| Median TPOT | 177.81014566814218 ms | 44.406860998909295 ms | +133.4032846692329 ms |
| Cached prompt tokens | 80/194 | 0/194 | +80 |
| Exact token sequences | 4/4 | reference | 0 mismatches |

Native request IDs exactly matched eager:

```text
c0000-t000  1098,5706,803,4481
c0000-t001  41677,7,2,1
c0001-t000  1098,5410,4100,856
c0001-t001  31466,7,2,1
```

Second turns reused 42/61 and 38/59 prompt tokens but measured TTFT at
3407.366 and 3675.869 ms. The current engine replays the uncached historical
assistant terminator and new user suffix through serial one-token decode from
the matched recurrent checkpoint; this measured boundary dominates the next
serving optimization slice.

# Evidence boundary

This is one warmed serial observation per runtime. It proves external request
handling, persistent cross-turn radix/KV reuse, fixed output counts, corrected
token-level timing, and exact eager-Q8 token parity for the smoke trace. It
does not execute the native long-context needle matrix or the semantic 5x3
profile.
