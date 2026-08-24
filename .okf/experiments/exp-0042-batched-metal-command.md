---
type: Experiment
title: 'Batched native Metal schedule submission'
description: 'Replace one synchronous Metal command buffer per generated kernel with one ordered command buffer per execution schedule and record the exact-parity ERS delta.'
tags: [experiment, ocaml, metal, serving, command-buffer, optimization, q8, ers, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-24T02:10:21Z' }
sources:
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Batched native schedule interpreter
  - id: metal-bindings
    resource: /native/ocaml_metal_stubs.m
    title: Ordered compute and blit command-buffer bindings
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-native-batched-command-2026-08-24.txt
    title: Matched before and after native HTTP observation
---

# Measured boundary

Before this change, every generated kernel dispatch created a Metal command
buffer and compute encoder, committed it, entered an OCaml blocking section,
and waited for completion. A decode schedule dispatches roughly 544 kernels,
so command submission inserted a CPU/GPU synchronization boundary after every
operation.

# Implementation

One `execute_schedule` call now owns an abstract batch handle. Generic and Q8
kernels change pipelines and buffers inside an ordered compute encoder. Typed
copies end compute encoding, append a blit encoder to the same command buffer,
and resume compute encoding when needed. Exact source/destination aliases are
no-ops; overlapping views are rejected. The command buffer is committed and
waited once after every schedule command has been encoded, or aborted if typed
interpretation returns an error.

Cache pack/unpack operations surrounding the generated schedule still use
their existing individual dispatch path, so this slice changes schedule
submission without changing radix/KV ownership or package ABI.

# Correctness evidence

At 60% free memory with no model process, the fixed primitive device package
ran 129 commands and 38 kernels from one 9,728-byte workspace. All 39 outputs
remained exact. The model observation then completed 4/4 scored requests with
the same 80/194 cached prompt tokens and the same four eager-Q8 output
sequences as the pre-change run.

# Matched observation

Both native measurements use one byte-distinct four-request warmup, one
four-request scored trace, serial scheduling, Q8 weights, Q8-group-64 KV, and
the same token/checkpoint capacities.

| Metric | Before | Batched | Batched minus before |
|---|---:|---:|---:|
| ERS | 0.06169548638841863 | 0.11058587181748172 | +0.04889038542906309 |
| Median TTFT | 1812.1075005328748 ms | 1095.193854504032 ms | -716.9136460288428 ms |
| Median TPOT | 177.81014566814218 ms | 106.2433541713593 ms | -71.56679149678288 ms |
| Cached prompt tokens | 80/194 | 80/194 | 0 |
| Exact eager-Q8 sequences | 4/4 | 4/4 | 0 mismatches |

The separate eager-Q8 observation remains ERS `0.36872784102635947`, median
TTFT `62.557083496358246 ms`, and median TPOT `44.406860998909295 ms`.

# Next measured boundaries

Second-turn TTFT remains 2,035.819 and 2,147.693 ms because prompt preparation
replays the uncached suffix through serial one-token decode. Short-prompt TPOT
remains 84.333-125.291 ms, leaving cache-conversion submissions and generated
Q8/decode kernels as separate optimization surfaces.
