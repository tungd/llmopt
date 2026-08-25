---
type: Experiment
title: 'Cached-suffix command-buffer batching attempt'
description: 'Batch dependent decode schedules across a matched prompt suffix while preserving per-token hybrid radix checkpoints; record the failed first model attempt and static correction without inventing a score.'
tags: [experiment, ocaml, metal, q8, radix-cache, kv-cache, batching, serving, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T06:02:21Z' }
sources:
  - id: engine
    resource: /lib/serving_engine.ml
    title: Cached-suffix replay coordinator
  - id: replay-state
    resource: /lib/serving_replay.ml
    title: Canonical dependent-decode buffer state
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Generated-schedule and cache command-buffer encoding
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-native-suffix-batching-attempt-2026-08-25.txt
    title: Failed model attempt and post-correction static evidence
---

# Optimization boundary

The matched scored prompts reuse 42/61 and 38/59 tokens, so their uncached
suffixes contain 19 and 21 tokens. Cache-submission batching still executes
each suffix token as three committed-and-waited transactions: unpack, generated
schedule, and pack. The intended replay performs one prefix unpack followed by
one command buffer containing every dependent schedule and cache write.

# Radix semantics

The replay reserves one attention slot and one recurrent checkpoint for every
suffix token. Immediately after each generated decode schedule, the same Metal
command buffer writes that token's attention slice and recurrent state to its
reserved physical locations. Once the command buffer completes, the logical
cache inserts each successive prefix. Page-size-one branch points therefore
retain the same checkpoint granularity as serial replay.

# First attempt

At 50% free memory with no resident model, Torch, or native-server process, one
240-second supervised Q8 attempt ran the four-request warmup trace. Both first
turns completed and exactly matched the preceding cache-batched warmup. Both
second turns failed with:

```text
runtime input is not bound: l_kwargs_past_key_values_layers_0_conv_states
```

The attempt completed only 2/4 warmup requests, never started the scored trace,
and produced no ERS result. Inspection afterward found 47% free memory and no
resident model or server process.

# Diagnosis and correction

The attention transition reconstructed the next decode input list while the
recurrent buffers lived in a second field, making omission possible.
`Serving_replay.Decode_buffers` now owns both classes in one parametric value;
runtime inputs and checkpoint sources are derived from it. Its regression test
failed on the original omission and passes after recurrent bindings are
retained.

The corrected runtime also permits generated schedules and physical cache
writes to share one ordered Metal command buffer. `ninja -f ninja.build all
test q8-serving-smoke native-schedule-smoke` passes, including all OCaml tests
and 37/37 Python tests.

# Evidence boundary

The fixed model attempt was not rerun. No post-correction token-parity,
cache-accounting, latency, or ERS claim is available. The latest valid native
score remains `0.11381808711306604` from
[cache-submission batching](exp-0044-cache-submission-batching.md).
