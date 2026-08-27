---
type: Experiment
title: 'W4A16 decode RoPE table elision'
description: 'Precompute the LFM2 RoPE cosine and sine rows on engine load, bind the current position directly, and prune the captured scalar position/trigonometric branch from decode.'
tags: [experiment, serving, scheduler, rope, w4a16, kvq8, metal]
status: draft
generated: { by: codex/gpt-5.6, at: '2026-08-27T04:00:00Z' }
sources:
  - id: schedule
    resource: /lib/serving_schedule.ml
    title: decode specialization and scalar RoPE-branch pruning
  - id: engine
    resource: /lib/serving_engine.ml
    title: CPU RoPE table ownership and position-row binding
  - id: schedule-interface
    resource: /lib/serving_schedule.mli
    title: canonical RoPE runtime input names
  - id: smoke
    resource: /_artifacts/w4-engine-2026-08-27-rope
    title: fresh W4A16/KVQ8 serving pair
  - id: static-receipt
    resource: /bench/results/lfm25-350m-w4a16-kvq8-rope-table-elision-2026-08-27.txt
    title: offline schedule and native dispatch audit
  - id: benchmark-receipt
    resource: /bench/results/lfm25-350m-w4a16-kvq8-rope-table-elision-vs-pre-rope-2026-08-27.json
    title: shared warmup and scored HTTP comparison
---

# RoPE table binding

The decode capture previously rebuilt the current position's cosine and sine
rows through scalar arange, indexing, matrix multiplication, trigonometric,
elementwise, and cast commands. `Serving_schedule.Lfm25.specialize_decode`
now replaces every captured `[1;1;1;64]` trigonometric input with the canonical
runtime inputs `__llmopt_rope_cosine` and `__llmopt_rope_sine`, then runs the
existing liveness prune. The old branch is not part of the specialized command
stream.

`Serving_engine.Rope_table` reads the f32 inverse-frequency tensor from the
shared archive once at engine load, computes all 128,000 position rows with
LFM2's float32 cosine/sine contract, stores duplicated half-dimension values in
FP16 Metal buffers, and binds one row for each decode position. Ordinary
decode copies the selected row into fixed slots; replay uses distinct table-row
views for each encoded position so one replay command buffer cannot observe a
later slot mutation.

# Static result

The candidate is built from the preserved W4A16/KVQ8 capture and shared
322,667,136-byte archive. The pre-RoPE binary is the clean pre-slice build at
`9652c81e`; both binaries load the same candidate package, so the scheduler
delta is isolated from the earlier cast-absorption graph rewrite.

| Decode specialization | Commands | Runtime inputs | Paged-Q8 attention |
|---|---:|---:|---:|
| Pre-RoPE scheduler | 479 | 13 | 6 |
| Table-elision scheduler | 448 | 15 | 6 |
| Delta | -31 | +2 | 0 |

The captured branch contributes 22 graph nodes to the old decode plan. The
native smoke observes 567 decode kernel records over three generated tokens
before elision and 522 after it: 45 fewer records, or 15 fewer executable
dispatches per token. Both paths produce `518,509,7,708`; the 22 graph-node
count is therefore larger than the executable dispatch count because the
specialized schedule also contains movement/barrier bookkeeping.

`llmopt-lfm-serving-check` passes for the fresh pair (ABI 17, grouped Q8 KV,
page size 1, six paged-attention commands). `ninja -f ninja.build all test`
passes 42 Python tests and the OCaml canonical W4A16/KVQ8 suite.

# Native comparison

The shared `lfm25-mps-warmup.json` and `lfm25-mps-smoke.json` traces ran
sequentially with four warmup and four scored requests per engine,
`max_workers=1`, and fresh HTTP connections. The pre-RoPE binary ran first;
both engines completed 4/4 scored requests with zero output-token mismatches
and 80 cached prompt tokens.

| Engine | ERS | median TTFT | median TPOT |
|---|---:|---:|---:|
| Pre-RoPE scheduler | 0.6021973649 | 61.7960000 ms | 3.8912360 ms |
| RoPE table | 0.5944044400 | 60.9687710 ms | 4.1506180 ms |
| Table minus pre-RoPE | -0.0077929249 | -0.8272290 ms | +0.2593820 ms |

This is one sequential sample per engine; timing is a mixed observation while
the dispatch-count reduction and token-ID parity are direct smoke evidence.
