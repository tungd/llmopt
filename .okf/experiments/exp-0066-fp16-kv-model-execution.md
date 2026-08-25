---
type: Experiment
title: 'Model-scale selectable FP16 KV execution'
description: 'Execute the paired full-Q8 350M package with native FP16 KV and recurrent checkpoints while retaining Q8 as the default policy.'
tags: [experiment, ocaml, metal, kv-cache, fp16, q8, radix-cache, benchmark, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T10:03:08Z' }
sources:
  - id: fp16-result
    resource: /bench/results/lfm25-350m-q8-paired-simd-fp16-kv-measurement-2026-08-25.txt
    title: FP16 KV model-scale observation
  - id: q8-result
    resource: /bench/results/lfm25-350m-q8-paired-simd-measurement-2026-08-25.txt
    title: Paired Q8 KV comparison observation
  - id: cache-owner
    resource: /lib/serving_cache.ml
    title: Native cache ownership and format policy
---

# Execution

One bounded LFM2.5-350M run selects `--kv fp16` on the same paired full-Q8
weight package used by the default Q8-cache observation. The persistent OCaml
server completes 4/4 warmup and 4/4 scored requests, reproduces all established
eager and paired-Q8 output IDs, and reports the same 80/194 radix reuse.

This exercises model-scale FP16 attention KV and recurrent checkpoint
pack/unpack rather than only package validation or a small fixture. Q8-group-64
remains the default serving configuration.

# Observation

FP16-cache ERS is `0.4297032150753201`, median TTFT is
`69.16322899633087 ms`, and median TPOT is `6.698208337184042 ms`. Relative to
the separate paired Q8-cache observation, those values change by
`+0.022687456709165554`, `-6.06170849641785 ms`, and
`-0.23844450091322233 ms`. All four per-request token sequences agree.

# Evidence boundary

The FP16 and Q8 cache reports are non-interleaved single observations. No
FP16-cache long-context matrix or new eager process was run.
