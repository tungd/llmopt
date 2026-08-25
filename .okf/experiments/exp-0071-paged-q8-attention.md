---
type: Experiment
title: 'Direct paged-Q8 decode attention'
description: 'Typed decode specialization reads radix-owned Q8 K/V slots directly in generated Metal while retaining the selectable materialized FP16 path.'
tags: [experiment, compiler, ocaml, metal, q8, attention, radix-cache, kv-cache, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T10:51:18Z' }
sources:
  - id: implementation
    resource: /lib/serving_schedule.ml
    title: Typed direct-attention schedule rewrite
  - id: kernel
    resource: /lib/metal.ml
    title: Generated paged-Q8 attention kernel
  - id: runtime
    resource: /lib/serving_engine.ml
    title: Q8 and FP16 serving-path selection
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-paged-attention-compiler-2026-08-25.txt
    title: Compiler and exact Metal evidence
---

# Decode boundary

The captured LFM decode graph exposes six layer-specific past-key and
past-value tensors. For Q8 serving, a typed specialization traces each GQA
concat/expand/index chain back to its unique past input and current generated
row, then replaces the materialized attention command with
`Paged_attention_q8`. The primitive receives query, current key/value, the
physical mixed-Q8 token pool, an int32 radix slot map, and the attention mask.

The generated width-64 Metal entry dequantizes past K/V values at their point
of use, preserving the previous FP16 rounding boundary, while the current row
remains FP16. One SIMD group performs the online-softmax attention row. Cache
writeback packs only the current K/V row. The selectable FP16 policy keeps the
existing materialized path.

# Structural result

The full-Q8 LFM2.5-350M package pair remains 810/864 commands with zero opaque
operations and one 489,377,152-byte binary tensor archive. Runtime Q8 decode
specialization becomes 804 commands, 13 runtime inputs, and six direct paged
attention operations; the FP16 path remains 864 commands and 23 inputs.

At decode past lengths 1, 127, and 4,095, Q8 specialized workspace is 171,008,
172,800, and 199,424 bytes. The materialized FP16 path reports 189,440,
2,260,736, and 71,371,520 bytes at the same lengths.

# Exact execution evidence

One preflighted 60-second Apple M4 Pro attempt dispatched the new entry from a
161-command package. A zero-score fixture with cached value 2 and current value
4 returned FP16 value 3 in every output lane; all 47 fixture outputs were
exact. Static Ninja validation, both full-Q8 metallibs, package checks, and Q8
plus FP16 serving checks pass.

# Evidence boundary

This experiment is compiler, package, and exact synthetic-device evidence. It
does not report model latency or ERS. The bounded model target is
`LiquidAI/LFM2.5-350M`; the 2.6B variant remains deferred.
