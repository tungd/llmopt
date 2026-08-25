---
type: Experiment
title: 'Serving-only last-token vocabulary projection'
description: 'Rewrite the LFM prefill LM head to project one vocabulary row instead of materializing full-sequence logits.'
tags: [experiment, compiler, ocaml, metal, prefill, logits, memory, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T09:13:47Z' }
sources:
  - id: pass
    resource: /lib/serving_schedule.ml
    title: Typed LFM serving specialization
  - id: checker
    resource: /bin/lfm_serving_check.ml
    title: Static serving-pair checker
  - id: evidence
    resource: /bench/results/lfm25-350m-last-token-projection-2026-08-25.txt
    title: LFM2.5-350M static projection evidence
---

# Boundary

The saved Dynamo/FX prefill template computes `[1,tokens,65536]` FP16 logits,
although native serving samples only the final row. The LFM prefill
specializer now recognizes the exact safe chain: named `logits` output, a
sole-consumer FP16 linear projection, and a sole-consumer full identity index.
It rewrites that index to the final hidden row and changes the linear `m`
dimension to one. Graphs without a named logits output remain unchanged;
unrecognized logits chains return a typed specialization error.

This is a serving specialization, not a capture-format or package-ABI change.
The ABI-v11 package and dynamic FP16 linear kernel are reused.

# Static evidence

The real LFM2.5-350M Q8 package specializes prefill lengths 13, 128, and 4,096
to exactly one logits row. At 4,096 tokens, the vocabulary output allocation is
131,072 bytes for `[1,1,65536]` FP16 instead of 536,870,912 bytes for the
captured full-sequence shape. The complete liveness-planned workspace is
184,680,448 bytes at that length.

The OCaml regression constructs a full-sequence logits graph, checks the final
row selectors, checks `linear[1x4x2]` and `[1,1,4]` output inference, and runs
the memory planner. Full Ninja OCaml/Python, Q8, serving-package, Metal, LLVM,
and native-schedule validation passes. No model or device workload ran in this
experiment; the bounded execution target is `LiquidAI/LFM2.5-350M`.
