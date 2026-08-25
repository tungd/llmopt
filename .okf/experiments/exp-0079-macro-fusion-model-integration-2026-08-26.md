---
type: Experiment
title: 'Macro fusion model integration evidence'
description: 'Validated current LFM2.5-350M Q8 prefill/decode packages with dual-linear, QKV, and ShortConv-step fusions and exact model-level token/logit parity.'
tags: [experiment, compiler, metal, q8, macro-fusion, lfm25, parity]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T00:00:00Z' }
sources:
  - id: report
    resource: /bench/results/lfm25-350m-q8-macro-fusion-integration-2026-08-26.txt
    title: Macro-fusion integration report
  - id: result
    resource: /_artifacts/lfm25-benchsuite-macro-2026-08-26/result.json
    title: Full model benchsuite result
---

# Integration result

The current `llmopt` benchsuite packages validate with 0 opaque commands. The
prefill packages contain 16 `q8-dual-linear+silu` and 6 `q8-qkv-linear`
operations. The decode packages contain those same 16 and 6 operations plus
10 `short-conv-step-fused` operations.

The model-level result has exact cross-process FP16 logits for shape
`[1,14,65536]`, exact 4/4 warmup token parity, and exact 4/4 scored token
parity. The residual-norm and LM-head-argmax operations remain unselected for
the documented graph-contract reasons.

The timing comparison is one run per candidate without counterbalancing, and
the package command deltas are `767/787` versus prior `795/815`; those facts
remain evidence boundaries for the separate end-to-end benchmark item.
