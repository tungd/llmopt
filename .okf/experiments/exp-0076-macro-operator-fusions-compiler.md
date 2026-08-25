---
type: Experiment
title: 'Macro-operator fusion compiler boundary'
description: 'Fresh full-Q8 LFM2.5 prefill and decode packages compile and validate with dual-linear, QKV, and decode ShortConv-step matches; residual-norm and LM-head argmax remain opt-in at the captured graph boundary.'
tags: [experiment, compiler, ocaml, metal, q8, macro-fusion, qkv, swiglu, short-conv, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T00:00:00Z' }
sources:
  - id: implementation
    resource: /bench/results/lfm25-350m-q8-macro-fusions-compiler-2026-08-26.txt
    title: Macro-fusion compiler boundary report
  - id: prefill
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-replan-v8-2026-08-24/prefill/graph.llmopt
    title: Captured Q8 prefill graph
  - id: decode
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-replan-v8-2026-08-24/decode/graph.llmopt
    title: Captured Q8 decode graph
---

# Compiler/package probe

The fresh full-Q8 plans compile and pass `llmopt-package-check` after generated
Metal source compilation. Prefill plans contain 16 `q8-dual-linear+silu` and 6
`q8-qkv-linear` operations; decode plans contain those same 22 operations plus
10 `short-conv-step-fused` operations.

The resulting command counts are 765 for prefill and 785 for decode, compared
with 795 and 815 in the prior validated package. Both packages retain zero
opaque commands. The generated packages use 95 and 92 declared kernels,
respectively, because the macro variants add typed entry points.

# Evidence boundary

The captured residual output has an external downstream consumer, so
`Q8_linear_add_norm` is not selected for that graph. The captured graph exposes
`logits`, not the opt-in `token_id` output, so `Q8_lm_head_argmax` is not
selected. This probe does not include a current-state model forward, token
parity, or latency result.
