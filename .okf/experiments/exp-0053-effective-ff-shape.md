---
type: Experiment
title: 'Effective LFM feed-forward shape correction'
description: 'Separate the checkpoint-declared intermediate size from the auto-adjusted 4608-wide SwiGLU projections observed in every captured layer.'
tags: [experiment, target, lfm25, shapes, swiglu, compiler, ocaml]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T07:32:19Z' }
sources:
  - id: target
    resource: /lib/lfm25.ml
    title: Typed LFM target descriptor
  - id: capture
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi11-simd-attention-v1-2026-08-25/prefill/plan.txt
    title: Captured executable projection shapes
  - id: evidence
    resource: /bench/results/lfm25-350m-effective-feed-forward-shape-2026-08-25.txt
    title: Shape derivation and static checks
---

# Declared and executable dimensions

The checkpoint declares `intermediate_size=6656` and enables automatic
feed-forward adjustment. Transformers first computes integer
`2 * 6656 / 3 = 4437`, then rounds upward to the configured multiple of 256,
producing 4608. It constructs `w1` and `w3` as `[4608,1024]` and `w2` as
`[1024,4608]`.

Every feed-forward command in the preserved prefill and decode captures uses
those effective shapes. The previous OCaml model-shaped fixture used the raw
6656 declaration and therefore exercised a projection that does not exist in
the executable checkpoint.

# Typed correction

`Lfm25.Config.t` now keeps `intermediate_size` for the checkpoint declaration
and adds `feed_forward_size` for the executable projection width. The default
values are 6656 and 4608 respectively, validation requires both to be
positive, and model-shaped Q8/FP16 linear fixtures consume the effective field.

# Evidence boundary

OCaml/Python tests and Q8/RMSNorm Ninja smoke targets pass. This changes only
the synthetic target fixture and domain documentation; captured model packages
already had the correct 4608 shapes. No model or device run occurred.
