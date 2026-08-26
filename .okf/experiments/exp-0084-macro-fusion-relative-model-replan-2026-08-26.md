---
type: Experiment
title: 'Macro-fusion replan with relative cost model'
description: 'Static full-Q8 package audit after installing the fixed-tile-relative selector predictor.'
tags: [experiment, compiler, ocaml, metal, q8, macro-fusion, cost-model, serving]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T02:26:33Z' }
sources:
  - id: report
    resource: /bench/results/lfm25-350m-q8-macro-relative-model-replan-2026-08-26.txt
    title: Relative-model macro-fusion replan report
  - id: audit
    resource: /bench/audit_macro_packages.py
    title: Macro package audit harness
  - id: model
    resource: /_artifacts/cost-model-repair-2026-08-26/model-relative-delta-16x3.json
    title: Installed relative-delta cost model
  - id: receipt
    resource: /_artifacts/lfm25-350m-q8-relative-model-2026-08-26-v3/macro-audit.json
    title: Fresh package audit receipt
---

# Static replan

The fresh prefill/decode package pair was generated after installing the
relative-delta predictor and retains all five macro families. Prefill contains
16 residual-norm, 16 dual-linear, 6 QKV, and 1 LM-head argmax operation;
decode adds 10 fused ShortConv steps.

The packages validate with 699 and 719 commands, respectively, zero opaque
commands, and 92 and 89 kernels. The Q8 serving-pair checker passes ABI 14,
grouped-Q8 width 64, page size 1, six prefill templates, six decode-past
templates, and the specialized decode metadata. The static command reduction
against the prior 795/815 pair is 192 combined commands.

# Boundary

This is a static package replan. Fresh model token parity and TPOT remain
unmeasured because the single native token-only probe stopped before model
execution at residual metadata validation.
