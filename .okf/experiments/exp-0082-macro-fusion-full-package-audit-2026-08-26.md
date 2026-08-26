---
type: Experiment
title: 'Macro-fusion full-package audit'
description: 'Static validation of the fresh full-Q8 package containing residual-norm, dual-linear, QKV, ShortConv-step, and LM-head argmax fusions.'
tags: [experiment, compiler, ocaml, metal, q8, macro-fusion, serving, benchmark]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T02:10:00Z' }
sources:
  - id: audit
    resource: /bench/audit_macro_packages.py
    title: Macro package audit harness
  - id: report
    resource: /bench/results/lfm25-350m-q8-macro-final-audit-2026-08-26.txt
    title: Full-package macro-fusion audit report
  - id: prefill
    resource: /_artifacts/lfm25-350m-q8-lm-head-contract-2026-08-26/token-q8-prefill-v2
    title: Full-Q8 token-output prefill package
  - id: decode
    resource: /_artifacts/lfm25-350m-q8-lm-head-contract-2026-08-26/token-q8-decode-v2
    title: Full-Q8 token-output decode package
---

# Inventory

The fresh opt-in token-output packages select all five macro families. Prefill
contains 16 residual-norm, 16 dual-linear, 6 QKV, and 1 LM-head argmax
operation. Decode contains those same operations plus 10 fused ShortConv
steps.

# Package gates

The prefill package validates with 92 kernels, 699 commands, zero opaque
commands, workspace `315136/4217600` bytes, and 242 allocations. Decode
validates with 89 kernels, 719 commands, zero opaque commands, workspace
`142080/1004032` bytes, and 244 allocations. The Q8 serving-pair checker
passes ABI 14, page size 1, six prefill templates, and six decode-past
templates.

Against the prior `795/815` package command counts, the static reduction is 96
commands per stage and 192 combined.

# Boundary

The fresh packages were not relaunched after the single native token-only
probe stopped before model execution at residual metadata validation. Fresh
token parity and TPOT are therefore unmeasured. The older package bench has
exact 4/4 warmup and scored parity, but its one-run timing comparison is
explicitly invalid for a relative-speed claim and is not evidence for these
fresh all-five packages.
