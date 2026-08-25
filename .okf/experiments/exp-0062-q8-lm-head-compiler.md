---
type: Experiment
title: 'Q8-default LM-head compiler boundary'
description: 'Quantize the LFM vocabulary projection by default and specialize typed Q8 prefill logits to one row.'
tags: [experiment, compiler, pytorch, ocaml, q8, lm-head, prefill, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T09:29:48Z' }
sources:
  - id: frontend
    resource: /python/llmopt_backend/quantization.py
    title: Q8 model rewrite boundary
  - id: specializer
    resource: /lib/serving_schedule.ml
    title: Typed final-row serving specialization
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-lm-head-compiler-2026-08-25.txt
    title: Static compiler evidence
---

# Boundary

The Q8 frontend previously skipped every module whose qualified name ended in
`lm_head`. The default now converts every eligible `nn.Linear`, including the
vocabulary projection. Callers retain an explicit `skip_suffixes` opt-out. For
LFM, replacing the tied linear module leaves the FP16 token embedding intact
and creates separate contiguous int8 projection weights plus one scale per
vocabulary row.

The OCaml LFM prefill specializer now treats `Linear` and `Q8_linear` as typed
vocabulary projections. Both require the same named-output, sole-consumer, and
identity-index invariants. The pass moves the final-row slice before the
projection and rewrites `m` to one. Existing Q8 runtime dispatch then selects
the packed SIMD-group GEMV path without a package-ABI or Metal-entry change.

# Verification

Python tests prove the new default and the explicit skip policy. An OCaml graph
test specializes `q8-linear[6x4x2]` to `q8-linear[1x4x2]`, checks its
`[1,1,4]` output, and runs liveness planning. Full Ninja OCaml/Python, Q8,
serving-package, Metal, LLVM, and native-schedule checks pass.

No model or device workload ran. The preserved ABI-v11 LFM2.5-350M package
still contains its historical FP16 tied head; a bounded 350M recapture must
materialize the additional Q8 projection tensors before runtime measurement.
