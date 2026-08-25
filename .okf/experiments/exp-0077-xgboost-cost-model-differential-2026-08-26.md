---
type: Experiment
title: 'XGBoost cost-model differential evidence'
description: 'Fresh LFM2.5-350M Q8 differential and four-request benchsuite observations record exact parity for the specified eager, fallback, generated-exact, and token-output comparisons while preserving the isolated timing evidence boundary.'
tags: [experiment, compiler, ocaml, metal, q8, xgboost, cost-model, differential, benchmark, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T00:00:00Z' }
sources:
  - id: report
    resource: /bench/results/lfm25-350m-q8-cost-model-differential-2026-08-26.txt
    title: Q8 cost-model differential evidence report
  - id: differential
    resource: /_artifacts/phase1-350m-differential-2026-08-26/result.json
    title: Fresh full differential result
  - id: bench
    resource: /bench/results/lfm25-350m-q8-macro-bench-2026-08-26.json
    title: Fresh four-request Q8 benchsuite result
---

# Differential and model-level probe

The fresh Q8 differential completes successfully on LFM2.5-350M. The direct
`[3, 29, 37]` FP16 probe is exact, eager/fallback and eager/generated-exact
logits are exact, and the generated-native path preserves argmax while
reporting `max_abs=0.056640625` against eager logits.

The isolated four-request benchsuite has `engine_pass=true`, exact FP16
cross-process logits digests, and exact token output parity for 4/4 scored and
4/4 warmup requests. Its scored median TPOT observations are
`42.8826249941873 ms` for eager and `23.47580531689649 ms` for llmopt, but the
runner marks the comparison invalid for a relative-speed claim because each
candidate ran once in a separate process without repeated or counterbalanced
samples.

# Evidence boundary

The artifacts record the parity and raw timing observations above. They do not
establish the LOOP item's static-versus-dynamic `>= 15%` Q8 linear speedup
condition, and the generated-native full-logit comparison is not exact even
though its argmax is exact.
