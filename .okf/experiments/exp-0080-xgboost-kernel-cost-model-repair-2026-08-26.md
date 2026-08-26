---
type: Experiment
title: 'XGBoost kernel cost-model repair audit'
description: 'Independent holdout and measured-oracle audit for the Q8 tile selector after the broad Apple Silicon sweep.'
tags: [experiment, compiler, ocaml, metal, q8, xgboost, cost-model, tiling, benchmark]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T00:00:00Z' }
sources:
  - id: report
    resource: /bench/results/lfm25-350m-q8-cost-model-repair-2026-08-26.txt
    title: Cost-model repair audit report
  - id: dataset
    resource: /_artifacts/cost-model-sweep-2026-08-26/device-dataset-broad.jsonl
    title: Broad native Q8 sweep dataset
  - id: model
    resource: /_artifacts/cost-model-sweep-2026-08-26/model-validation-24x2.json
    title: Existing XGBoost validation result
---

# Repair audit

The broad native sweep contains 8,960 rows over 224 `(M,N,K)` shapes and eight
parameterized tiles. Reducing each shape/tile to its median yields 1,792 rows.

A deterministic seed-23 holdout of 45 shapes was evaluated with all eight
candidate tiles retained per shape. The current 24-tree/depth-2 absolute
latency model selected the measured winner for `2/45` held-out shapes. An
offline pilot whose target was latency relative to fixed `16x16x64`, with the
same tree budget, also selected the measured winner for `2/45`; the pilot was
not installed in the runtime selector.

The measured oracle across all 224 shapes reduces summed latency by
`0.7342222268607368%` over fixed `16x16x64`. The existing transpilation result
remains separately recorded at `R2=0.9569819197141608`, portable max relative
error `1.4329246283350704e-07`, and measured-winner selection `1/16`.

# Boundary

These measurements do not establish the LOOP's declared `>= 15%` Q8 linear
speedup or a valid static-versus-dynamic full-model comparison. A revised
measurement/model-selection plan is required before another device probe.
