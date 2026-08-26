---
type: Experiment
title: 'XGBoost static-versus-dynamic cost-model audit'
description: 'Reproducible paired-median comparison of the existing transpiled tile model against the fixed Q8 tile policy.'
tags: [experiment, compiler, ocaml, metal, q8, xgboost, cost-model, benchmark]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T02:02:54Z' }
sources:
  - id: audit
    resource: /bench/evaluate_cost_model.py
    title: Static-versus-dynamic cost-model audit harness
  - id: report
    resource: /bench/results/lfm25-350m-q8-cost-model-repair-2026-08-26.txt
    title: Cost-model repair report
  - id: dataset
    resource: /_artifacts/cost-model-repair-2026-08-26/device-dataset-broad-median.jsonl
    title: Five-sample broad-sweep median dataset
  - id: model
    resource: /_artifacts/cost-model-repair-2026-08-26/model-24x2.json
    title: Existing 24-tree depth-2 XGBoost artifact
---

# Method

`bench/evaluate_cost_model.py` loads one measured median for every `(M,N,K)`
shape and each of the eight candidate tiles. It evaluates the existing
24-tree/depth-2 artifact with the same feature-row contract used by training,
selects the predicted minimum per shape, and compares the resulting measured
sum with the fixed `16x16x64` measured sum.

The dataset contains 1,792 rows over 224 shapes, with `sample_count=5` for
every row. The independent holdout is the first 45 shapes after sorting and
shuffling with `random.Random(23)`, matching the preceding repair audit.

# Results

| Set | Fixed `16x16x64` (us) | Model-selected (us) | Relative reduction | Measured winners |
| --- | ---: | ---: | ---: | ---: |
| All 224 shapes | 731266.884 | 800881.744 | -0.09519761050741068 | 27/224 |
| Holdout 45 shapes | 164409.671 | 180172.915 | -0.09587783920569963 | 2/45 |

The artifact selects `8x32x64` for every shape in both sets. The JSON receipt
is `_artifacts/cost-model-repair-2026-08-26/static-vs-dynamic.json`.

# Boundary

This is a valid static-versus-dynamic comparison over paired medians, not a
new device probe. It does not establish the LOOP's declared `>= 15%` Q8-linear
speedup or its exact four-prompt full-model token result; the runtime selector
therefore remains on its fixed-tile behavior while a separate model-repair
plan is pending.
