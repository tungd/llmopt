---
type: Experiment
title: 'XGBoost relative-target selector repair'
description: 'Grouped holdout and cross-validation evidence for a fixed-tile-relative Q8 cost-model target, plus a fresh static package replan.'
tags: [experiment, compiler, ocaml, metal, q8, xgboost, cost-model, tiling]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T02:26:33Z' }
sources:
  - id: report
    resource: /bench/results/lfm25-350m-q8-cost-model-relative-repair-2026-08-26.txt
    title: Relative-target cost-model repair report
  - id: validator
    resource: /bench/validate_cost_model.py
    title: Grouped cost-model validation harness
  - id: model
    resource: /_artifacts/cost-model-repair-2026-08-26/model-relative-delta-16x3.json
    title: Installed relative-delta XGBoost model
  - id: validation_receipt
    resource: /_artifacts/cost-model-repair-2026-08-26/relative-model-validation.json
    title: Grouped holdout and cross-validation receipt
  - id: package_receipt
    resource: /_artifacts/lfm25-350m-q8-relative-model-2026-08-26-v3/macro-audit.json
    title: Fresh static package replan receipt
---

# Target

The repaired target predicts each candidate tile's measured latency relative
to the same shape's fixed `16x16x64` latency. The dataset contract requires one
aggregated row per shape/tile and a fixed-tile row for every shape, preventing
repeated samples or cross-shape fixed baselines from entering the target.

# Results

The median native dataset contains 1,792 rows over 224 shapes and eight tiles.
With seed `23`, a 45-shape group holdout trains on 179 shapes and measures
`164409.671 us` fixed versus `163824.794 us` model-selected, a relative
reduction of `0.003557436715508102`; the model matches the measured winner on
25/45 shapes. Five grouped folds measure `731266.884 us` fixed versus
`728923.884 us` model-selected, a relative reduction of
`0.00320402858554732`, with 133/224 measured winners.

The 16-tree/depth-3 predictor is installed in the generated OCaml module,
while the typed selector and selector interface remain unchanged. The fresh
static full-package replan validates 699 prefill and 719 decode commands with
zero opaque commands.

# Boundary

The receipts are offline/static evidence. They do not establish the LOOP's
declared `>= 15%` Q8-linear speedup or exact four-prompt full-model result, and
the model was not executed on the device in this repair slice.
