---
type: Experiment
title: 'XGBoost kernel cost-model device sweep'
description: 'Parameterized native Q8 tile dispatch, broad Apple Silicon measurements, numerical comparison, and offline XGBoost validation with the selection boundary preserved.'
tags: [experiment, compiler, ocaml, metal, q8, xgboost, cost-model, tiling, benchmark]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T00:00:00Z' }
sources:
  - id: report
    resource: /bench/results/lfm25-350m-q8-cost-model-sweep-2026-08-26.txt
    title: Cost-model device sweep report
  - id: dataset
    resource: /_artifacts/cost-model-sweep-2026-08-26/device-dataset-broad.jsonl
    title: Broad native Q8 sweep dataset
  - id: validation
    resource: /_artifacts/cost-model-sweep-2026-08-26/model-validation-24x2.json
    title: XGBoost and OCaml validation result
  - id: correctness
    resource: /_artifacts/cost-model-sweep-2026-08-26/parameterized-correctness.json
    title: Parameterized native Q8 correctness probe
---

# Device sweep

The native bridge now resolves the generated parameterized Q8 linear entry
points from the metallib and passes the selected `(Tm,Tn,Tk)` through to the
Metal dispatch grid. The fixed `16x16x64` name remains the compatibility entry;
float32 input uses the `_f32` family. The focused Python bridge suite passes
5/5 and the native bridge recompiles with `ninja -f ninja.build metal-runtime`.

The fresh device data contains 640 rows over 16 shapes, a 25-sample repeat with
3,200 rows, and a broad 8,960-row sweep over 224 shapes with `M` from 1 to
4096. The broad exhaustive oracle has 21 of 112 shape groups with `M=2..64`
at or above the declared 15% fixed-tile delta, while its mean oracle delta is
7.30039710966991%; this does not describe a learned selector.

# Model and numerical evidence

The 24-estimator/depth-2 model fit on the 640-row dataset reaches held-out
`R2=0.9569819197141608`; the transpiled OCaml evaluator has portable max
relative error `1.4329246283350704e-07` and source size `13,355` bytes. Its
exhaustive top-1 selection is `1/16 = 0.0625` on the recorded held-out
measurements for all three device metadata configurations. The model remains
an artifact and is not installed as the runtime selector.

The parameterized correctness probe uses shape `(13,1024,2048)` and the MPS
dequantized linear result as reference. All eight tiles, including the fixed
baseline, report `max_abs=0.0625` and `mean_abs=1.3649463653564453e-05`, so
the result records common native Q8 drift rather than exact tile parity.

# Boundary

These artifacts close parameterized dispatch and measurement coverage. They do
not close the LOOP item: the learned selector misses the supporting decision's
declared top-1 criterion, and the model-level static-versus-dynamic speedup
condition remains unestablished.
