---
type: Decision
title: 'Retired XGBoost kernel cost model'
description: 'Historical decision record for the removed learned kernel selector; current compilation consumes direct measurements from the native hardware-probing phase.'
tags: [decision, compiler, cost-model, xgboost, retired]
status: deprecated
generated: { by: codex/gpt-5, at: '2026-09-01T01:30:00+07:00' }
sources:
  - id: replacement
    resource: /decisions/measured-hardware-probing.md
    title: Measured hardware probing decision
  - id: probe
    resource: /bin/hardware_probe.ml
    title: Native target measurement CLI
---

# Decision

The offline XGBoost training, Python-to-OCaml transpiler, generated latency
predictor, evaluation scripts, optional dependency, and predictor tests are
removed. The selector did not reliably choose measured winners across the
recorded shape sweep, and retaining a generated predictor duplicated the
device facts already required by AOT compilation.

`llmopt-hardware-probe` now runs real Metal work on the selected device before
compilation. It measures empty-dispatch time and a 16/64/256 MiB streaming-copy
curve, records every working-set observation, and calibrates the target with
the highest directly observed effective bandwidth. `llmopt-fx` and
`llmopt-pipeline` consume that receipt through `--target-profile` and preserve
the measured target in emitted artifacts.

The historical XGBoost experiment records remain evidence of the retired
approach; they are not descriptions of the current compiler.
