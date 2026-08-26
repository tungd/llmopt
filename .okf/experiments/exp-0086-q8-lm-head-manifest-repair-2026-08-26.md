---
type: Experiment
title: 'Q8 LM-head kernel-manifest repair and bounded native probe'
description: 'Diagnose the token-only serving failure, repair mixed-fusion kernel manifest selection, and record the remaining fresh execution boundary.'
tags: [experiment, compiler, ocaml, metal, q8, macro-fusion, serving, argmax]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T03:53:51Z' }
sources:
  - id: emitter
    resource: /lib/metal.ml
    title: Q8 Metal emitter and kernel manifest selection
  - id: regression
    resource: /test/test.ml
    title: Mixed residual-norm and LM-head ABI regression
  - id: probe
    resource: /bench/results/lfm25-350m-q8-lm-head-manifest-repair-2026-08-26.txt
    title: Single native probe and repair receipt
  - id: v4_audit
    resource: /_artifacts/lfm25-350m-q8-relative-model-2026-08-26-v4/macro-audit.json
    title: Corrected static v4 package audit
  - id: v3_probe
    resource: /_artifacts/native-http-relative-model-v3-2026-08-26/server.stderr
    title: v3 native server kernel lookup failure
---

# Finding

The authorized native serving probe reached the v3 server and failed before
model dispatch because the serving package declared no
`llmopt_q8_lm_head_argmax_f32` entry for the f32 activation to Int32 token-id
path. The generated `kernel.metal` and `kernel.metallib` already contained that
function, so the failure was in package manifest selection rather than Metal
emission.

# Repair

`q8_entries` in `lib/metal.ml` now appends residual-norm and LM-head entry sets
independently. A mixed graph regression in `test/test.ml` requires the f32
LM-head entry to remain declared when residual-norm fusion is present.

The corrected v4 static pair validates as follows:

| Stage | Kernels | Commands | Opaque | LM-head operation |
|---|---:|---:|---:|---:|
| Prefill | 96 | 699 | 0 | 1 |
| Decode | 93 | 719 | 0 | 1 |

The ABI-14 grouped-Q8 serving-pair checker passes. The v4 pair was generated
after the single native attempt and has not been dispatched on the device.

# Boundary

The one native probe recorded `0/4` successful warmup requests and the exact
kernel lookup error; it produced no token parity or TPOT measurement. No
second device run was issued. The static repair leaves the macro token-output,
full-model TPOT, and XGBoost `>= 15%` speedup gates open.
