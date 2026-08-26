---
type: Experiment
title: 'Q8 residual/output metadata validation repair'
description: 'Reject generic Q8 residual schedules with runtime-invalid output metadata before native dispatch.'
tags: [experiment, compiler, ocaml, q8, macro-fusion, serving, metadata]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T02:44:23Z' }
sources:
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Runtime residual/output metadata check
  - id: schedule
    resource: /lib/serving_schedule.ml
    title: Serving schedule validation
  - id: regression
    resource: /test/test.ml
    title: Malformed residual schedule regression
  - id: report
    resource: /bench/results/lfm25-350m-q8-residual-metadata-validator-2026-08-26.txt
    title: Offline repair receipt
  - id: packages
    resource: /_artifacts/lfm25-350m-q8-relative-model-2026-08-26-v3
    title: Current relative-model serving package pair
---

# Finding

The native runtime already rejected a generic `Q8_linear_add` or
`Q8_linear_mul_add` command when the residual input's dtype or logical shape
differed from the command output. The schedule validator did not enforce that
same invariant, allowing a malformed package to reach native execution.

# Repair and regression

`Serving_schedule.validate_command` now checks the final residual input of
both generic residual epilogues with exact dtype and logical-shape equality.
The OCaml test suite constructs a `[1,3]` residual against a `[2,3]` output and
requires `Serving_schedule.of_graph` to reject it. Existing abstract Q8
projection fixtures remain valid because the repair is limited to the
runtime-relevant residual/output contract.

# Current package evidence

The rebuilt validator accepts the current relative-model package pair:

| Stage | Kernels | Commands | Opaque | Workspace live/peak |
|---|---:|---:|---:|---:|
| Prefill | 92 | 699 | 0 | 315136/4217600 bytes |
| Decode | 89 | 719 | 0 | 142080/1004032 bytes |

The ABI-14 grouped-Q8 serving-pair checker also passes its six prefill
templates, six decode-past templates, and specialized decode metadata.

# Boundary

This is an offline validator repair. The native token-id differential, fresh
full-model token parity, and TPOT gates in the remaining macro LOOP items were
not rerun because the repository's one-attempt boundary prohibits a second
device probe in this slice.
