---
type: Experiment
title: 'Qualified FX target matching'
description: 'Prevent short operator suffixes from capturing unrelated method names and recover all expand nodes in the saved LFM graph.'
tags: [experiment, fx, ocaml, target-matching, expand, regression]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T19:00:06Z' }
sources:
  - id: planner
    resource: /lib/fx_plan.ml
    title: FX target matcher and planner
  - id: regression
    resource: /test/test.ml
    title: Expand and logical-and collision regression
  - id: replan
    resource: /_artifacts/lfm25-350m-q8-v3-expand-replan-rebuilt-2026-08-24/plan.txt
    title: Offline replan of the saved manifest-v2 graph
---

# Finding

The target matcher accepted an operator whenever its text ended with a known
candidate. Consequently, the method target `expand` matched the short logical
operator candidate `and` before the planner reached its expand branch. All 14
real-model expand nodes became opaque despite an existing typed expand
primitive.

# Change

A non-exact target now matches a candidate only when the suffix starts after a
`.` qualification boundary. Fully qualified targets such as
`aten.add.Tensor` still match their intended candidate, while `expand` no
longer matches `and`.

The regression graph expands `[1,8,1,6,64]` to `[1,8,2,6,64]` and requires a
typed movement command with zero opaque commands. The complete OCaml and
Python test target passes.

# Saved-manifest evidence

Rebuilding `llmopt-fx` and replanning the already captured manifest produced:

```text
planned 1115 FX nodes into 835 IR nodes
valid compiled-graph package: 3 kernels, 835 commands, 28 opaque, tensor-store=none
```

The same manifest had 42 opaque commands before the fix. Its 14 expand nodes
are now typed, leaving 10 conv1d, 6 scaled-dot-product attention, 5 arange, 2
advanced getitem, and one each of embedding, diff, cumsum, new_ones, and a
logging side effect. This was an offline replan; no model/device process,
parity measurement, generation, cache request, needle request, or ERS run was
performed.
