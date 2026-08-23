---
type: Experiment
title: 'Static indexing, chunk elimination, and typed concat'
description: 'Normalize LFM tensor indexing, eliminate tuple-valued chunk nodes at planning time, and preserve concat in the binary schedule.'
tags: [experiment, fx, ocaml, indexing, chunk, concat, fusion, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T18:53:41Z' }
sources:
  - id: saved-capture
    resource: /_artifacts/lfm25-350m-q8-binary-replan-2026-08-24/fx.json
    title: Saved LFM2.5-350M Q8 manifest v1
  - id: shape-domain
    resource: /lib/tensor_shape.ml
    title: Static tensor shape and index normalization
  - id: planner
    resource: /lib/fx_plan.ml
    title: FX planner and deferred chunk lowering
  - id: cpu-reference
    resource: /lib/cpu.ml
    title: CPU primitive interpreter
  - id: schedule
    resource: /lib/serving_schedule.ml
    title: Binary serving schedule codec and validator
---

# Question

Can the high-frequency LFM indexing boundary become typed without introducing
a tuple-valued runtime object or materializing `torch.chunk` outputs?

# Capture inventory

The saved 1,115-node no-cache manifest contains 85 `_operator.getitem` nodes.
Thirty consume 10 `chunk` nodes, 30 consume transposes, 10 consume `conv1d`,
and the remainder index other tensors. The adjacent graph contains 13 `cat`
nodes. This inventory describes the saved v1 graph; it is not a new v2 model
capture.

# Implementation

`Tensor_shape.Index` is an abstract normalized index with smart construction.
It expands one ellipsis, normalizes negative integer indices and Python slice
bounds, represents inserted axes, and derives the output shape. Shape-domain
operations also verify concat compatibility and reproduce `torch.chunk`'s
possibly-fewer-than-requested partition sizes.

The FX planner keeps a valid `chunk` as a deferred compile-time partition. A
following integer `getitem` becomes an index command over the original tensor,
so the captured IR contains neither a tuple value nor a chunk materialization.
Regular static tensor indexing and N-dimensional concat lower to typed movement
commands. Dynamic/advanced tensor indexing remains opaque.

The CPU handler interprets normalized index and concat commands over flattened
N-dimensional storage. Binary schedule version 3 serializes their compact
integer descriptors, retains read compatibility with versions 1 and 2, and
re-derives result shapes while validating a decoded command.

# Static evidence

`ninja -f ninja.build all test` passes:

* 27 Python tests, including tuple-valued Dynamo `example_value` metadata for
  chunk;
* the OCaml shape and CPU-reference tests for chunk slices, concat, ellipsis,
  and static slicing; and
* a synthetic v2 FX graph that becomes six commands with zero opaque nodes:
  one input, three normalized index commands, one concat, and one output.

The current package checker also accepts the saved 116,861-byte schedule-v1
model package and reports its original 1,115 commands, 736 opaque commands, and
241 tensor-store bindings, providing backward-reader evidence after the v3
writer change.

No model or Metal device process was launched for this slice. The only real
operator count remains the saved v1 result with 736 opaque commands.

# Remaining boundary

Index and concat Metal kernels and native command dispatch are not implemented.
The real v2 recapture is also still pending, so this slice does not claim a new
model-level opaque count, parity result, latency, or ERS score.
