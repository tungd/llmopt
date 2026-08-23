---
type: Experiment
title: 'Rank-aware FX primitives and fused RMSNorm Metal kernel'
description: 'Capture Dynamo result metadata, lower common LFM pointwise/layout operations into typed commands, and fuse the RMSNorm chain.'
tags: [experiment, fx, ocaml, rank, rmsnorm, fusion, metal, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T18:34:45Z' }
sources:
  - id: exporter
    resource: /python/llmopt_backend/__init__.py
    title: Dynamo FX manifest exporter
  - id: planner
    resource: /lib/fx_plan.ml
    title: Rank-aware FX primitive lowering
  - id: passes
    resource: /lib/passes.ml
    title: RMSNorm graph fusion
  - id: metal
    resource: /lib/metal.ml
    title: Fused RMSNorm Metal emitter
  - id: tests
    resource: /test/test.ml
    title: OCaml primitive, schedule, CPU, and fusion tests
---

# Question

Can the next real Dynamo capture preserve the shapes and constants required to
replace LFM2.5's RMSNorm and layout chains with typed, optimizable commands?

# Implementation

The exporter now reads Dynamo's `example_value` metadata in addition to `val`
and `tensor_meta`. Manifest v2 carries an explicit typed argument tree,
including Python ellipsis, and v2 nodes are rejected if that tree is absent.
The planner does not reinterpret the old v1 capture as v2 because its constants
and computed shapes were never recorded.

The OCaml IR now has typed N-dimensional primitives for:

* scalar/tensor pointwise add, multiply, subtract, logical-and, equality,
  inequality, less-or-equal, negation, reciprocal square root, SiLU, sine,
  cosine, and scalar power;
* dtype casts and mean reductions with normalized axes;
* view, reshape, transpose, unsqueeze, expand, and contiguous movement; and
* a fused RMSNorm command.

The CPU effect handler interprets N-dimensional broadcasting, mean reduction,
and movement semantics over its flattened Bigarray storage. The binary schedule
codec preserves each primitive and writes schedule version 2 while retaining
read compatibility with version 1.

The optimizer recognizes the seven-command square/mean/epsilon/rsqrt/scale/
cast/weight chain and replaces it with one RMSNorm command when each removed
intermediate has no other consumer. The Metal emitter supplies
`llmopt_rms_norm_f32_f16` and `llmopt_rms_norm_f16`; the initial kernel uses a
serial width reduction per row and is a correctness boundary, not a latency
result.

# Static evidence

`ninja -f ninja.build all test rms-norm-smoke fx-smoke q8-smoke q8-fx-smoke
q8-serving-smoke` passes:

* 26 Python tests;
* the OCaml reference suite, including N-dimensional RMSNorm evaluation;
* binary round-trip of pointwise, reduction, movement, fused RMSNorm, and
  ellipsis-bearing arguments;
* the existing LLVM, Metal, package, and Q8 checks; and
* Xcode Metal compilation of the fused RMSNorm source.

The model-shaped RMSNorm fixture changes from 10 captured commands to four:
two inputs, one fused command, and one output.

# Bounded model attempt

Before the only model launch, `memory_pressure -Q` reported 58% system-wide
memory free on a 25,769,803,776-byte host, 48 GiB disk available, and no active
model process. The six-token LFM2.5-350M Q8 capture stopped before OCaml or
Metal compilation because the v2 argument exporter did not yet represent
Python `Ellipsis`. The codec was corrected and covered offline; the model was
not retried. The failed attempt produced no graph package, latency comparison,
or ERS score, and memory subsequently reported 55% free with no model process.

# Remaining boundary

The only real 350M package remains the saved v1 no-cache capture with 736 opaque
commands. A later single v2 capture must establish real primitive and RMSNorm
counts. Getitem, concat/chunk, embedding, ShortConv, attention, generated
pointwise dispatch, and native command-stream execution remain unimplemented.
