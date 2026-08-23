---
type: Experiment
title: 'Typed LFM token embedding and Metal gather'
description: 'Lower the captured int64-to-float16 token lookup into typed OCaml IR, a versioned binary command, CPU bounds-checked gather, and generated MSL.'
tags: [experiment, lfm25, embedding, gather, ocaml, metal, schedule]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T19:24:26Z' }
sources:
  - id: shape
    resource: /lib/tensor_shape.ml
    title: Embedding shape validation
  - id: planner
    resource: /lib/fx_plan.ml
    title: Captured embedding lowering
  - id: cpu
    resource: /lib/cpu.ml
    title: Bounds-checked CPU gather
  - id: metal
    resource: /lib/metal.ml
    title: Generated embedding MSL
  - id: replan
    resource: /_artifacts/lfm25-350m-q8-v6-embedding-serving-replan-2026-08-24/package.llmopt
    title: Offline serving-package replan
---

# Targeted form

The captured model starts with one inference embedding call:

```text
indices [1,6] int64
weight  [65536,1024] float16
padding_idx 0, max_norm none, norm_type 2
scale_grad_by_freq false, sparse false
output  [1,6,1024] float16
```

The planner accepts only int64 indices, float16 weights, no `max_norm`, and the
dense unscaled inference options. Unsupported variants retain opaque fallback.

# Implementation

Embedding shape inference appends the rank-two weight width to the index
shape. The schedule-v6 command has no redundant dimensions; decoding derives
and checks output shape and dtype from the two inputs. The CPU handler requires
finite integral in-range indices and copies the selected rows exactly.

The generated MSL assigns one output scalar to each thread, reads int64 token
IDs, and gathers float16 weights. Invalid IDs are represented defensively as
zero in the device kernel; the serving request boundary must reject them
before dispatch to preserve the CPU error behavior.

# Evidence

The fixture gathers IDs `[2,0]` from a three-row table and verifies output rows
`[5,6]` and `[1,2]`. FX lowering and schedule-v6 round trip have zero opaque
commands, and:

```sh
ninja -f ninja.build test embedding-smoke
```

compiles the generated source with Xcode Metal. Offline replanning against the
saved single safetensors archive, without loading the model, produced:

```text
planned 1115 FX nodes into 835 IR nodes
valid serving package: 6 kernels, 835 commands, 11 opaque, tensor-store=241
```

The embedding is now typed. Every remaining opaque command belongs to
position/mask construction (5 arange, 2 advanced getitem, diff, cumsum, and
new_ones) except one framework logging side effect.

# Exact boundary

This slice compiles but does not dispatch the embedding kernel. Native command
interpretation and complete generated-kernel coverage remain open. No
model/device process, parity measurement, generation, cache request, needle
request, or ERS run was performed.
