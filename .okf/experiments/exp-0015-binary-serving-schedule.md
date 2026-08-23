---
type: Experiment
title: 'Versioned binary serving schedule'
description: 'Preserve rank and FX arguments in typed commands, then load the Metal fixture without JSON or a textual plan.'
tags: [experiment, ocaml, fx, binary, schedule, metal, serving]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T18:05:24Z' }
sources:
  - id: shape
    resource: /lib/tensor_shape.ml
    title: N-dimensional logical shape model
  - id: schedule
    resource: /lib/serving_schedule.ml
    title: Versioned typed command codec
  - id: package
    resource: /lib/serving_package.ml
    title: Binary package codec and runtime metadata
  - id: loader
    resource: /lib/metal_runtime.ml
    title: Native OCaml Metal loader
  - id: build
    resource: /ninja.build
    title: Ninja build and package checks
---

# Question

Can the compile boundary retain the rank and constants needed to lower LFM
operators while the native serving path stops consuming `fx.json` and
`plan.txt`?

# Implementation

FX manifest v2 encodes node references, integers, floats, booleans, nulls,
strings, symbols, lists, tuples, mappings, slices, and keyword arguments as a
typed tree. OCaml converts that tree to IR arguments referencing typed values.
Each value owns an N-dimensional logical shape while retaining a checked 2-D
tile projection for the current kernels.

`package.llmopt` starts with the `LLMOPTPK` magic and a version field. It embeds
the `LLMOSCH` command stream, including value IDs, dtype, full rank, operation
arguments, kernel entries, cache policy, tensor-store bindings, and the
metallib path. The FX snapshot, pretty-printed plan, MSL, and LLVM text remain
compiler diagnostics and are not package dependencies.

# Static evidence

The Python suite contains 25 tests. The new manifest test preserves a rank-3
`view` plus integer dimensions and tuple/slice indexing. OCaml tests round-trip
that graph through the binary schedule while retaining `[2,3,4]` and the
opaque command. Ninja validates these generated packages:

| Fixture | Package bytes | Commands | Opaque | Tensor bindings |
|---|---:|---:|---:|---:|
| FP32 linear | 307 | 5 | 0 | 0 |
| Q8 linear | 538 | 6 | 0 | 0 |
| Q8 serving | 562 | 6 | 0 | 3 |

# Device evidence

Before launch, `memory_pressure -Q` reported 56% system-wide memory free on a
25,769,803,776-byte Apple Silicon host. The one probe copied exactly three
files into an otherwise empty directory:

* `package.llmopt`;
* `kernel.metallib`; and
* `weights.safetensors`.

The native OCaml process selected Apple M4 Pro, dispatched
`llmopt_q8_linear_f32`, and returned `[3.5, 8, 1, 1.5, 4, 2]` exactly. No model
was loaded and neither JSON nor the textual plan was present.

# Saved-model offline replan

Without loading the model or launching Metal, the new compiler replanned the
saved LFM2.5-350M Q8 FX snapshot. `llmopt-package-check` accepted a
116,861-byte `package.llmopt` containing 1,115 commands, 736 opaque commands,
and 241 tensor bindings against the existing 422,144,400-byte safetensors
archive. The archive was hard-linked for the check, so this did not create a
second 422 MB payload copy.

# Remaining boundary

The binary stream faithfully represents opaque commands; it does not make
them executable. The saved real 350M capture still records 736 opaque
operations and predates manifest v2 argument capture. Rank-aware operator
lowering, prefill and decode schedules, KV/checkpoint commands, and native
command interpretation remain open.
