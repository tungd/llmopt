---
type: Experiment
title: 'Native OCaml execution of the binary Q8 schedule'
description: 'Replace manual fixture dispatch with package-command interpretation and cached Metal pipeline states.'
tags: [experiment, ocaml, metal, runtime, schedule, q8, binary-archive]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T20:40:37Z' }
sources:
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Native package schedule executor
  - id: bindings
    resource: /native/ocaml_metal_stubs.m
    title: Metal buffers, pipeline cache, and Q8 dispatch bindings
  - id: smoke
    resource: /bin/ocaml_metal_smoke.ml
    title: Schedule-driven Q8 fixture probe
  - id: package
    resource: /_build/q8-serving-example/package.llmopt
    title: Binary Q8 serving fixture package
---

# Runtime path

`Metal_runtime.execute` consumes `Serving_package.schedule` directly. It binds
runtime inputs by declared name, resolves tensor-store inputs to retained views
of the mapped `weights.llmopt`, validates exact byte lengths from schedule
shape/dtype metadata, allocates output storage, dispatches Q8 linear commands,
and returns named output buffers.

The same interpreter supports explicit equal-sized buffer copies, allocation,
zero-copy view/reshape/unsqueeze aliases, and same-dtype casts. Unsupported
commands return the exact node ID and operation rather than being skipped.
These additional paths are implementation state, not device evidence from this
probe.

The native library now owns a pipeline-state cache keyed by Metal function
name. Repeated dispatches no longer reconstruct
`MTLComputePipelineState` for every command. Command submission remains
synchronous and one-command-at-a-time.

# Probe

The complete probe was fixed before touching the device: load one 4 KiB binary
fixture archive, supply one 2x4 float16 runtime tensor, execute the six-command
package, compare all six output half bit patterns, and stop on any mismatch.
Immediately before launch, `memory_pressure -Q` reported 61% system-wide free
memory and no model process was active.

One execution produced:

```text
device: Apple M4 Pro
stage: serving
kernel: llmopt_q8_linear
shape: [2, 3, 4]
output: [3.5, 8.0, 1.0, 1.5, 4.0, 2.0]
dispatch: ocaml-metal-schedule
```

The output matched all expected float16 bit patterns exactly. The standard
offline suite also passes 29 Python tests, OCaml tests, and binary package
validation.

# Boundary

This proves native execution of the Q8 fixture's input, static binding,
allocation, dispatch, and output commands against the replacement binary
archive. It does not execute either 350M package. Pointwise, non-identity
casts, transpose/index/expand/concat/contiguous, matmul/FP16 linear,
normalization, convolution, attention, embedding, mask construction,
recurrent-state kernels, batched command submission, physical Q8 KV, and the
serving loop remain outside this evidence.
