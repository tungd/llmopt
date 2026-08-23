---
type: Experiment
title: 'Typed LFM ShortConv lowering and Metal kernel'
description: 'Lower the exact LFM2.5 depthwise prefill convolution into typed OCaml IR, a versioned binary command, a CPU reference, and generated MSL.'
tags: [experiment, lfm25, shortconv, conv1d, ocaml, metal, schedule]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T19:14:20Z' }
sources:
  - id: shape
    resource: /lib/tensor_shape.ml
    title: Depthwise conv1d shape validation
  - id: planner
    resource: /lib/fx_plan.ml
    title: LFM conv1d FX lowering
  - id: cpu
    resource: /lib/cpu.ml
    title: Direct-style CPU reference handler
  - id: metal
    resource: /lib/metal.ml
    title: Generated ShortConv MSL
  - id: replan
    resource: /_artifacts/lfm25-350m-q8-v4-short-conv-serving-replan-2026-08-24/package.llmopt
    title: Offline serving-package replan
---

# Targeted form

All ten opaque model convolutions have the same LFM prefill form:

```text
input  [1,1024,6] f16
weight [1024,1,3] f16
bias   none
stride 1, padding 2, dilation 1, groups 1024
output [1,1024,8] f16
```

This is a depthwise convolution: each hidden channel owns one three-tap
filter. The compiler models this form directly rather than accepting unrelated
conv1d configurations through an under-specified operation.

# Implementation

`Tensor_shape.depthwise_conv1d` validates rank, group/channel relationships,
kernel shape, parameters, overflow, and inferred output width. An abstract
`Ir.Short_conv.t` owns the positive stride, padding, dilation, and group
parameters. The FX planner accepts only the no-bias float16 depthwise form and
otherwise retains the existing opaque fallback.

Schedule version 4 encodes the ShortConv parameters as fixed-width integers,
reads versions 1 through 4, and re-infers shape/dtype metadata when decoding.
The CPU effect handler implements the padded grouped convolution over flattened
N-dimensional storage.

The MSL kernel assigns one output element to each grid thread, accumulates the
three taps in float32, and stores float16. Its ABI entry uses a 256x1x1
threadgroup. This is the first correct scalar kernel boundary; tiling and
vectorization remain subsequent optimization passes.

# Evidence

The reference test computes two channels with distinct filters and checks all
12 output values exactly. Synthetic FX lowering has zero opaque commands,
schedule-v4 round-trip validation passes, and:

```sh
ninja -f ninja.build test short-conv-smoke
```

compiles the emitted source with Xcode Metal.

Replanning the saved real manifest and reusing its single safetensors archive
without loading the model produced:

```text
planned 1115 FX nodes into 835 IR nodes
valid serving package: 4 kernels, 835 commands, 18 opaque, tensor-store=241
```

The preceding compiler had 28 opaque commands. All ten `conv1d` operations
are now typed. The remaining 18 are 6 scaled-dot-product attention, 5 arange,
2 advanced getitem, and one each of embedding, diff, cumsum, new_ones, and a
logging side effect.

# Exact boundary

This slice compiles but does not dispatch the ShortConv kernel. The package
still lacks generated kernels and native command interpretation for much of
the typed schedule, including its Q8 linear commands. No model/device process,
parity measurement, generation, cache request, needle request, or ERS run was
performed.
