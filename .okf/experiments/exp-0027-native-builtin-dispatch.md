---
type: Experiment
title: 'Native dispatch for every emitted built-in Metal family'
description: 'Exercise generic typed buffer and parameter binding through one JSON-free, multi-operation package.'
tags: [experiment, ocaml, metal, runtime, schedule, binary-package]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T20:55:02Z' }
sources:
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Typed schedule interpreter and parameter encoders
  - id: bindings
    resource: /native/ocaml_metal_stubs.m
    title: Generic Metal command encoder
  - id: fixture
    resource: /bin/native_schedule_fixture.ml
    title: JSON-free typed package generator
  - id: probe
    resource: /bin/ocaml_metal_primitives_smoke.ml
    title: Exact native output probe
---

# Runtime change

The Objective-C bridge accepts a declared function, an ordered list of retained
Metal buffers, fixed-width parameter bytes, and grid/threadgroup dimensions.
It binds those resources, obtains the cached pipeline state, submits one
command, and reports Metal completion errors to OCaml. The typed interpreter
owns operation-specific shape projection and parameter packing.

The implemented command families are matmul, fused float32 linear, float32
linear, Q8 linear, RMSNorm, depthwise ShortConv, masked attention, embedding,
arange, bool/float fill, prepended diff, bool-to-int64 cumsum, and two-index
gather. The fixture exercises every family currently emitted together except
the already-probed Q8 path and the mutually exclusive primary linear variants.

# Fixed probe

`llmopt-native-schedule-fixture` constructs the IR directly in OCaml and writes
only `package.llmopt` and generated MSL. Ninja compiles the MSL to a metallib
and validates the 42-command, zero-opaque package with 13 declared kernels.
There is no FX JSON in this path.

Before launch, `memory_pressure -Q` reported 57% free memory, no LFM/PyTorch
model process was active, and the package plus metallib occupied 59,325 bytes.
One execution on Apple M4 Pro reported:

```text
device: Apple M4 Pro
dispatch: binary-schedule
commands: 42
kernels: 12
outputs: 12 exact
```

Every output was compared byte-for-byte: embedding, descending arange, three
fill dtypes, diff, cumsum, Gather2, float32-to-float16 RMSNorm, ShortConv,
masked attention, and float32 matmul.

# Boundary

This is native execution evidence for the built-in kernel families, not for
the preserved LFM2.5-350M packages. Their pointwise, non-identity cast,
transpose/index/expand/concat/contiguous, reduction, roll/update-slice,
float16 vocabulary projection, physical Q8 KV, and batched command-buffer paths
still need implementation before a complete model schedule can execute.
