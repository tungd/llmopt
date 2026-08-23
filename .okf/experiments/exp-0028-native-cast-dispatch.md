---
type: Experiment
title: 'Native typed cast dispatch'
description: 'Add package ABI v3 cast entries and execute every conversion direction used by the captured LFM graph.'
tags: [experiment, ocaml, metal, runtime, cast, binary-package]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T21:20:37Z' }
sources:
  - id: emitter
    resource: /lib/metal.ml
    title: Generated cast kernel family
  - id: package
    resource: /lib/serving_package.ml
    title: Package ABI v3 codec
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Native cast command dispatch
  - id: probe
    resource: /bin/ocaml_metal_primitives_smoke.ml
    title: Exact expanded native probe
---

# ABI and kernels

Package ABI v3 assigns a typed `Cast` kernel operation while retaining read
compatibility with package ABI v2. The generated family contains the three
non-identity conversions present in the preserved prefill/decode schedules:

- float16 to float32;
- float32 to float16;
- int64 to float32.

Identity casts remain zero-copy aliases in the OCaml interpreter. Each
non-identity command allocates its typed output, packs one uint32 element count,
and dispatches the matching declared entry point.

# Fixed probe

The direct OCaml fixture was expanded from 42 to 51 commands and from 13 to 16
declared kernels. Its package and metallib occupy 71,507 bytes. Before the only
device launch, the system reported 60% free memory and no model or Torch
process. Apple M4 Pro execution reported:

```text
device: Apple M4 Pro
dispatch: binary-schedule
commands: 51
kernels: 15
outputs: 15 exact
```

The three added outputs were compared byte-for-byte for values `1`, `-2`, and
`0.5` in both floating directions and `0`, `-3`, and `7` for int64 to float32.

# Boundary

This does not execute the 350M packages. They remain stored as readable ABI-v2
artifacts and need an offline ABI-v3 replan to declare cast kernels. Pointwise,
movement, reduction, recurrent-state, float16 vocabulary projection, and
batched submission remain outside this probe.
