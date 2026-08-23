---
type: Experiment
title: 'Native LFM pointwise dispatch'
description: 'Add package ABI v4 pointwise entries and execute the exact scalar and broadcast operation family required by the preserved LFM plans.'
tags: [experiment, ocaml, metal, runtime, pointwise, binary-package]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T21:30:47Z' }
sources:
  - id: emitter
    resource: /lib/metal.ml
    title: Generated pointwise kernel family
  - id: package
    resource: /lib/serving_package.ml
    title: Package ABI v4 codec
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Native pointwise dispatch and parameter packing
  - id: probe
    resource: /bin/ocaml_metal_primitives_smoke.ml
    title: Exact expanded native probe
---

# Captured operation family

The optimized preserved prefill and decode plans require nine pointwise forms
before reaching the first transformer block:

- float16 add and multiply, including tensor broadcasting;
- int64 add with scalar offsets;
- float32 multiply with scalar rotary scales;
- int64 less-equal producing bool masks;
- float16 negate and SiLU;
- float32 cosine and sine.

Package ABI v4 assigns these entry points the typed `Pointwise` operation and
retains read compatibility with ABI v2 and v3. Runtime lookup includes the
entry-point name because several operations share the same input/output dtype
pair.

# Runtime encoding

Binary kernels accept either scalar side and row-major broadcasting through
rank eight. One fixed-width 136-byte parameter block carries the output count,
rank, scalar flags, three padded shape vectors, and integer and float scalar
representations. Unary kernels receive only their element count. The serving
command stream and all parameter blocks remain binary.

# Fixed probe

The direct OCaml fixture contains 81 commands and 25 declared kernels. Its
4,222-byte package and 108,567-byte metallib occupy 112,789 bytes. Before the
only device launch, the system reported 58% free memory and no LFM, PyTorch, or
llmopt model process. Apple M4 Pro execution reported:

```text
device: Apple M4 Pro
dispatch: binary-schedule
commands: 81
kernels: 24
outputs: 24 exact
```

The nine added outputs check scalar arithmetic, two-dimensional broadcasting,
boolean comparison, unary signs, SiLU at zero, and the exact cosine/sine values
at zero byte-for-byte.

# Boundary

This was not a 350M model run. The preserved model packages remain ABI v2 and
have not yet been replanned to declare the ABI-v4 kernels. Tensor movement,
reduction, recurrent-state materialization, float16 vocabulary projection, and
batched command submission remain outside this probe.
