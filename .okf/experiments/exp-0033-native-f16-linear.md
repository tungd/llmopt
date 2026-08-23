---
type: Experiment
title: 'Native float16 vocabulary projection'
description: 'Execute the final LFM vocabulary projection with a SIMD-reduced Metal kernel and preserve complete fixed-shape metallibs.'
tags: [experiment, ocaml, metal, runtime, linear, vocabulary]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T22:43:43Z' }
sources:
  - id: emitter
    resource: /lib/metal.ml
    title: Generated float16 linear kernel
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Native linear parameter packing and dispatch
  - id: probe
    resource: /bin/ocaml_metal_primitives_smoke.ml
    title: Exact expanded native probe
  - id: prefill
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi6-linear-2026-08-24/prefill/package.llmopt
    title: Preserved prefill package with final projection
  - id: decode
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi6-linear-2026-08-24/decode/package.llmopt
    title: Preserved decode package with final projection
---

# Captured operation

Both preserved graphs end with the same tied float16 vocabulary weight:

```text
prefill linear[6x65536x1024] -> [1x6x65536]:f16
decode  linear[1x65536x1024] -> [1x1x65536]:f16
```

The model rewrite intentionally retains this shared weight in float16 because
the same storage backs token embedding. All other eligible transformer linear
layers use Q8 weights.

# Kernel and dispatch

`llmopt_linear_f16` uses a fixed 256-thread group containing eight SIMD groups.
Each SIMD group owns one output channel and reads its contiguous weight row
cooperatively in 32-element strides. A weight value is reused across blocks of
up to eight input rows, each lane holds float32 partial sums, `simd_sum` performs
the cross-lane reduction, and lane zero stores float16 output. The runtime sends
the binary `{m,n,k}` parameter block and rounds the launch to complete SIMD
groups.

The entry reuses the existing typed `Linear` operation, so package ABI remains
v6. Exact-name dispatch distinguishes it from the older float32 linear entry.

# Fixed probe

The direct fixture contains 129 commands and 39 declared kernels. Its
6,861-byte package and 174,942-byte metallib occupy 181,803 bytes. Before the
only device launch, system memory was 59% free and no model or torch process was
present. Apple M4 Pro execution reported:

```text
device: Apple M4 Pro
dispatch: binary-schedule
commands: 129
kernels: 38
outputs: 39 exact
```

The added two-row, three-channel projection uses integer-valued float16 inputs
and weights and checks all six outputs byte-for-byte.

# Preserved packages

Offline binary-input compilation produced:

| Graph | Commands | Kernels | Opaque | Tensor bindings |
|---|---:|---:|---:|---:|
| Prefill | 872 | 38 | 0 | 241 |
| Decode | 926 | 36 | 0 | 241 |

Both MSL programs compile, both packages validate against hard links to the
same 422,137,216-byte binary tensor archive, and neither compilation loaded the
model or dispatched the full schedule. There are no JSON files in either
artifact directory.

# Boundary

Every operation family observed in the fixed six-token prefill and one-token
decode schedules now has a generated entry and an exact fixed-fixture dispatch.
This does not establish complete-model parity: the runtime still submits and
waits one command at a time, has not bound physical persistent KV state to the
radix cache, and has not executed either full model schedule.
