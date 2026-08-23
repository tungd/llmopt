---
type: Experiment
title: 'Native LFM movement dispatch'
description: 'Add package ABI v5 movement entries and execute every dense tensor-movement family required by the preserved LFM plans.'
tags: [experiment, ocaml, metal, runtime, movement, binary-package]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T21:58:38Z' }
sources:
  - id: emitter
    resource: /lib/metal.ml
    title: Generated movement kernel family
  - id: package
    resource: /lib/serving_package.ml
    title: Package ABI v5 codec
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Native movement dispatch and parameter packing
  - id: probe
    resource: /bin/ocaml_metal_primitives_smoke.ml
    title: Exact expanded native probe
  - id: prefill
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi5-movement-2026-08-24/prefill/package.llmopt
    title: Preserved prefill ABI-v5 package
  - id: decode
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi5-movement-2026-08-24/decode/package.llmopt
    title: Preserved decode ABI-v5 package
---

# Captured operation family

The binary-input replans contain these materialized movement commands:

| Command | Prefill | Decode |
|---|---:|---:|
| transpose | 45 | 45 |
| static index | 91 | 91 |
| expand | 14 | 14 |
| concat | 13 | 25 |
| roll | 0 | 10 |
| contiguous alias | 22 | 22 |

All observed concat commands have exactly two inputs. The runtime also treats
view, reshape, and unsqueeze as dense metadata aliases.

# Binary ABI and kernels

Package ABI v5 assigns eleven exact entry points to the typed `Movement`
operation and retains ABI-v2/v3/v4 reads. Generated MSL covers
float16/float32 transpose, float16/float32/int64 static index,
float16/float32/bool expand, float16/float32 two-input concat, and float16
roll. Every materializing kernel writes a dense row-major output and supports
rank eight.

Fixed-width binary parameter blocks carry ranks, axes, padded shapes, normalized
slice starts/steps, concat partition width, and roll shift. There is no JSON in
the compiler or runtime path.

# Fixed probe

The direct OCaml fixture contains 118 commands and 36 declared kernels. Its
6,223-byte package and 160,230-byte metallib occupy 166,453 bytes. Before the
only device launch, the system reported 7.67 GB (29.8%) reclaimable memory and
no model or benchmark process. Apple M4 Pro execution reported:

```text
device: Apple M4 Pro
dispatch: binary-schedule
commands: 118
kernels: 35
outputs: 36 exact
```

The twelve added outputs cover both transpose dtypes, all three index dtypes
including negative steps and new axes, all three expand dtypes, both concat
dtypes, float16 roll, and contiguous aliasing. Every output is compared
byte-for-byte.

# Preserved packages

Offline compilation from `graph.llmopt` produced JSON-free ABI-v5 artifacts:

| Graph | Commands | Kernels | Opaque | Tensor bindings |
|---|---:|---:|---:|---:|
| Prefill | 872 | 37 | 0 | 241 |
| Decode | 926 | 33 | 0 | 241 |

Both MSL programs compile with Xcode Metal, both packages validate against hard
links to the same 422,137,216-byte binary tensor archive, and neither operation
loaded the model or executed its full schedule.

# Boundary

The fixed probe establishes native movement semantics, not full-model parity or
performance. Reduction, functional recurrent-state update, the final float16
vocabulary projection, batched submission, physical KV storage, and request
serving remain open.
