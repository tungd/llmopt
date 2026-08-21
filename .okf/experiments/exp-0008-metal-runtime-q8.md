---
type: Experiment
title: 'Generated Q8 Metal library loading and tiled dispatch'
description: 'Load the OCaml-emitted Q8 metallib through a PyTorch MPS bridge, exercise half and float32 entry points, and record the bounded 350M FX integration result.'
tags: [experiment, runtime, metal, mps, q8, tiling]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-21T09:00:28Z' }
sources:
  - id: emitter
    resource: /lib/metal.ml
    title: OCaml-generated tiled Q8 Metal source
  - id: bridge
    resource: /native/metal_runtime.cpp
    title: PyTorch MPS library loading and dispatch bridge
  - id: loader
    resource: /python/llmopt_backend/metal_runtime.py
    title: Python metallib compilation and activation boundary
  - id: smoke
    resource: /python/examples/metal_runtime_smoke.py
    title: small non-model MPS dispatch probe
  - id: model-smoke
    resource: /bench/lfm25_mps.py
    title: bounded 350M FX integration probe
  - id: pytorch-kernel-api
    resource: https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/mps/OperationUtils.mm
    title: PyTorch MetalKernelFunction implementation
---

# Question

Can the OCaml-emitted Q8 MSL be compiled into a Metal library, loaded through
PyTorch's MPS runtime, and launched with a tiled Q8 weight-only linear ABI?

# Implementation

The OCaml emitter now produces shape-parameterized 16x16 cooperative kernels
for both float16 and float32 activations. The Python backend compiles
`kernel.metal` to AIR and then `.metallib` when the Ninja-built C++ bridge is
available. The bridge caches the library and selects the matching kernel,
binds contiguous MPS tensors for signed int8 weights, float16 scales, optional
bias, and output, then submits the work on PyTorch's current MPS stream.

The launch grid is rounded up to complete 16x16 threadgroups. The kernel still
uses `m`, `n`, and `k` bounds from its parameter block, so padded threads only
participate in cooperative loads and do not store out-of-range outputs.

# Static verification

```sh
ninja -f ninja.build metal-runtime q8-fx-smoke test
```

The native extension compiled, the Q8 AIR and `.metallib` artifacts linked,
the OCaml and Python suites passed 19 tests, and the generated MSL/LLVM checks
completed. The model integration target was added as
`metal-runtime-model-smoke`.

# Device observation

Before the launch probe, `memory_pressure -Q` reported 65% system-wide free
memory. The non-model probe used `M=3`, `N=29`, and `K=37` so all three
dimensions exercised non-aligned boundaries and ran both input dtypes:

```sh
PYTHONPATH=python python3.13 python/examples/metal_runtime_smoke.py \
  --library _build/q8-fx-example/kernel.metallib
```

The library loaded and both dispatches returned:

```text
{"dispatch": "generated-metal-q8-tiled", "max_abs_error": {"float16": 0.0078125, "float32": 2.86102294921875e-06}, "shape": [3, 29, 37]}
```

The observed failure is explained by the launch shape: PyTorch's
`MetalKernelFunction::dispatch` maps the requested dimensions directly to
`dispatchThreads`, so a non-aligned grid creates partial final threadgroups.
The cooperative loads require every 16x16 lane, but the partial groups do not
execute all lanes. The bridge correction rounds the grid to full tiles while
retaining kernel bounds checks.[^pytorch-kernel-api]

The bounded model integration then ran through the Dynamo/FX backend:

```sh
ninja -f ninja.build metal-runtime-model-smoke
```

Before model launch, `memory_pressure -Q` reported 58% system-wide free
memory on a 25.8 GB host. The model loaded, the OCaml planner processed 1,115
FX nodes, and the graph artifact contained 92 `llmopt.q8_linear` nodes, all
with float32 activations. Its `runtime.json` selected `generated-metal-q8`,
but the existing exact-logit comparison failed:

```text
RuntimeError: llmopt MPS output differs from eager MPS: max_abs=0.03515625 mean_abs=0.004932403564453125
```

# Boundary

The runtime loading path, dual-dtype ABI, static build path, and direct device
dispatch are implemented. The bounded model probe confirms the FX artifact
selects the generated library but does not satisfy the repository's exact-logit
check; it produced no ERS score. The model-level logit-drift versus token-ID
parity question remains open, and Phase 2 optimization is not recorded here.

[^pytorch-kernel-api]: PyTorch's official implementation uses `dispatchThreads`
with the supplied grid dimensions and threadgroup size, and binds tensor
arguments through `mtl_setBuffer`.
