---
type: Experiment
title: 'Generated Q8 Metal library loading and tiled dispatch'
description: 'Load the OCaml-emitted Q8 metallib through a PyTorch MPS bridge and validate a non-tile-aligned launch without starting a model benchmark.'
tags: [experiment, runtime, metal, mps, q8, tiling]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-21T08:33:08Z' }
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
  - id: pytorch-kernel-api
    resource: https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/mps/OperationUtils.mm
    title: PyTorch MetalKernelFunction implementation
---

# Question

Can the OCaml-emitted Q8 MSL be compiled into a Metal library, loaded through
PyTorch's MPS runtime, and launched with a tiled Q8 weight-only linear ABI?

# Implementation

The OCaml emitter now produces a shape-parameterized 16x16 cooperative kernel.
The Python backend compiles `kernel.metal` to AIR and then `.metallib` when the
Ninja-built C++ bridge is available. The bridge caches the library and kernel,
binds contiguous MPS tensors for FP16 activations, signed int8 weights,
float16 scales, optional bias, and output, then submits the work on PyTorch's
current MPS stream.

The launch grid is rounded up to complete 16x16 threadgroups. The kernel still
uses `m`, `n`, and `k` bounds from its parameter block, so padded threads only
participate in cooperative loads and do not store out-of-range outputs.

# Static verification

```sh
ninja -f ninja.build metal-runtime q8-fx-smoke test
```

The native extension compiled, the Q8 AIR and `.metallib` artifacts linked,
the OCaml and Python suites passed 18 tests, and the generated MSL/LLVM checks
completed. No LFM2.5-350M or LFM2.5-2.6B process was launched for this slice.

# Device observation

Before the launch probe, `memory_pressure -Q` reported 63% system-wide free
memory. The single non-model probe used `M=3`, `N=29`, and `K=37` so all three
dimensions exercised non-aligned boundaries:

```sh
PYTHONPATH=python python3.13 python/examples/metal_runtime_smoke.py \
  --library _build/q8-fx-example/kernel.metallib
```

The library loaded and the dispatch returned, but the parity assertion failed:

```text
AssertionError: Tensor-likes are not close!
Mismatched elements: 87 / 87
Greatest absolute difference: 17.40625 at index (2, 28)
Greatest relative difference: 18640.0
```

The observed failure is explained by the launch shape: PyTorch's
`MetalKernelFunction::dispatch` maps the requested dimensions directly to
`dispatchThreads`, so a non-aligned grid creates partial final threadgroups.
The cooperative loads require every 16x16 lane, but the partial groups do not
execute all lanes. The bridge correction rounds the grid to full tiles while
retaining kernel bounds checks.[^pytorch-kernel-api]

# Boundary

The runtime loading path, ABI, and static build path are implemented. The
captured device observation is a failed pre-correction numerical probe; it is
not a model result, speed result, or post-correction parity claim. The probe
was not rerun in this slice, and no model-level benchmark was launched.

[^pytorch-kernel-api]: PyTorch's official implementation uses `dispatchThreads`
with the supplied grid dimensions and threadgroup size, and binds tensor
arguments through `mtl_setBuffer`.
