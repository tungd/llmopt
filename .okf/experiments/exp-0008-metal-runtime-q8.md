---
type: Experiment
title: 'Generated Q8 Metal library loading, vectorized dispatch, and differential probe'
description: 'Load the OCaml-emitted Q8 metallib through a PyTorch MPS bridge, exercise the Phase 2 vectorized entry points, and compare eager, fallback FX, and generated FX on LFM2.5-350M.'
tags: [experiment, runtime, metal, mps, q8, tiling]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-21T09:24:27Z' }
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
  - id: differential
    resource: /python/examples/metal_runtime_differential.py
    title: one-shot eager/fallback/generated differential probe
  - id: pytorch-kernel-api
    resource: https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/mps/OperationUtils.mm
    title: PyTorch MetalKernelFunction implementation
---

# Question

Can the OCaml-emitted Q8 MSL be compiled into a Metal library, loaded through
PyTorch's MPS runtime, and launched with a tiled Q8 weight-only linear ABI?

# Implementation

The OCaml emitter now produces shape-parameterized 16x16 cooperative kernels
for both float16 and float32 activations. Phase 2 uses aligned `half4`/`float4`
input loads and `char4` weight loads for each tile, with scalar bounds-safe
loads for unaligned tails and a source-order scalar reduction. It also emits
f16/f32 dequantization kernels. The Python backend compiles `kernel.metal` to
AIR and then `.metallib` when the Ninja-built C++ bridge is available, using
safe Metal FP32 math and recording the compiler flags in a cache stamp. The
bridge caches the library and selects either the native tiled kernel or the
exact generated-dequantization path, binds contiguous MPS tensors for signed
int8 weights, float16 scales, optional bias, and output, then submits the work
on PyTorch's current MPS stream.

The launch grid is rounded up to complete 16x16 threadgroups. The kernel still
uses `m`, `n`, and `k` bounds from its parameter block, so padded threads only
participate in cooperative loads and do not store out-of-range outputs.

# Static verification

```sh
ninja -f ninja.build metal-runtime q8-fx-smoke test
```

The native extension compiled, the Q8 AIR and `.metallib` artifacts linked,
the OCaml suite and Python suite passed 20 tests, and the generated MSL/LLVM
checks completed. Python byte-compilation and `git diff --check` also passed.
The differential target is `metal-runtime-differential`.

# Device observation

Before the one combined launch probe, `memory_pressure -Q` reported 61%
system-wide free memory on the 25.8 GB host. The direct part used `M=3`,
`N=29`, and `K=37` so all three dimensions exercised non-aligned boundaries
and ran both input dtypes through the generated native vectorized library:

```sh
PYTHONPATH=python python3.13 python/examples/metal_runtime_smoke.py \
  --library _build/q8-fx-example/kernel.metallib
```

The library loaded and both dispatches returned:

```text
{"dispatch": "generated-metal-q8-native-vectorized-tiled", "errors": {"float16": {"max_abs": 0.0, "mean_abs": 0.0, "exact": true, "argmax_exact": true, "dispatches": 1}, "float32": {"max_abs": 4.76837158203125e-06, "mean_abs": 4.2870811967077316e-07, "exact": false, "argmax_exact": true, "dispatches": 1}}, "shape": [3, 29, 37]}
```

The original observed failure was explained by the launch shape: PyTorch's
`MetalKernelFunction::dispatch` maps the requested dimensions directly to
`dispatchThreads`, so a non-aligned grid creates partial final threadgroups.
The cooperative loads require every 16x16 lane, but the partial groups do not
execute all lanes. The bridge correction rounds the grid to full tiles while
retaining kernel bounds checks.[^pytorch-kernel-api]

The combined differential model probe then ran through the Dynamo/FX backend:

```sh
ninja -f ninja.build metal-runtime-differential
```

The model loaded once, the OCaml planner processed 1,115 FX nodes, and the
graph artifacts contained 92 `llmopt.q8_linear` nodes, all with float32
activations. The differential artifact reports zero native dispatches for
compiled fallback, 92 generated exact-mode dispatches, and 92 generated native
Phase 2 dispatches. Eager versus fallback and eager versus generated exact-mode
output were bit-exact:

```text
eager_vs_generated_exact: max_abs=0 mean_abs=0 exact=true argmax_exact=true
```

The native Phase 2 matmul remains a separate numerical optimization path:

```text
eager_vs_generated_native: max_abs=0.078125 mean_abs=0.00713115930557251 exact=false argmax_exact=true
```

# Boundary

The runtime loading path, dual-dtype ABI, vectorized Phase 2 emitter, exact
generated-dequantization path, static build path, and direct device dispatch
are implemented. The model-level exact-logit comparison now passes with 92
generated exact-mode dispatches. Native Phase 2 is wired and exercised, but its
float32 matmul reduction does not yet match MPS logits; its numerical lowering
remains an optimization research question. The probe produced no ERS score.

[^pytorch-kernel-api]: PyTorch's official implementation uses `dispatchThreads`
with the supplied grid dimensions and threadgroup size, and binds tensor
arguments through `mtl_setBuffer`.
