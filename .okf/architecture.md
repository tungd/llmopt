---
type: Architecture
title: 'Dynamo/FX frontend with OCaml effect-based planning'
description: 'PyTorch Dynamo supplies FX graphs, OCaml plans them, and the direct FX executor can dispatch generated Q8 Metal libraries through PyTorch MPS.'
tags: [architecture, pytorch, fx, ocaml, effects, metal]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-20T11:24:21Z' }
sources:
  - id: pytorch-backend-contract
    resource: https://docs.pytorch.org/docs/2.9/torch.compiler_custom_backends.html
    title: PyTorch custom backend contract
  - id: local-python-backend
    resource: /python/llmopt_backend/__init__.py
    title: llmopt Dynamo/FX adapter
  - id: local-ocaml-planner
    resource: /lib/fx_plan.ml
    title: OCaml FX planner
  - id: local-q8-lowering
    resource: /lib/metal.ml
    title: Q8 weight-only linear lowering
  - id: local-metal-runtime
    resource: /native/metal_runtime.cpp
    title: PyTorch MPS Metal library bridge
  - id: local-runtime-loader
    resource: /python/llmopt_backend/metal_runtime.py
    title: generated metallib loader and fallback boundary
---

# Overview

The public frontend is a PyTorch `torch.compile` backend. Dynamo calls the
backend with an FX `GraphModule` and example inputs, and the backend returns a
callable with the same forward contract.[^pytorch-backend-contract]

The Python adapter serializes a deliberately small manifest containing node
names, op kinds, targets, references, static shape metadata, and dtypes. The
OCaml executable parses that manifest, lowers supported nodes by performing
typed tile effects, preserves other nodes as opaque effect actions, captures
the graph IR, with optional pure passes available for later slices, and emits
optional Metal source plus textual LLVM IR.[^local-python-backend]
[^local-ocaml-planner]

# Ownership

| Concern | Owner |
|---|---|
| Python bytecode/Dynamo and FX graph acquisition | Python adapter |
| FX manifest serialization and subprocess boundary | Python adapter |
| op support, shape checks, effect planning | OCaml planner |
| graph transforms | Pure OCaml passes |
| Metal/LLVM source generation | OCaml backends |
| direct FX execution and device dispatch | Python FX GraphModule plus PyTorch MPS |
| generated Q8 library loading and tensor binding | Python loader plus PyTorch MPS C++ bridge |
| custom Metal buffers and command submission | PyTorch MPS stream through the bridge |

# Current scope

The cross-language planner supports static matrix-like placeholders, `linear`,
`mm`/`matmul`, `add`, `relu`, and `gelu`, and preserves other FX nodes as
opaque plan nodes. The first runtime optimization pass returns the captured FX
GraphModule directly, removing per-node `torch.fx.Interpreter` dispatch while
the complete LFM2.5 forward still runs through PyTorch MPS. Short-convolution
and GQA are therefore executed by PyTorch rather than custom OCaml effects in
this slice. The model-shaped compiler fixture and model-level MPS loader now
default to Q8 weight-only linear lowering with FP16 activations. For graphs
containing the generated Q8 kernel, the loader compiles MSL to AIR/metallib and
the C++ bridge binds MPS tensors and submits a tiled launch on the current MPS
stream. Unsupported dtypes or unavailable bridge builds use the PyTorch
dequantizing operator as fallback.

The first non-tile-aligned device probe exposed a partial-threadgroup launch
bug in the bridge: the 3x29 probe returned a numerical mismatch before the
launch grid was rounded to full 16x16 tiles. That exact observation is recorded
in [exp-0008](experiments/exp-0008-metal-runtime-q8.md); no model benchmark was
launched for this runtime slice.

[^pytorch-backend-contract]: PyTorch custom backend documentation.
[^local-python-backend]: `python/llmopt_backend/__init__.py` in this repository.
[^local-ocaml-planner]: `lib/fx_plan.ml` in this repository.
