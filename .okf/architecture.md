---
type: Architecture
title: 'Dynamo/FX compiler with an OCaml Metal serving runtime'
description: 'PyTorch Dynamo supplies FX graphs, OCaml plans and emits Metal, and the intended OCaml serving runtime owns prefix/KV state and dispatch.'
tags: [architecture, pytorch, fx, ocaml, effects, metal, serving, radix-cache]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T16:03:35Z' }
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
  - id: local-kv-cache
    resource: /lib/kv_cache.ml
    title: OCaml KV format, accounting, and slot allocator
  - id: local-radix-cache
    resource: /lib/radix_cache.ml
    title: OCaml compressed radix prefix cache
  - id: local-serving-cache
    resource: /lib/serving_cache.ml
    title: OCaml serving-cache owner
  - id: sglang-radix-cache
    resource: https://github.com/sgl-project/sglang/blob/d1af3c89233c475fc1bf11939d86787e6cddd58c/python/sglang/srt/mem_cache/radix_cache.py
    title: SGLang RadixCache reference revision
  - id: sglang-mamba-radix-cache
    resource: https://github.com/sgl-project/sglang/blob/d1af3c89233c475fc1bf11939d86787e6cddd58c/python/sglang/srt/mem_cache/mamba_radix_cache.py
    title: SGLang hybrid recurrent-cache reference revision
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
| serving prefix lookup, eviction, and cache ownership | OCaml serving runtime |
| serving KV format policy and slot allocation | OCaml serving runtime |

The last two rows are implemented as the native cache layer. The Python bridge
still owns executable Metal loading and command submission until those rows are
moved into the OCaml serving process.

# Current scope

The cross-language planner supports static matrix-like placeholders, `linear`,
`mm`/`matmul`, `add`, `relu`, and `gelu`, and preserves other FX nodes as
opaque plan nodes. The first runtime optimization pass returns the captured FX
GraphModule directly, removing per-node `torch.fx.Interpreter` dispatch while
the complete LFM2.5 forward still runs through PyTorch MPS. Short-convolution
and GQA are therefore executed by PyTorch rather than custom OCaml effects in
this slice. The model-shaped compiler fixture and model-level MPS loader now
default to Q8 weight-only linear lowering. For graphs containing the generated
Q8 library, the loader compiles MSL to AIR/metallib. `native` selects the
float16/float32 Phase 2 tiled entry points; `exact` selects generated f16/f32
dequantization and then the same PyTorch MPS `linear` operation used by the
fallback, preserving model-logit parity. Unsupported dtypes or unavailable
bridge builds use the PyTorch dequantizing operator as fallback.

The OCaml serving cache is now a mandatory part of the intended runtime design,
not an optional execution mode. It implements compressed radix edges, separate
namespaces, protected-prefix leases, LRU leaf eviction, and an owned KV slot
pool. LFM2.5's recurrent ShortConv state is represented by checkpoints only at
materialized radix nodes: splitting an edge does not synthesize recurrent
state, so prefix matching falls back to the deepest valid checkpoint. This
adapts the corresponding behavior from SGLang's radix and hybrid/Mamba cache
implementations.[^sglang-radix-cache] [^sglang-mamba-radix-cache]

The KV layout accepts FP16 or grouped Q8 and defaults to Q8 at the serving
configuration boundary. Current Q8 support defines typed policy, capacity, and
byte accounting (int8 values plus FP16 group scales); physical Metal buffers
and quantize/dequantize kernels are not connected to the cache in this slice.

The first non-tile-aligned device probe exposed a partial-threadgroup launch
bug in the bridge: the 3x29 probe returned a numerical mismatch before the
launch grid was rounded to full 16x16 tiles. The corrected half/float32 probes
pass. Phase 2 now emits aligned `half4`/`float4` and `char4` cooperative loads,
with an ordered reduction and safe Metal FP32 compilation flags. The combined
350M differential probe recorded 92 generated exact-mode dispatches with
bit-exact eager logits, plus 92 generated native Phase 2 dispatches whose
separate numerical result was `max_abs=0.078125`; both paths are recorded in
[exp-0008](experiments/exp-0008-metal-runtime-q8.md).

[^pytorch-backend-contract]: PyTorch custom backend documentation.
[^local-python-backend]: `python/llmopt_backend/__init__.py` in this repository.
[^local-ocaml-planner]: `lib/fx_plan.ml` in this repository.
[^sglang-radix-cache]: SGLang `RadixCache`, pinned to revision `d1af3c89233c475fc1bf11939d86787e6cddd58c`.
[^sglang-mamba-radix-cache]: SGLang `MambaRadixCache`, pinned to the same revision.
