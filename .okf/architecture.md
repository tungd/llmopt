---
type: Architecture
title: 'Dynamo/FX compiler with an OCaml Metal serving runtime'
description: 'PyTorch Dynamo supplies FX graphs, OCaml plans and emits Metal, and the intended OCaml serving runtime owns prefix/KV state and dispatch.'
tags: [architecture, pytorch, fx, ocaml, effects, metal, serving, radix-cache]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T18:53:41Z' }
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
  - id: local-package-abi
    resource: /lib/serving_package.ml
    title: Versioned serving-package representation and validator
  - id: local-serving-schedule
    resource: /lib/serving_schedule.ml
    title: Binary typed command schedule
  - id: local-metal-runtime
    resource: /native/metal_runtime.cpp
    title: PyTorch MPS Metal library bridge
  - id: local-runtime-loader
    resource: /python/llmopt_backend/metal_runtime.py
    title: generated metallib loader and fallback boundary
  - id: local-ocaml-runtime
    resource: /lib/metal_runtime.ml
    title: Native OCaml package loader and typed Q8 dispatch
  - id: local-safetensors
    resource: /lib/safetensors.ml
    title: Safetensors metadata parser and tensor index
  - id: local-ocaml-stubs
    resource: /native/ocaml_metal_stubs.m
    title: Metal device, library, shared-buffer, and command bindings
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
names, op kinds, targets, references, static shape metadata from `val`,
`tensor_meta`, or Dynamo `example_value`, dtypes, lossless typed arguments, and
an explicit runtime-input or tensor-store binding. Dynamo's lifted-static-input
metadata identifies model parameters and buffers; `get_attr` nodes are static
by construction. The
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
| compiled graph package manifest and artifact validation | OCaml compiler and Ninja |
| tensor archive indexing, mapping, and Metal buffer views | OCaml runtime plus Objective-C Metal bindings |
| direct FX execution and device dispatch | Python FX GraphModule plus PyTorch MPS |
| generated Q8 library loading and tensor binding | Python loader plus PyTorch MPS C++ bridge |
| custom Metal buffers and command submission | PyTorch MPS stream through the bridge |
| standalone package loading and Q8 Metal dispatch | OCaml runtime plus Objective-C Metal bindings |
| serving prefix lookup, eviction, and cache ownership | OCaml serving runtime |
| serving KV format policy and slot allocation | OCaml serving runtime |

The cache rows and standalone OCaml Metal primitives are implemented. The
model-level execution path still uses the Python/PyTorch bridge because the
complete LFM2.5 Q8 archive and schedule have not moved into the OCaml process.

# Current scope

The cross-language planner supports N-dimensional placeholders, `linear`,
Q8 linear, `mm`/`matmul`, pointwise arithmetic/comparison and common unary
operators, mean, casts, views, reshape, transpose, unsqueeze, expand,
contiguous, normalized static indexing, concat, `relu`, and `gelu`, and
preserves other FX nodes as opaque plan
nodes. A pure pass fuses the LFM RMSNorm chain into one typed command, and its
float32-to-float16 and float16 Metal kernels compile with Xcode. The first
runtime optimization pass returns the captured FX
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

The FX compiler emits a versioned binary `package.llmopt` containing the typed
command stream, N-dimensional value shapes, operator arguments, kernel ABI,
cache policy, metallib path, and optional tensor-store path. The copied
`fx.json`, pretty-printed `plan.txt`, MSL, and LLVM IR are diagnostics and are
not referenced by the serving runtime. Kernel entry points carry an operation,
input/output dtype, and threadgroup shape. A compiled-graph package has no
tensor store; `--tensor-store weights.safetensors` emits a serving package that
references exactly one binary tensor archive. Tensor payload metadata remains
authoritative in that archive rather than being split into per-tensor files.

When a captured graph contains static tensors, the Python adapter streams them
into `weights.safetensors` one at a time. This bounds CPU staging to the current
tensor instead of retaining a second whole-model CPU copy. The real 350M Q8
capture emitted 241 tensor keys totaling 422,104,704 payload bytes. The OCaml
package validator resolves every binary-schedule tensor binding against archive
dtype and N-dimensional shape before runtime loading.

The Ninja-built OCaml runtime consumes that binary package directly, validates
the metallib, tensor archive, schedule bindings, and declared entry points,
parses the safetensors index,
maps the archive once, and creates retained Metal views at tensor byte offsets.
A standalone 2x3x4 FP32/Q8 fixture bound its int8 weight, FP16 scale, and FP16
bias from the mapped archive; `llmopt_q8_linear_f32` returned
`[3.5, 8, 1, 1.5, 4, 2]` exactly on Apple M4 Pro. This proves the binary
archive-to-command boundary, not complete model execution. A later minimal
probe repeated the exact result from a directory containing only
`package.llmopt`, `kernel.metallib`, and `weights.safetensors`, proving that
neither FX JSON nor the textual plan is part of native startup.

The memory-bounded manifest-v2 recapture now reaches package generation. Its
1,115 FX nodes initially became 835 schedule commands: 793 typed and 42 opaque,
compared with 379 typed and 736 opaque in the saved v1 package. A subsequent
offline replan corrected a target-suffix collision and moved all 14 expand
commands into the typed set. Typed depthwise ShortConv lowering then moved the
ten model `conv1d` commands, and masked-attention lowering moved all six SDPA
commands. Embedding lowering moves the token lookup as well, leaving 824 typed
and 11 opaque. Direct-FX
execution is bit exact against eager MPS for the six-token probe. This is
capture and planning evidence; PyTorch MPS, not the OCaml package runtime,
executed the parity check.

The source graph measures 85 getitem, 10 chunk, and 13 concat nodes. For v2,
the planner now holds chunk partitions as compile-time descriptors and
emits normalized slices directly at integer getitem consumers, avoiding a
tuple-valued runtime command. Static tensor indices and concat survive the
schedule-v6 binary round trip and CPU interpretation. The latest offline
package structurally validates with 241 tensors and declares six generated
kernels; native Metal emission and command dispatch for the complete schedule
remain open.

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
