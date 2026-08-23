---
type: Architecture
title: 'Dynamo/FX compiler with an OCaml Metal serving runtime'
description: 'PyTorch Dynamo supplies FX graphs, OCaml plans and emits Metal, and the intended OCaml serving runtime owns prefix/KV state and dispatch.'
tags: [architecture, pytorch, fx, ocaml, effects, metal, serving, radix-cache]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T22:57:37Z' }
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
  - id: local-weight-archive
    resource: /lib/weight_archive.ml
    title: Versioned binary weight-archive parser and tensor index
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

The Python adapter serializes a versioned binary `graph.llmopt` containing node
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

Binary transport ABI v1 carries manifest schema v2 with recursively tagged
arguments, explicit bindings, fixed-width numeric fields, and length-prefixed
UTF-8 strings. The OCaml importer detects the `LLMOPTFX` magic and rejects
unknown versions, malformed tags, truncation, and trailing bytes. Legacy JSON
input remains readable, while `LLMOPT_FX_DIAGNOSTICS=1` explicitly emits JSON
diagnostics; neither format is read by serving.

# Ownership

| Concern | Owner |
|---|---|
| Python bytecode/Dynamo and FX graph acquisition | Python adapter |
| Binary FX graph serialization and subprocess boundary | Python adapter |
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
complete LFM2.5 Q8 schedule is not yet interpreted inside the OCaml process.

# Current scope

The cross-language planner supports N-dimensional placeholders, `linear`,
Q8 linear, `mm`/`matmul`, pointwise arithmetic/comparison and common unary
operators, mean, casts, views, reshape, transpose, unsqueeze, expand,
contiguous, normalized static indexing, concat, `relu`, `gelu`, static integer
ranges, prepended differences, boolean cumulative sums, scalar fills, and
LFM's rank-two/two-index gather, and preserves other FX nodes as opaque plan
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

The FX compiler consumes the versioned binary `graph.llmopt` and emits a
versioned binary `package.llmopt` containing the typed
command stream, N-dimensional value shapes, operator arguments, kernel ABI,
cache policy, metallib path, and optional tensor-store path. The copied binary
graph, pretty-printed `plan.txt`, MSL, and LLVM IR are compiler artifacts and
are not referenced by the serving runtime; JSON diagnostics are opt-in. Kernel entry points carry an operation,
input/output dtype, and threadgroup shape. A compiled-graph package has no
tensor store; `--weights weights.llmopt` emits a serving package that references
exactly one binary tensor archive. Package ABI v6 names weight-archive ABI v1
and reads ABI-v2, ABI-v3, ABI-v4, and ABI-v5 packages.
Tensor dtype, rank, shape, offset, and byte length remain authoritative in that
archive rather than being split into per-tensor files.

When a captured graph contains static tensors, the Python adapter streams them
into `weights.llmopt` one at a time. The index uses fixed-width little-endian
fields and each payload starts at a 256-byte boundary; neither the index nor the
payload is JSON. This bounds CPU staging to the current tensor instead of
retaining a second whole-model CPU copy. The preceding safetensors-based real
350M Q8 capture emitted 241 tensor keys totaling 422,104,704 payload bytes. Its
saved tensors were converted offline into a 422,137,216-byte `weights.llmopt`.
The later use-cache capture wrote the same-sized binary archive directly and
shared it across both specializations. The
OCaml package validator resolves every binary-schedule tensor binding against
archive dtype and N-dimensional shape before runtime loading.

The Ninja-built OCaml runtime consumes that binary package directly, validates
the metallib, tensor archive, schedule bindings, and declared entry points,
parses the binary weight index, maps the archive once, and creates retained
Metal views at tensor byte offsets. The runtime now walks the binary schedule
for runtime and tensor-store inputs, allocation, exact buffer copies,
metadata-only aliases, identity casts, Q8 linear, and named outputs. Pipeline
states are cached per loaded library instead of rebuilt per dispatch. With 61%
system memory free, one 2x3x4 float16/Q8 fixture loaded `weights.llmopt`, bound
all schedule values, dispatched `llmopt_q8_linear`, and returned
`[3.5, 8, 1, 1.5, 4, 2]` exactly on Apple M4 Pro. This proves automatic
execution of that schedule subset against the replacement archive; it does not
prove complete model-schedule execution.

The runtime's generic native command path now selects declared kernels by
typed operation and input/output dtype, packs fixed-width Metal parameters,
binds schedule buffers, and reuses the per-library pipeline cache. It executes
the currently emitted matmul, RMSNorm, ShortConv, masked-attention, embedding,
arange, fill, diff, cumsum, and Gather2 families in addition to Q8 linear. A
fixed JSON-free fixture contains 42 commands and 13 kernel declarations. With
57% system memory free and no model process, one Apple M4 Pro execution
dispatched 12 kernel commands and matched all 12 outputs byte-for-byte. This
was a 59,325-byte package-plus-metallib probe, not a 350M model run. Pointwise,
non-identity cast, tensor movement, reduction, recurrent-state kernels, the
float16 vocabulary projection, and batched command submission remain outside
that device evidence.

Package ABI v3 adds a typed cast kernel family. Three generated kernels cover
the model's non-identity conversions: float16 to float32, float32 to float16,
and int64 to float32. A second fixed run started at 60% free memory and executed
the expanded 51-command package once; 15 kernel dispatches produced 15 exact
outputs, including all three cast directions. The existing ABI-v2 model
packages remain readable and require only offline replanning to declare these
new kernels.

Package ABI v4 adds a typed `Pointwise` operation with exact entry-name
selection. Nine generated kernels cover the pointwise inventory before the
first transformer block in both preserved plans: float16 add/multiply/negate/
SiLU, int64 add and less-equal, and float32 multiply/cosine/sine. Binary
operands support rank-eight row-major broadcasting and either operand may be a
packed scalar. One fixed 81-command package started at 58% free memory with no
model process, dispatched 24 kernels on Apple M4 Pro, and matched all 24 outputs
byte-for-byte. Its package and metallib totaled 112,789 bytes. The preserved
model graphs were subsequently replanned through the binary compiler transport
and now declare these kernels.

Package ABI v5 adds a typed `Movement` operation and eleven exact entry points:
float16/float32 transpose, float16/float32/int64 static index,
float16/float32/bool expand, float16/float32 two-input concat, and float16 roll.
The kernels materialize dense row-major outputs through rank eight. View,
reshape, unsqueeze, and contiguous are zero-copy aliases because every
layout-changing predecessor is materialized. One 118-command package started
with 7.67 GB (29.8%) reclaimable and no model process, dispatched 35 kernels on
Apple M4 Pro, and matched all 36 outputs byte-for-byte. Its 6,223-byte package
and 160,230-byte metallib totaled 166,453 bytes; this was not a model run.

Package ABI v6 adds typed `Reduction` and `Update_slice` operations. The
captured decode plan uses ten float16 sums over axis two and ten functional
updates of the trailing recurrent-cache slot. The sum kernel accumulates in
float32 before writing float16; the destination-driven update kernel preserves
unselected elements and maps normalized `At`, `Slice`, and `New_axis`
selectors back to the source. One 125-command package started with 7.54 GB
(29.3%) reclaimable and no model process, dispatched 37 kernels on Apple M4
Pro, and matched all 38 outputs byte-for-byte. Its 6,630-byte package and
169,895-byte metallib totaled 176,525 bytes; this was not a model run.

The same ABI-v6 package can now declare `llmopt_linear_f16` for the untied
runtime operation used by the final vocabulary projection. One SIMD group owns
one output channel, reads its contiguous 1,024-element weight row cooperatively,
reuses each weight across blocks of up to eight prompt rows, reduces float32
partial sums with `simd_sum`, and stores float16. A 129-command fixed package
started at 59% free system memory with no model process, dispatched 38 kernels
on Apple M4 Pro, and matched all 39 outputs byte-for-byte. Its 6,861-byte package
and 174,942-byte metallib total 181,803 bytes; this was not a model run.

Native execution now derives a pure alias-aware liveness plan from the typed
schedule before opening its workspace. Runtime and tensor-archive inputs remain
external; metadata-only movement and identity casts share canonical ownership;
materialized values receive deterministic 256-byte-aligned offsets; and named
outputs remain live until execution completes. The runtime allocates one Metal
workspace and binds retained views instead of retaining one independent buffer
per intermediate. Offline package checks report 1,153,792 bytes for the
872-command prefill high-water mark versus 9,855,488 aligned bytes without
reuse, and 271,360 versus 2,151,680 bytes for decode. One 58%-free-memory fixed
device run used a 9,728-byte workspace and preserved all 39 exact outputs. No
complete model schedule ran in that probe.

The memory-bounded manifest-v2 recapture now reaches package generation. Its
1,115 FX nodes initially became 835 schedule commands: 793 typed and 42 opaque,
compared with 379 typed and 736 opaque in the saved v1 package. A subsequent
offline replan corrected a target-suffix collision and moved all 14 expand
commands into the typed set. Typed depthwise ShortConv lowering then moved the
ten model `conv1d` commands, and masked-attention lowering moved all six SDPA
commands. Embedding lowering moved the token lookup. Schedule v7 then typed the
five static ranges, prepended difference, boolean cumulative sum, scalar fill,
and two advanced gathers, while explicitly eliding the one unused framework
telemetry call. The resulting no-cache prefill package contains 834 commands
and zero opaque operations. Direct-FX
execution is bit exact against eager MPS for the six-token probe. This is
capture and planning evidence; PyTorch MPS, not the OCaml package runtime,
executed the parity check.

A later bounded use-cache attempt captured both specializations before a
post-capture check failed because mixed matmul/Q8 graphs omitted the additive
Q8 Metal family. The preserved manifests contain 1,155 prefill nodes and 1,195
decode nodes. Prefill has one runtime input and 23 outputs; decode has 23
runtime inputs, including ten ShortConv states and twelve attention K/V
tensors, and 13 outputs. Both graphs bind the same 241 static tensors. The
capture session seals one 422,137,216-byte `weights.llmopt` and hard-links it
into graph directories after rebinding static aliases by storage identity.

The Metal emitter now includes Q8 kernels additively when another operation is
the graph's primary kernel family. Schedule v8 types prefill cache crop,
float16 zero fill, copy, and empty-concat identity, plus decode roll, functional
slice update, copy, and sum. Offline replanning of the preserved manifests
emits 872 prefill commands and 926 decode commands with zero opaque operations.
Their 14-kernel and 10-kernel Metal programs compile to metallibs, and both
packages validate all 241 archive bindings. The failed capture process did not
write its parity result, so the preserved graphs provide structure and package
evidence but no retained eager/compiled parity measurement.

The same preserved manifests now round-trip exactly through binary FX
transport ABI v1. Prefill occupies 253,354 bytes instead of 776,844 bytes of
diagnostic JSON; decode occupies 259,928 instead of 796,970. Offline binary
replanning now writes JSON-free ABI-v6 artifact directories with 872/926
commands, 38/36 kernels, zero opaque operations, and all 241 archive bindings
validated. Both packages add the shared float16-linear entry for their
`6x65536x1024` and `1x65536x1024` projections; decode also adds one sum and one
slice-update entry shared across its ten recurrent blocks. This conversion and
replan loaded no model and launched no Metal device work.

The source graph measures 85 getitem, 10 chunk, and 13 concat nodes. For v2,
the planner now holds chunk partitions as compile-time descriptors and
emits normalized slices directly at integer getitem consumers, avoiding a
tuple-valued runtime command. Static tensor indices, concat, and the position
and mask primitives survive the schedule-v7 binary round trip and CPU
interpretation. The latest offline package structurally validates with 241
tensors, declares 11 generated kernels, and its 12,443-byte MSL compiles to a
49,342-byte metallib. These additions cover the previously opaque prefill
nodes; native command interpretation and generated materialization of every
other typed schedule command remain open.

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
