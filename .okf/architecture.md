---
type: Architecture
title: 'Dynamo/FX compiler with an OCaml Metal serving runtime'
description: 'PyTorch Dynamo supplies FX graphs, OCaml plans and emits Metal, and the intended OCaml serving runtime owns prefix/KV state and dispatch.'
tags: [architecture, pytorch, fx, ocaml, effects, metal, serving, radix-cache]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T09:57:58Z' }
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
  - id: batched-command-result
    resource: /bench/results/lfm25-350m-q8-native-batched-command-2026-08-24.txt
    title: Schedule-wide Metal command-buffer observation
  - id: q8-gemv-result
    resource: /bench/results/lfm25-350m-q8-native-gemv-2026-08-24.txt
    title: Decode-specialized Q8 GEMV observation
  - id: cache-batching-result
    resource: /bench/results/lfm25-350m-q8-native-cache-batching-2026-08-25.txt
    title: Physical cache-submission batching observation
  - id: local-serving-engine
    resource: /lib/serving_engine.ml
    title: Native prefill, decode, and radix coordinator
  - id: local-serving-protocol
    resource: /lib/openai_protocol.ml
    title: Typed external OpenAI compatibility edge
  - id: local-serving-server
    resource: /bin/lfm_serve.ml
    title: Persistent native OCaml HTTP server
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
  - id: current-full-q8-result
    resource: /bench/results/lfm25-350m-q8-lm-head-measurement-2026-08-25.txt
    title: Current full-Q8 native serving observation
  - id: paired-simd-result
    resource: /bench/results/lfm25-350m-q8-paired-simd-compiler-2026-08-25.txt
    title: Paired SIMD Q8 compiler evidence
  - id: paired-simd-measurement
    resource: /bench/results/lfm25-350m-q8-paired-simd-measurement-2026-08-25.txt
    title: Paired SIMD Q8 model measurement
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

The direct FX callable remains a PyTorch MPS comparison path. Separately, the
native OCaml server loads and executes the complete captured LFM2.5-350M
prefill/decode package, owns radix and physical Q8 KV/recurrent state, and
serves the HTTP/SSE benchmark without Python or PyTorch in its hot path.

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
configuration boundary. Native OCaml owns physical token and recurrent
checkpoint `MTLBuffer` pools. Package ABI v8 declares eight cache entries and
adds source slicing for decode append, while retaining ABI-v2 through ABI-v7
reads. `Serving_engine` now coordinates model execution, slot/checkpoint
reservation, physical packing, radix insertion, leased prefix reuse, state
unpacking, and rollback. The current serial HTTP trace completes 4/4 warmup
and 4/4 scored requests, reuses 80/194 prompt tokens, and matches all full-Q8
eager output sequences.

The FX compiler consumes the versioned binary `graph.llmopt` and emits a
versioned binary `package.llmopt` containing the typed
command stream, N-dimensional value shapes, operator arguments, kernel ABI,
cache policy, metallib path, and optional tensor-store path. The copied binary
graph, pretty-printed `plan.txt`, MSL, and LLVM IR are compiler artifacts and
are not referenced by the serving runtime; JSON diagnostics are opt-in. Kernel entry points carry an operation,
input/output dtype, and threadgroup shape. A compiled-graph package has no
tensor store; `--weights weights.llmopt` emits a serving package that references
exactly one binary tensor archive. Package ABI v11 names weight-archive ABI v1,
adds typed Q8-linear/SiLU, Q8-linear/residual, and multiplied-input Q8
down-projection operations on top of sliced cache operations, and reads ABI-v2
through ABI-v10 packages.
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

The pair loader validates both packages before opening Metal, shares one device
context, and keys mapped archives by filesystem device and inode. The fixed
prefill/decode paths therefore share the hard-linked 422,137,216-byte archive
mapping. A full Q8 run dispatched 522 prefill and 544 decode commands, sampled
tokens `19130,11040`, reused a six-token radix prefix, and retained seven token
slots plus two recurrent checkpoints. The 0.432272/0.119545-second stage times
are single observations. A separate eager-Q8 probe returned the same two token
IDs; exact logits, variable-length generation, request serving, needle
retrieval, and ERS remain unmeasured.

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
replanning now writes JSON-free ABI-v8 artifact directories with 872/926
commands, 46/44 kernels, zero opaque operations, and all 241 archive bindings
validated. Both packages add the shared float16-linear entry for their
`6x65536x1024` and `1x65536x1024` projections; decode also adds one sum and one
slice-update entry shared across its ten recurrent blocks, and each package
declares eight FP16/Q8 cache conversion entries. This conversion and replan
loaded no model and launched no Metal device work.

The ABI-v8 schedules now serve as typed sequence templates rather than fixed
six-token executables. `Serving_schedule.Lfm25` substitutes prefill, decode
past, and decode total lengths, rewrites the dependent scalar/index parameters,
and re-infers every SSA output shape from transformed inputs while retaining
static archive tensors unchanged. The runtime creates a liveness workspace for
the specialized schedule and passes dynamic `m/n/k` parameters to float32
matmul and linear kernels. Offline real-package checks cover prefill lengths
13/128/4,096 and decode-past lengths 1/127/4,095.

One native Q8 execution ran the captured prefill template and three consecutive
decode specializations. Radix matches grew through prefixes 6/7/8, physical
state grew to nine token slots and four recurrent checkpoints, and the four
greedy tokens `19130,11040,11207,1414` matched a separately run eager-Q8
reference. Tokenizer/chat integration, an HTTP request owner, native needle
requests, and ERS remain outside that observation.

`Generation_core` now owns the device-independent autoregressive state machine
behind a typed engine functor. The native adapter composes `LLMOPTTK`,
`Lfm_chat`, radix-aware `Serving_engine.prompt`, greedy float16 sampling,
repeated decode, stop/length outcomes, text decoding, and latency/cache
observations. For a cached conversation, prompt preparation leases the deepest
checkpoint before the final prompt token and replays only the uncached suffix;
a cold prompt runs the specialized prefill schedule.

The first real chat run encoded `user: Say hi.` into 13 IDs, generated
`36309,510,2213,1011`, decoded `Hello! How can`, and matched a separate
eager-Q8 reference exactly. This establishes the direct native generation
path.

`llmopt-serve` now owns that generation value across serial HTTP requests. Its
OpenAI-compatible JSON/SSE adapter is the only JSON serving boundary; binary
tokenizer, package, tensor, schedule, radix/KV, and Metal representations stay
inside the native process. Incremental UTF-8 decoding emits every generated
token ID, including empty-text special tokens, so the benchmark timestamps
tokens rather than visible text fragments. One warmed four-request scored
smoke reused 80/194 prompt tokens, matched all eager-Q8 token sequences, and
measured native ERS `0.06169548638841863`. The optimized fixed-12-token native
long-context matrix retrieves 6/6 at 2,048/4,096 tokens and exactly matches all
12 eager-Q8 IDs in every request; exact-only text is 0/6 because generation
continues through EOS.

Schedule execution now accumulates its generic and Q8 compute dispatches into
one compute encoder and inserts ordered blit encoders for typed copies. It
commits and waits once per schedule instead of once per generated kernel. The
fixed 39-output device probe remains exact, and the matched warmed HTTP smoke
raises native ERS from `0.06169548638841863` to `0.11058587181748172` while
preserving all token IDs and cached-prefix counts.

Decode schedules now select a vectorized one-row Q8 GEMV entry when `m = 1`.
Multi-row prefill retains the 16 by 16 output tile but stages 64 reduction
elements as sixteen activation4/dequantized-weight4 vectors. A 40-output
device probe selects `llmopt_q8_gemv` and remains exact. On the one matched warmed HTTP
trace, all four request TPOT values fall by 9.12 to 10.71 ms, median TTFT falls
by 87.971 ms, and ERS changes from `0.11058587181748172` to
`0.10860341576307225`; token IDs and 80/194 cache reuse remain unchanged.

Current ABI-v11 packages replace that scalar decode entry with a SIMD-group
family. One 32-lane group owns an output channel, lanes traverse the reduction
dimension at stride 32, and `simd_sum` combines their partial accumulators.
Eight groups share each 256-thread threadgroup. The runtime prefers the new
name and falls back to the scalar name and grid when loading an older package.
All four Q8 operation families compile in float16 and float32. The combined
optimized model run exercises them with exact token parity, without isolating
their individual latency contribution.

The subsequent decode package vectorizes work within each SIMD lane. Every
lane loads four activations and a `char4` weight, applies the per-channel scale
to the four weight elements, accumulates one float4 dot, and advances by 128
reduction elements. Scalar stride-32 cleanup remains for alignment and tails.
The model's 1024/4608 decode reductions use the packed path throughout. One
bounded model run preserves all four eager-Q8 sequences and 80/194 reuse while
observing ERS
`0.3253700872862615`, median TTFT `95.60127052827738 ms`, and median TPOT
`7.93296533326308 ms`.

The next decode mapping shares each packed activation load across two adjacent
output channels. One SIMD group owns two independent accumulators and retains
the existing per-channel dot and `simd_sum` order; eight groups cover 16
channels per threadgroup. The 65,536-channel vocabulary projection therefore
uses 4,096 instead of 8,192 threadgroups. Paired entries exist for all four Q8
epilogue families and both activation dtypes, with typed fallback to the prior
single-channel or scalar layouts. Both full-Q8 350M stages compile and one
odd-tail synthetic Metal fixture returns 46/46 exact outputs. The bounded
model trace preserves all short tokens and 80/194 radix reuse while observing
ERS `0.40701575836615456` and median TTFT/TPOT `75.225/6.937 ms`; relative to
the prior single-channel report, those medians change by `+3.748/-0.374 ms`.

The vector-staged prefill package reduces synchronization without changing the
16 by 16 output ownership. For k=1024/4608, 64/288 scalar reduction tiles
become 16/72 vector tiles and emitted barrier counts change from 128/576 to
32/144. The partial-k 2x4 Metal fixture remains bit exact; both model
metallibs compile and both selectable KV formats validate. One bounded 350M
run preserves all four eager-Q8 sequences and 80/194 reuse while observing ERS
`0.3377415731686302`, median TTFT `93.155520997243 ms`, and median TPOT
`7.948180340463296 ms`. Against the preceding native observation, those
aggregate changes are `+0.012371485882368694`, `-2.44574953103438 ms`, and
`+0.015215007200215958 ms`; per-request deltas are mixed and the reports are
not interleaved.

RMSNorm uses the same launch geometry at the row level. One SIMD group owns a
row, lanes traverse its final dimension at stride 32, `simd_sum` combines the
squared-value partials, and the lanes cooperatively write normalized values.
Eight rows share a 256-thread group. Both input-dtype variants compile, while
the runtime retains scalar-name and scalar-grid fallback for older packages.
The 350M prefill and decode templates each contain 45 such commands. The
combined optimized model run exercises them with exact token parity, without
isolating their individual latency contribution.

The width-64 attention specialization also owns one query row per SIMD group.
It computes each query-key dot product once, updates the softmax maximum and
denominator online, and rescales two output accumulators per lane. The prior
scalar form recomputed a full score for the maximum, denominator, and every
output dimension: 66 times per key at width 64. The scalar entry remains
declared for wider heads and old packages. All six attention commands per 350M
stage select and execute the new entry in the exact-token aggregate run.

Physical KV and recurrent cache conversion now batches each unpack or pack
phase into one ordered command buffer. For the six-attention, ten-recurrent
350M model, decode changes from 45 synchronous submissions to three including
the generated schedule. The exact Q8/FP16 cache probe retains all bytes. The
matched HTTP trace retains token parity and 80/194 reuse, lowers all four TPOT
values by 2.82 to 5.45 ms, and measures ERS `0.11381808711306604`.

Dependent cached-suffix replay now goes further: it unpacks the matched prefix
once, then encodes each dependent decode schedule and its per-token cache
writes into one ordered batch while retaining a radix checkpoint at every
suffix position. The corrected path completes 4/4 scored requests with exact
eager-Q8 tokens and unchanged 80/194 reuse. Together with the current fusion
and SIMD kernels, one aggregate run measures ERS `0.23655514122115978`, median
TTFT `136.7437920125667 ms`, and median TPOT `14.803701342316344 ms`; this
observation cannot attribute the delta to one component.

The following packed Q8 decode package keeps that schedule and cache behavior
but processes four activation/weight elements per SIMD-lane iteration. One
bounded 350M observation preserves all four eager-Q8 token sequences and
80/194 radix reuse while measuring ERS `0.3253700872862615`, median TTFT
`95.60127052827738 ms`, and median TPOT `7.93296533326308 ms`. The previous
and packed reports are non-interleaved single observations, so their delta is
not assigned wholly to the packed loop.

The first epilogue optimizer replaces a Q8 projection and its sole SiLU
consumer with `Q8_linear_silu`; any second consumer prevents the rewrite.
Package and schedule ABI v9 carry distinct fused GEMM and GEMV kernel families.
The float16 Metal epilogue explicitly reproduces the intermediate half
rounding before activation. Offline replanning finds 16 pairs in both stages,
reducing prefill from 872 to 856 commands and decode from 926 to 910 while
retaining zero opaque operations and all 241 tensor bindings. Both metallibs
compile. The combined optimized run executes this pass with exact model tokens,
but does not isolate its latency contribution.

The second epilogue optimizer recognizes same-shape residual addition after a
sole-consumer Q8 projection. It rejects broadcast residuals and preserves the
materialized half-rounding point before addition. Schedule/package ABI v10 and
the runtime bind the residual as one extra typed Metal buffer. All 32 expected
pairs fuse in each stage, reducing prefill from 856 to 824 commands and decode
from 910 to 878. The generated metallibs and Q8/FP16-selectable package pair
validate against the same archive inode. The combined optimized run executes
this pass with exact model tokens; its contribution is not isolated.

The third Q8 pass recognizes a sole-consumer float16 multiply feeding an
already-fused Q8 down projection plus residual. It preserves both upstream
projection outputs and multiplies them while loading the down-projection
reduction tile, reproducing the materialized float16 product before float32
accumulation. All 16 expected boundaries fuse in each stage, reducing prefill
from 824 to 808 commands and decode from 878 to 862. Captured-template
workspace falls to 1,098,496/262,144 bytes. ABI-v11 packages and both
metallibs validate against the shared archive. The combined optimized run
executes this pass with exact model tokens; its contribution is not isolated.

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
