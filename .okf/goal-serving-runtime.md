---
type: Research Goal
title: 'Complete OCaml Metal serving stack for LFM2.5'
description: 'The requirement-by-requirement completion map from torch.compile capture through OCaml cached serving and ERS measurement.'
tags: [goal, compiler, ocaml, metal, serving, radix-cache, kv-cache, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T22:57:37Z' }
sources:
  - id: frontend
    resource: /python/llmopt_backend/__init__.py
    title: Current torch.compile backend
  - id: compiler
    resource: /bin/fx_compile.ml
    title: Current OCaml FX compiler entry point
  - id: metal-emitter
    resource: /lib/metal.ml
    title: Current Metal source emitter
  - id: package-abi
    resource: /lib/serving_package.ml
    title: Versioned compiled-graph and serving package ABI
  - id: python-runtime
    resource: /python/llmopt_backend/metal_runtime.py
    title: Current Python generated-library loader
  - id: native-bridge
    resource: /native/metal_runtime.cpp
    title: Current PyTorch MPS C++ dispatch bridge
  - id: ocaml-metal-runtime
    resource: /lib/metal_runtime.ml
    title: Native OCaml package loader and Metal dispatch API
  - id: serving-cache
    resource: /lib/serving_cache.ml
    title: OCaml radix and KV ownership layer
  - id: benchmark
    resource: /bench/lfm25_benchsuite.py
    title: LFM2.5 ERS and needle benchsuite
  - id: build
    resource: /ninja.build
    title: Ninja build graph
---

# Objective

Build an optimizing LLM compiler whose frontend is a PyTorch
`torch.compile`/Dynamo FX backend and whose serving data plane is entirely
OCaml plus Metal. Compilation emits a self-contained serving package. The
OCaml process loads that package, tokenizes and schedules requests, dispatches
the generated Metal kernels, and owns mandatory radix-prefix reuse with a
configurable FP16 or grouped-Q8 KV cache. LFM2.5-350M Q8 is the primary
development target; the design must retain a path to the 2.6B variant.

Ninja remains the only build orchestrator. Dune is not part of this goal.

# Completion map

| Requirement | Evidence that proves it | Current evidence | State |
|---|---|---|---|
| Dynamo/FX graph capture | PyTorch invokes `backend=llmopt` and the captured graph reaches OCaml | One memory-bounded use-cache attempt preserved a 1,155-node prefill graph and a 1,195-node decode graph, with one and 23 runtime inputs respectively, while sharing all 241 static tensors through one binary archive | implemented for fixed six-token prefill and one-token decode specializations |
| Binary compiler transport | Dynamo graph metadata reaches OCaml through a versioned binary format; JSON is optional diagnostics only | Default capture now writes `LLMOPTFX` ABI-v1 `graph.llmopt`; Python and OCaml round trips cover every typed argument form, malformed input is rejected, and preserved prefill/decode graphs round-trip exactly. `LLMOPT_FX_DIAGNOSTICS=1` is required for JSON output | implemented |
| Complete LFM2.5 compiler coverage | One captured model package has no opaque or PyTorch-fallback operations needed by prefill/decode | Replanning the preserved manifests produces 872 prefill and 926 decode commands with zero opaque operations; every observed kernel family, including the final float16 projection, is emitted into both metallibs | implemented for the fixed six-token prefill and one-token decode specializations |
| Generated serving-package ABI | Versioned package contains graph schedule, kernel entry points, one memory-mappable tensor archive, and cache layout; OCaml validates it | Package ABI v6 retains ABI-v2/v3/v4/v5 reads. Binary-input replanning validates 872-command/38-kernel prefill and 926-command/36-kernel decode packages against the shared 422,137,216-byte weight-ABI-v1 archive | partial |
| Metal compilation artifacts | Package build emits loadable metallib kernels for every scheduled model operation | Binary-input ABI-v6 replanning compiles 38 prefill and 36 decode entries covering every observed family. The final `6x65536x1024` and `1x65536x1024` float16 projections share one SIMD-reduction entry | implemented for the fixed captured specializations |
| Native OCaml Metal runtime | Ninja-built OCaml executable selects a device, loads metallib functions, maps tensor storage, binds tensor views, and submits commands without Python or PyTorch in the serving hot path | The archive-backed Q8 schedule remains exact. One ABI-v6 fixture executes 129 commands from one 9,728-byte liveness-planned workspace and matches 39 outputs byte-for-byte across every observed family; 38 commands dispatch Metal. Full-model execution and batched submission remain absent | partial |
| Model data ownership | OCaml loads package weights and persistent activations in the declared Q8/FP16 layouts | All 241 captured 350M tensors are in one 422,137,216-byte binary archive. The OCaml checker validates every binding and plans transient prefill/decode high-water marks of 1,153,792/271,360 bytes; physical persistent KV/checkpoint ownership remains absent | partial |
| Tokenization, sampling, and serving protocol | OCaml accepts the benchmark request contract, applies the LFM chat template/tokenizer, streams generated tokens, and reports cache usage | Current request loop and tokenizer are Python/Transformers | open |
| Mandatory radix-prefix reuse | Multi-turn requests produce non-zero cached-prefix accounting and reuse the matched KV/recurrent checkpoint while preserving output parity | Compressed radix structure and ownership tests pass, but no request reaches it | partial |
| Configurable KV quantization | FP16 and Q8 runs bind physical Metal KV/checkpoint buffers and execute matching quantize/dequantize paths | Typed format and byte accounting exist; physical buffers and kernels do not | partial |
| Benchmark correctness and measurement | Exact logits/token parity, retrieval and response-format results, request counts, raw TTFT/TPOT, ERS, and cache-hit accounting are written by one reproducible command | Existing suite covers parity and latency; needle grading is corrected in the current slice; cached tokens remain zero | partial |

# Completion condition

The goal is complete only when one Ninja-built flow starts from a captured
LFM2.5-350M graph, crosses a versioned binary compiler boundary, produces a
versioned serving package, launches the OCaml
runtime, serves the benchmark through generated Metal kernels, records actual
radix/KV reuse across turns, and emits parity, needle, latency, and ERS
evidence. Q8 must be the default cache/weight policy with FP16 selectable.
There is no predeclared ERS score threshold; the measured value is evidence,
not a substitute completion gate.

# Ordered work

1. Keep benchmark semantics trustworthy, including separate retrieval and
   exact-response-format observations.
2. Define and emit the serving-package ABI.
3. Implement OCaml Metal device/library/buffer/command primitives under Ninja.
4. Export all LFM2.5 parameters into the single binary `weights.llmopt` archive, then lower
   and schedule every prefill/decode operation against its tensor keys.
5. Implement OCaml tokenization, sampling, request handling, and persistent
   generation state.
6. Bind physical FP16/Q8 KV buffers and recurrent checkpoints to mandatory
   radix-cache ownership.
7. Measure exact behavior, cache reuse, raw latency, and ERS; then optimize the
   measured Metal boundaries.
