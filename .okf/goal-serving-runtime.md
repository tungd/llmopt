---
type: Research Goal
title: 'Complete OCaml Metal serving stack for LFM2.5'
description: 'The requirement-by-requirement completion map from torch.compile capture through OCaml cached serving and ERS measurement.'
tags: [goal, compiler, ocaml, metal, serving, radix-cache, kv-cache, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T18:53:41Z' }
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
| Dynamo/FX graph capture | PyTorch invokes `backend=llmopt` and the captured graph reaches OCaml | One memory-bounded manifest-v2 capture reached OCaml, preserved all 241 static tensors and typed arguments, and planned the complete six-token no-cache graph | implemented for the no-cache probe; decode capture open |
| Complete LFM2.5 compiler coverage | One captured model package has no opaque or PyTorch-fallback operations needed by prefill/decode | Replanning the real manifest-v2 no-cache capture produces 835 commands with 11 opaque after embedding also became typed; only position/mask construction plus one logging side effect remain opaque in prefill, while decode and KV-state operations remain | partial |
| Generated serving-package ABI | Versioned package contains graph schedule, kernel entry points, one memory-mappable tensor archive, and cache layout; OCaml validates it | Schedule v6 package validation passes with 835 commands, 11 opaque commands, 241 archive bindings, and 6 declared kernels while reading v1-v5; full scheduled-kernel coverage and native invocation metadata remain incomplete | partial |
| Metal compilation artifacts | Package build emits loadable metallib kernels for every scheduled model operation | Q8 linear, fused RMSNorm, scalar depthwise ShortConv, correctness-first masked attention, and embedding kernels compile with Xcode Metal; pointwise/movement materialization and complete model lowering are absent | partial |
| Native OCaml Metal runtime | Ninja-built OCaml executable selects a device, loads metallib functions, maps tensor storage, binds tensor views, and submits commands without Python or PyTorch in the serving hot path | With 56% memory free, the loader returned the exact Q8 fixture result from a directory containing only `package.llmopt`, `kernel.metallib`, and `weights.safetensors`; complete command-stream interpretation is not present | partial |
| Model data ownership | OCaml loads package weights and persistent activations in the declared Q8/FP16 layouts | Dynamo exported all 241 captured 350M tensors into one archive and the OCaml checker validated every binding; native full-model execution and persistent activation ownership remain absent | partial |
| Tokenization, sampling, and serving protocol | OCaml accepts the benchmark request contract, applies the LFM chat template/tokenizer, streams generated tokens, and reports cache usage | Current request loop and tokenizer are Python/Transformers | open |
| Mandatory radix-prefix reuse | Multi-turn requests produce non-zero cached-prefix accounting and reuse the matched KV/recurrent checkpoint while preserving output parity | Compressed radix structure and ownership tests pass, but no request reaches it | partial |
| Configurable KV quantization | FP16 and Q8 runs bind physical Metal KV/checkpoint buffers and execute matching quantize/dequantize paths | Typed format and byte accounting exist; physical buffers and kernels do not | partial |
| Benchmark correctness and measurement | Exact logits/token parity, retrieval and response-format results, request counts, raw TTFT/TPOT, ERS, and cache-hit accounting are written by one reproducible command | Existing suite covers parity and latency; needle grading is corrected in the current slice; cached tokens remain zero | partial |

# Completion condition

The goal is complete only when one Ninja-built flow starts from a captured
LFM2.5-350M graph, produces a versioned serving package, launches the OCaml
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
4. Export all LFM2.5 parameters into the single safetensors archive, then lower
   and schedule every prefill/decode operation against its tensor keys.
5. Implement OCaml tokenization, sampling, request handling, and persistent
   generation state.
6. Bind physical FP16/Q8 KV buffers and recurrent checkpoints to mandatory
   radix-cache ownership.
7. Measure exact behavior, cache reuse, raw latency, and ERS; then optimize the
   measured Metal boundaries.
