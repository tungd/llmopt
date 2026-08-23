---
type: Research Goal
title: 'Complete OCaml Metal serving stack for LFM2.5'
description: 'The requirement-by-requirement completion map from torch.compile capture through OCaml cached serving and ERS measurement.'
tags: [goal, compiler, ocaml, metal, serving, radix-cache, kv-cache, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T16:28:23Z' }
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
| Dynamo/FX graph capture | PyTorch invokes `backend=llmopt` and the captured graph reaches OCaml | FX manifest exporter and planner tests pass | implemented |
| Complete LFM2.5 compiler coverage | One captured model package has no opaque or PyTorch-fallback operations needed by prefill/decode | Current planner preserves unsupported FX nodes as opaque | partial |
| Generated serving-package ABI | Versioned package contains graph schedule, kernel entry points, tensor metadata, quantized weights, and cache layout; OCaml validates it | Ninja emits and validates a version-1 `compiled-graph` package with FX, plan, MSL, metallib, LLVM, typed kernel entries, and Q8-default cache policy; exported model weights and a complete operation schedule remain absent | partial |
| Metal compilation artifacts | Package build emits loadable metallib kernels for every scheduled model operation | Q8 linear and small graph fixtures emit Metal; complete model lowering is absent | partial |
| Native OCaml Metal runtime | Ninja-built OCaml executable selects a device, loads metallib functions, allocates/binds buffers, and submits commands without Python or PyTorch in the serving hot path | Current loader is Python and dispatch is a PyTorch MPS C++ extension | open |
| Model data ownership | OCaml loads package weights and persistent activations in the declared Q8/FP16 layouts | Weight ownership remains in PyTorch modules | open |
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
4. Lower and package every LFM2.5 prefill/decode operation and weight.
5. Implement OCaml tokenization, sampling, request handling, and persistent
   generation state.
6. Bind physical FP16/Q8 KV buffers and recurrent checkpoints to mandatory
   radix-cache ownership.
7. Measure exact behavior, cache reuse, raw latency, and ERS; then optimize the
   measured Metal boundaries.
