---
type: Research Goal
title: 'Complete OCaml Metal serving stack for LFM2.5'
description: 'The requirement-by-requirement completion map from torch.compile capture through OCaml cached serving and ERS measurement.'
tags: [goal, compiler, ocaml, metal, serving, radix-cache, kv-cache, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T07:07:41Z' }
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
  - id: tokenizer
    resource: /lib/tokenizer.ml
    title: Native binary LFM tokenizer
  - id: chat
    resource: /lib/lfm_chat.ml
    title: Typed text-only LFM chat encoding
  - id: native-server
    resource: /bin/lfm_serve.ml
    title: Persistent native OCaml HTTP and SSE server
  - id: benchmark
    resource: /bench/lfm25_benchsuite.py
    title: LFM2.5 ERS and needle benchsuite
  - id: native-http-result
    resource: /bench/results/lfm25-350m-q8-native-http-2026-08-24.txt
    title: Native HTTP token parity and ERS observation
  - id: native-needle-result
    resource: /bench/results/lfm25-350m-q8-native-needle-stop-eos-2026-08-24.txt
    title: Native long-context retrieval observation
  - id: batched-command-result
    resource: /bench/results/lfm25-350m-q8-native-batched-command-2026-08-24.txt
    title: Native command-buffer batching result
  - id: q8-gemv-result
    resource: /bench/results/lfm25-350m-q8-native-gemv-2026-08-24.txt
    title: Native decode-specialized Q8 result
  - id: cache-batching-result
    resource: /bench/results/lfm25-350m-q8-native-cache-batching-2026-08-25.txt
    title: Native cache-submission batching result
  - id: q8-silu-fusion-result
    resource: /bench/results/lfm25-350m-q8-linear-silu-fusion-2026-08-25.txt
    title: Q8 linear-SiLU compiler fusion result
  - id: q8-add-fusion-result
    resource: /bench/results/lfm25-350m-q8-linear-add-fusion-2026-08-25.txt
    title: Q8 linear-residual compiler fusion result
  - id: q8-mul-add-fusion-result
    resource: /bench/results/lfm25-350m-q8-multiplied-input-fusion-2026-08-25.txt
    title: Q8 multiplied-input compiler fusion result
  - id: q8-simd-gemv-result
    resource: /bench/results/lfm25-350m-q8-simdgroup-gemv-2026-08-25.txt
    title: SIMD-group Q8 GEMV compiler result
  - id: simd-rms-result
    resource: /bench/results/lfm25-350m-q8-simdgroup-rmsnorm-2026-08-25.txt
    title: SIMD-group RMSNorm compiler result
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
| Dynamo/FX graph capture | PyTorch invokes `backend=llmopt` and the captured graph reaches OCaml | One memory-bounded use-cache attempt preserved a 1,155-node prefill graph and a 1,195-node decode graph, with one and 23 runtime inputs respectively, while sharing all 241 static tensors through one binary archive | implemented as captured prefill/decode templates |
| Binary compiler transport | Dynamo graph metadata reaches OCaml through a versioned binary format; JSON is optional diagnostics only | Default capture now writes `LLMOPTFX` ABI-v1 `graph.llmopt`; Python and OCaml round trips cover every typed argument form, malformed input is rejected, and preserved prefill/decode graphs round-trip exactly. `LLMOPT_FX_DIAGNOSTICS=1` is required for JSON output | implemented |
| Complete LFM2.5 compiler coverage | One captured model package has no opaque or PyTorch-fallback operations needed by prefill/decode | Replanning fuses 16 Q8-linear/SiLU, 32 Q8-linear/residual, and 16 multiplied-input down-projection boundaries per stage, producing 808 prefill and 862 decode commands with zero opaque operations. Typed specialization re-infers real schedules at prefill 13/128/4,096 and decode-past 1/127/4,095; every observed kernel family remains emitted | implemented for captured templates and observed LFM shapes |
| Generated serving-package ABI | Versioned package contains graph schedule, kernel entry points, one memory-mappable tensor archive, and cache layout; OCaml validates it | Package ABI v11 retains ABI-v2 through ABI-v10 reads and adds the typed Q8 multiplied-input/residual family. Offline package checks validate 808-command/60-entry prefill and 862-command/58-entry decode schedules against the shared 422,137,216-byte tensor archive | implemented for the captured template pair |
| Metal compilation artifacts | Package build emits loadable metallib kernels for every scheduled model operation | Binary-input ABI-v11 replanning compiles 60 prefill and 58 decode entries, including tiled Q8 prefill, SIMD-group Q8 decode families, 45 SIMD-group RMSNorm commands per stage, and FP16/Q8 cache conversion. Both generated MSL programs compile; the SIMD fixtures remain unlaunched | implemented for captured templates |
| Native OCaml Metal runtime | Ninja-built OCaml executable selects a device, loads metallib functions, maps tensor storage, binds tensor views, and submits commands without Python or PyTorch in the serving hot path | One persistent `llmopt-serve` process loads both stages and the inode-keyed archive. The latest successful trace uses one unpack, one generated-schedule, and one pack command buffer per decode with the earlier scalar GEMV and RMSNorm. New ABI-v11 packages prefer eight-output Q8 and eight-row RMSNorm SIMD-group entries, with scalar-name/grid fallback for older packages, but these reductions and the corrected dependent-suffix path have not run on device. Exact model logits remain absent | partial |
| Model data ownership | OCaml loads package weights and persistent activations in the declared Q8/FP16 layouts | The repeated Q8 run mapped the shared 241-tensor archive, grew to nine attention slots and four recurrent checkpoints, and passed cache validation. FP16 remains package-validated and small-fixture-executed rather than model-executed | implemented for Q8 prefill plus repeated decode |
| Tokenization, sampling, and serving protocol | OCaml accepts the benchmark request contract, applies the LFM chat template/tokenizer, streams generated tokens, and reports cache usage | `llmopt-serve` accepts the OpenAI-compatible chat contract, incrementally decodes UTF-8, streams every generated token ID plus visible text, and reports usage. The warmed scored smoke completed 4/4 requests with pinned output counts | implemented for the HTTP smoke contract |
| Mandatory radix-prefix reuse | Multi-turn requests produce non-zero cached-prefix accounting and reuse the matched KV/recurrent checkpoint while preserving output parity | Scored second turns reused 42/61 and 38/59 prompt tokens; total reuse was 80/194 while all four output sequences matched eager Q8 exactly. The new dependent suffix plan reserves and inserts one checkpoint per suffix token, preserving page-size-one branch reuse, but only static tests cover the corrected path | implemented for serial multi-turn smoke requests; batched suffix unmeasured |
| Configurable KV quantization | FP16 and Q8 runs bind physical Metal KV/checkpoint buffers and execute matching quantize/dequantize paths | Small exact probes cover FP16 and Q8-group-64. The full fixed model run used Q8 by default; the ABI-v8 pair validates FP16 but has not executed it at model scale | partial |
| Benchmark correctness and measurement | Exact logits/token parity, retrieval and response-format results, request counts, raw TTFT/TPOT, ERS, and cache-hit accounting are written by one reproducible command | The latest valid cache-batched trace records 4/4 exact eager-Q8 sequences, native/eager ERS `0.11381808711306604/0.36872784102635947`, median TTFT/TPOT `1014.696/91.146 ms`, and 80/194 cached prompt tokens. The suffix-batching attempt completed 2/4 warmup requests and produced no scored report. A stop-on-EOS long matrix retrieves 6/6 and matches the first seven eager IDs; the corrected fixed-12-token matrix and native exact logits remain open | partial |

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
