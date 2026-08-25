---
type: Research Goal
title: 'Complete OCaml Metal serving stack for LFM2.5'
description: 'The requirement-by-requirement completion map from torch.compile capture through OCaml cached serving and ERS measurement.'
tags: [goal, compiler, ocaml, metal, serving, radix-cache, kv-cache, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T10:03:08Z' }
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
  - id: native-fixed-needle-result
    resource: /bench/results/lfm25-350m-q8-native-needle-fixed12-2026-08-25.txt
    title: Native fixed-output long-context parity observation
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
  - id: online-attention-result
    resource: /bench/results/lfm25-350m-q8-online-softmax-attention-2026-08-25.txt
    title: Single-pass SIMD attention compiler result
  - id: optimized-stack-result
    resource: /bench/results/lfm25-350m-q8-native-optimized-stack-2026-08-25.txt
    title: Previous native optimized-stack parity and ERS result
  - id: packed-gemv-result
    resource: /bench/results/lfm25-350m-q8-packed-simd-gemv-2026-08-25.txt
    title: Packed SIMD Q8 compiler result
  - id: packed-gemv-measurement
    resource: /bench/results/lfm25-350m-q8-packed-simd-gemv-measurement-2026-08-25.txt
    title: Previous packed-decode native parity and ERS result
  - id: vector-prefill-result
    resource: /bench/results/lfm25-350m-q8-vector-prefill-2026-08-25.txt
    title: Vector-staged Q8 prefill compiler result
  - id: vector-prefill-measurement
    resource: /bench/results/lfm25-350m-q8-vector-prefill-measurement-2026-08-25.txt
    title: Previous native parity and ERS result
  - id: last-token-projection-measurement
    resource: /bench/results/lfm25-350m-q8-last-token-projection-measurement-2026-08-25.txt
    title: Previous final-row FP16-head native parity and ERS result
  - id: q8-lm-head-compiler
    resource: /bench/results/lfm25-350m-q8-lm-head-compiler-2026-08-25.txt
    title: Q8-default LM-head static compiler evidence
  - id: q8-lm-head-capture
    resource: /bench/results/lfm25-350m-q8-lm-head-capture-2026-08-25.txt
    title: Full-Q8 350M capture evidence
  - id: q8-lm-head-measurement
    resource: /bench/results/lfm25-350m-q8-lm-head-measurement-2026-08-25.txt
    title: Current native, eager, and needle evidence
  - id: paired-simd-compiler
    resource: /bench/results/lfm25-350m-q8-paired-simd-compiler-2026-08-25.txt
    title: Paired SIMD Q8 compiler evidence
  - id: paired-simd-measurement
    resource: /bench/results/lfm25-350m-q8-paired-simd-measurement-2026-08-25.txt
    title: Current paired SIMD native and needle evidence
  - id: fp16-kv-measurement
    resource: /bench/results/lfm25-350m-q8-paired-simd-fp16-kv-measurement-2026-08-25.txt
    title: Model-scale selectable FP16 KV evidence
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
development and memory-bounded validation target; the design must retain a
deferred portability path to the 2.6B variant.

Ninja remains the only build orchestrator. Dune is not part of this goal.

# Completion map

| Requirement | Evidence that proves it | Current evidence | State |
|---|---|---|---|
| Dynamo/FX graph capture | PyTorch invokes `backend=llmopt` and the captured graph reaches OCaml | One memory-bounded use-cache attempt preserved a 1,155-node prefill graph and a 1,195-node decode graph, with one and 23 runtime inputs respectively, while sharing all 241 static tensors through one binary archive | implemented as captured prefill/decode templates |
| Binary compiler transport | Dynamo graph metadata reaches OCaml through a versioned binary format; JSON is optional diagnostics only | Default capture now writes `LLMOPTFX` ABI-v1 `graph.llmopt`; Python and OCaml round trips cover every typed argument form, malformed input is rejected, and preserved prefill/decode graphs round-trip exactly. `LLMOPT_FX_DIAGNOSTICS=1` is required for JSON output | implemented |
| Complete LFM2.5 compiler coverage | One captured model package has no opaque or PyTorch-fallback operations needed by prefill/decode | The full-Q8 capture produces 810 prefill and 864 decode commands with zero opaque operations and all 93 linears quantized. Typed specialization re-infers prefill 13/128/4,096 and decode-past 1/127/4,095, projecting only the final `[1,1,65536]` Q8 vocabulary row | implemented for captured full-Q8 templates and observed LFM shapes |
| Generated serving-package ABI | Versioned package contains graph schedule, kernel entry points, one memory-mappable tensor archive, and cache layout; OCaml validates it | Package ABI v11 retains ABI-v2 through ABI-v10 reads. Package checks validate 810-command/60-entry prefill and 864-command/58-entry decode schedules against one 489,377,152-byte, 243-tensor binary archive; both Q8 and selectable FP16 cache policies validate | implemented for the captured full-Q8 pair |
| Metal compilation artifacts | Package build emits loadable metallib kernels for every scheduled model operation | The current full-Q8 replan compiles 68 prefill and 66 decode entries, adding paired-channel SIMD kernels to vector-staged Q8 prefill, packed Q8 decode and vocabulary projection, SIMD RMSNorm/attention, and cache conversion. One synthetic Metal run selects every paired epilogue family with 46/46 exact outputs, and the paired package executes the 4+4 model trace with exact eager IDs | implemented for captured full-Q8 templates |
| Native OCaml Metal runtime | Ninja-built OCaml executable selects a device, loads metallib functions, maps tensor storage, binds tensor views, and submits commands without Python or PyTorch in the serving hot path | One persistent `llmopt-serve` process loads both paired full-Q8 stages and the inode-keyed archive. One bounded run completes 4/4 warmup and scored requests with exact eager tokens, 80/194 reuse, ERS `0.40701575836615456`, and median TTFT/TPOT `75.225/6.937 ms` | partial |
| Model data ownership | OCaml loads package weights and persistent activations in the declared Q8/FP16 layouts | The full-Q8 run maps one 243-tensor archive containing FP16 embedding plus Q8 head weight/scale. Separate bounded traces execute Q8 and FP16 physical KV/recurrent pools across repeated decode with exact matching token IDs and identical 80/194 reuse | implemented for both cache formats on the serial trace |
| Tokenization, sampling, and serving protocol | OCaml accepts the benchmark request contract, applies the LFM chat template/tokenizer, streams generated tokens, and reports cache usage | `llmopt-serve` accepts the OpenAI-compatible chat contract, incrementally decodes UTF-8, streams every generated token ID plus visible text, and reports usage. The warmed scored smoke completed 4/4 requests with pinned output counts | implemented for the HTTP smoke contract |
| Mandatory radix-prefix reuse | Multi-turn requests produce non-zero cached-prefix accounting and reuse the matched KV/recurrent checkpoint while preserving output parity | Scored second turns reuse 42/61 and 38/59 prompt tokens; total reuse is 80/194 while all four output sequences match eager Q8 exactly. The dependent suffix plan reserves and inserts one checkpoint per suffix token and now completes the measured path | implemented for serial multi-turn smoke requests |
| Configurable KV quantization | FP16 and Q8 runs bind physical Metal KV/checkpoint buffers and execute matching quantize/dequantize paths | Q8-group-64 remains the default. Separate bounded paired-package model runs execute Q8 and selectable FP16 attention KV/recurrent storage, preserve the same four eager token sequences and 80/194 reuse, and record both latency reports | implemented for the serial 4+4 trace |
| Benchmark correctness and measurement | Exact logits/token parity, retrieval and response-format results, request counts, raw TTFT/TPOT, ERS, and cache-hit accounting are written by reproducible commands | The current paired full-Q8 trace records 4/4 exact native/eager scored sequences, native/established-eager ERS `0.40701575836615456/0.3663754874502978`, median native TTFT/TPOT `75.225/6.937 ms`, and 80/194 cached prompt tokens. The paired long matrix retrieves 6/6 and matches all 12 established eager-Q8 IDs, while exact-only text is 0/6 | partial |

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
