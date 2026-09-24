---
type: Research Goal
title: 'Graph-Captured GGUF/UD Serving Across Model Families'
description: 'Track the architecture-neutral trajectory from PyTorch Dynamo FX capture through GGUF/UD tensor ingestion, AOT megakernel fusions, zero-JIT Metal dispatch, and continuous batching serving under Model Program ABI v7.'
tags: [goal, compiler, pytorch, fx, gguf, unsloth, metal, serving, continuous-batching, megakernels]
status: stable
generated: { by: 'autoagent/union-alpha', at: '2026-09-24T05:37:07Z' }
sources:
  - id: frontend
    resource: /python/llmopt_backend/__init__.py
    title: torch.compile capture and explicit external tensor binding
  - id: compiler
    resource: /bin/fx_compile.ml
    title: OCaml FX compiler entry point
  - id: model-program
    resource: /lib/model_program.ml
    title: Model Program ABI v7
  - id: graph-decision
    resource: /decisions/model-program-boundary.md
    title: Graph authority and Model Program decision
  - id: quant-decision
    resource: /decisions/gguf-unsloth-dynamic-quantization.md
    title: GGUF and Unsloth Dynamic quantization decision
  - id: fewest-hops
    resource: /decisions/fewest-hops-megakernel-compiler.md
    title: Fewest-Hops Whole-Block Megakernel Compiler
  - id: package
    resource: /lib/serving_package.ml
    title: Native serving package (ABI v27)
  - id: schedule
    resource: /lib/serving_schedule_codec.ml
    title: Native serving schedule codec (format v29; additive diagnostic opaque opcode)
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Native OCaml Metal runtime
  - id: tactic-registry
    resource: /lib/kernel_abi.ml
    title: Shared shape-aware Linear tactic registry
  - id: memory-plan
    resource: /lib/serving_memory_plan.ml
    title: Static interval-graph liveness memory planner
  - id: tokenizer
    resource: /lib/tokenizer.ml
    title: Native tokenizer archive
  - id: chat
    resource: /lib/chat_template.ml
    title: Explicit chat template contract
  - id: queue
    resource: /lib/serving_queue.ml
    title: Continuous batching queue with age-weighted SRPT scheduler
  - id: sampling
    resource: /lib/sampling.ml
    title: Zero-allocation ARM NEON SIMD stochastic sampler
  - id: server
    resource: /bin/serve.ml
    title: Generic HTTP serving entry point
  - id: full-model-comparison
    resource: /experiments/exp-0101-gguf-full-model-comparison-2026-08-29.md
    title: Historical captured full-model GGUF comparison
  - id: slice-31-receipt
    resource: /bench/results/compiler-generalization-slice-31-2026-08-30.json
    title: Historical cross-model benchmark receipt
  - id: smollm-matched-sustained
    resource: /bench/results/smollm-slice31-llamacpp-matched-trace-probe-2026-09-01.json
    title: SmolLM2 matched-token sustained LLMOpt and llama.cpp comparison
  - id: smollm-matched-q8kv
    resource: /bench/results/smollm-slice31-llamacpp-q8kv-matched-trace-probe-2026-09-01.json
    title: SmolLM2 matched-token Q8-KV LLMOpt and llama.cpp comparison
  - id: build
    resource: /ninja.build
    title: Ninja build graph
---

# 1. Objective

Build an optimizing LLM compiler and zero-JIT serving system whose frontend is
PyTorch `torch.compile` / Dynamo FX and whose execution data plane is OCaml plus
Metal on Apple Silicon unified memory:

1. **Topology Authority in Captured Graphs**: PyTorch FX capture defines graph
   topology, operation semantics, and tensor dataflow. The compiler and serving
   runtime contain no model-family selectors or hardcoded layer loops.
2. **First-Class GGUF & Unsloth Dynamic Quantization**: Ingest real quantized
   model distributions from GGUF files and specialize dequantization and linear
   tactics at compile time.
3. **Fewest-Hops SRAM Megakernels**: Fuse pointwise, normalization, attention,
   recurrent, and contraction work where legality and target cost permit it,
   without creating a model-specific kernel zoo.
4. **Zero-Overhead Serving Runtime**: Host continuous batching, radix prefix
   caching, precompiled AOT Metal command lists, and ARM NEON sampling under
   the unified `model.llmopt` ABI v7 contract.

# 2. Current Implementation and Evidence

| Area | Current state | Evidence boundary |
|---|---|---|
| FX capture | Implemented | Binary `graph.llmopt` transport and Python capture tests pass |
| OCaml IR and dependency analysis | Implemented | `Ir`, `Compute_plan`, constructive regions, and separate antichain stages are exercised by the test suite |
| Typed contraction groups | Implemented | Single-slot, multi-slot, and multi-export groups have algebraic, schedule, and Metal device coverage |
| Model Program contract | Implemented | ABI v7 is linked by the pipeline, cache, server, checks, and MTP/prompt-lookup metadata |
| GGUF and block-quant ingestion | Implemented for the documented formats | Evidence is scoped to tested model assets, shapes, and quant layouts |
| Metal runtime | Implemented for the current backend | Vulkan is a separate experimental probe, not a second full-model backend |
| Cache and queue | Implemented primitives and serving paths | Multi-tenant saturation evidence is not yet a universal claim |
| Sampling and HTTP server | Implemented | Parameterized sampling and OpenAI-compatible streaming code are tested |
| Cross-model performance | Scoped probe evidence | Small-model rows are historical measurements; the Gemma 4 12B UOC result is not within the requested llama.cpp envelope |
| Unsupported FX fallback | Not established | Unknown operations currently remain a hard executable/coverage boundary |

The current typed-assignment transition is recorded in
[Constructive Assignment Lowering and AOT GPU Command-Queue Compilation](/decisions/constructive-primitive-loop-lowering-and-gpu-queue-compilation.md).
The live implementation is stronger than the historical transition notes: typed
multi-slot and multi-export contraction groups now reach Metal emission and
runtime dispatch.

# 3. Historical Milestones

The project has completed the major compiler and serving slices that led to
the current state. Historical receipts remain evidence of the changes that
introduced each capability:

- GGUF ingestion, quant dequantization, layout provenance, and block-quant
  linear tactics.
- Gated Linear, RMSNorm, rotary QK, ShortConv, recurrence recovery, and DAG
  co-scheduling passes.
- Binary FX transport, typed IR, model-linked packages, tokenizer/chat assets,
  physical cache state, radix caching, and continuous queue primitives.
- AOT package and schedule formats, static liveness memory planning, direct
  Metal dispatch, and parameterized sampling.
- The large-file modular refactor and typed-contraction-group round, including
  broad block-quant codegen and multi-export execution.
- The Gemma 4 12B UOC path, whose latest recorded package reduces decode
  commands substantially and executes on device; the FFN-down x16 result remains
  opt-in and experimental.

Historical experiment files retain the ABI values, benchmark protocol, and
claims that were true at the time of the experiment. They are not rewritten by
this goal document.

# 4. Current Evidence Boundary

Performance and correctness claims must be read per receipt. In particular:

- A matching argmax token is not a full-logit parity claim.
- A command-count reduction is not a latency improvement by itself.
- A single-device probe is not a general model-adoption result.
- A llama.cpp comparison is valid only when model asset, token protocol,
  precision, harness, and measurement scope match.
- Gemma 4 12B UOC execution is not the same claim as arbitrary-model support.
- The explicit `LLMOPT_EXPERIMENTAL_Q4_DOWN_X16` override remains disabled by
  default. The pre-cache-ownership-fix 32-token A/B selected x16 in both modes
  and matched output; its 128-token comparison diverged at token 6 and remains
  historical. After the cache ownership fix, fresh default and explicit-x16
  128-token runs selected x16 96 times each and produced the same completion
  hash, but still differ from the preserved exp-0172 golden stream at token 6;
  no full-model golden-parity or selector-speed claim is made.

The live status and implementation order are maintained in
`CURRENT_STATUS.md` at the repository root.

# 5. Agreed Next Horizons

1. Keep the primary `ninja -f ninja.build all` and `test` gates green and make
   recovery from installed-interface changes less dependent on manual cleanup.
2. Measure adoption cost for each new model: opaque nodes, new primitives,
   new fusion strategies, new backend tactics, first correct token, parity
   boundary, and same-harness llama.cpp ratio.
3. Keep primitive semantics, fusion strategies, and backend tactics explicitly
   separate; report unsupported FX nodes with actionable diagnostics.
4. Continue quantized-projection bandwidth work, especially gated and FFN-down
   shapes, and feed measured relationships into the cost model.
5. Keep ROCm/HIP and other backend portability work deferred until the current
   compiler contract, adoption evidence, and Metal measurement path are stable.
