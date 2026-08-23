---
type: Research Tracking
title: 'llmopt research register'
description: 'The ordered compiler slices, evidence state, and unresolved integration questions.'
tags: [tracking, research, roadmap, evidence]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T18:05:24Z' }
sources:
  - id: repository-build
    resource: /ninja.build
    title: Ninja build graph
  - id: benchmark-protocol
    resource: /bench/README.md
    title: benchmark setup and measurements
---

# Ordered slices

The authoritative end state and requirement-level evidence are tracked in the
[complete OCaml serving goal](goal-serving-runtime.md).

| Slice | State | Evidence |
|---|---|---|
| Ninja-built OCaml 5 effect/IR prototype | implemented | `ninja test` passes |
| Python Dynamo/FX manifest exporter | implemented | v2 captures rank plus typed node, integer, float, bool, null, string, symbol, sequence, mapping, and slice arguments; Python unittest passes |
| OCaml FX importer and effect planner | implemented | typed arguments and N-dimensional logical shapes survive effect capture and binary schedule round-trip |
| LLVM textual emitter | implemented | `clang -x ir` accepts the linear smoke |
| Metal source emitter | implemented | Xcode `metal` accepts the linear smoke |
| Direct FX GraphModule MPS callable returned to PyTorch | implemented | fixed direct-forward logits match eager MPS exactly; generation routing is now explicit |
| LFM2.5 short-convolution lowering | open | executed as opaque FX nodes through PyTorch MPS |
| LFM2.5 GQA/KV-cache lowering | open | executed as opaque FX nodes through PyTorch MPS |
| model weight loading for the MPS probe | implemented | Transformers checkpoint loads on MPS |
| end-to-end PyTorch MPS comparison | implemented | short smoke proves routed generation; semantic 5x3 result has exact fixed-forward digest and exact generated-token parity |
| ERS trace/report benchsuite | implemented; 350M baseline recorded | racebench score math, reference-style HTTP runner, shape-matched semantic 5x3 and full 70x6 profiles, distinct warmup, isolated reports, exact token-ID parity, and `/bench/results/lfm25-350m-racebench-baseline.json` with `engine_pass: true`, eager ERS `0.0003597708408867709`, and 15/15 successful requests per candidate |
| LFM2.5-350M memory-safe benchmark path | implemented; engine pass and baseline recorded | `bench-suite` completed 15/15 warmup and scored requests per candidate, exact token/digest parity, eager ERS `0.0003597708408867709` |
| Q8 weight-only linear optimizer/codegen | implemented; 350M Q8 fallback run recorded | `Lfm25.Config.default` and model-level runners select Q8 weight-only linear lowering; CPU reference, Q8 IR, Python model rewrite, FX boundary, Metal `char` emitter, LLVM `i8` emitter, and `ninja -f ninja.build q8-smoke` pass; the bounded Q8 result has exact digest/token parity, and its saved outputs prove 6/6 control-code retrieval with 0/6 exact-only formatting |
| generated Q8 Metal runtime loading and dispatch | implemented; exact model path verified; native numerical parity remains open | Ninja builds the PyTorch MPS C++ bridge, links the generated `.metallib`, and the Python FX backend selects generated exact dequantization or Phase 2 native Q8 entry points. The combined 350M differential probe records 92 exact-mode generated dispatches with `max_abs=0`, `mean_abs=0`, and 92 native Phase 2 dispatches with `max_abs=0.078125`, `mean_abs=0.00713115930557251`; no ERS result was written |
| OCaml serving radix/KV cache | implemented | mandatory compressed radix cache, hybrid recurrent checkpoints, namespace isolation, protected leases, LRU leaf eviction, FP16/Q8 layout accounting, and owned slot allocation pass `ninja -f ninja.build ocaml-test` |
| Versioned generated package ABI | partial; real graph serialized | `llmopt-fx` emits `package.llmopt` with commands, logical shapes, arguments, kernel ABI, cache policy, and tensor bindings. The saved 350M graph became a 116,861-byte package with 1,115 commands and 241 validated bindings; 736 commands remain opaque and native interpretation remains open |
| OCaml tensor-store ownership | partial; full archive validated and fixture mapped | Dynamo streams static inputs one tensor at a time; the real 350M archive contains 241 tensors and 422,104,704 payload bytes. OCaml validates all binding dtypes/shapes, while the fixture proves no-copy Metal mapping; full-model dispatch is not implemented |
| OCaml Metal serving loader and dispatch | partial; binary-only startup verified | With 56% memory free, a minimal three-file directory containing only the binary package, metallib, and safetensors archive returned `[3.5, 8, 1, 1.5, 4, 2]` exactly on Apple M4 Pro; the model schedule still runs through Python/PyTorch |
| Complete 350M operation schedule | partial | The binary command ABI is implemented, but the exported no-cache forward still has 1,115 IR nodes: 379 typed/lowered and 736 opaque. Decode/KV-state graphs and native command interpretation remain open |
| natural needle-in-a-haystack validation | implemented; grader corrected | 2,048/4,096-token contexts at 10/50/90 placement retrieve `RAVEN-4271` in 6/6 outputs for both candidates; exact only-the-code formatting is separately 0/6 |

# Evidence rule

Each slice records the exact command and artifact used to observe it. A
measurement is evidence for comparison; this register does not turn a chosen
measurement into a release gate.

# Open questions

- Which FX decomposition boundary gives the cleanest LFM2.5 conv/GQA op set?
- How should symbolic sequence length be represented when Dynamo specializes or
  recompiles a graph?
- What additional Q8 tile shapes and launch policies should be selected for
  the LFM2.5 projection dimensions beyond the initial 16x16 kernel?
- How should generated libraries be versioned and invalidated when the FX
  graph, target device, or compiler flags change?
- What prompt/template and response budget should the LFM2.5-350M needle probe
  use so semantic retrieval is measured independently from explanatory output
  formatting?
- How much repeat/counterbalance sampling should be used when comparing MPS
  latency distributions after the isolated profile is recorded?
- Which LFM2.5 linear subgraphs can use the generated Q8 callable without
  falling back to PyTorch dequantization?
- Which reduction schedule or MPS-compatible matmul lowering can make the
  native Phase 2 float32 Q8 path match the exact generated dequantization path?
- Which typed view/layout operations should be represented as zero-cost
  schedule metadata before lowering the remaining compute nodes?
- What grouped-Q8 scale layout and Metal quantize/dequantize kernels should
  back the cache's current Q8 ownership and byte-accounting policy?
