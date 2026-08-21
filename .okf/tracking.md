---
type: Research Tracking
title: 'llmopt research register'
description: 'The ordered compiler slices, evidence state, and unresolved integration questions.'
tags: [tracking, research, roadmap, evidence]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-21T09:00:28Z' }
sources:
  - id: repository-build
    resource: /ninja.build
    title: Ninja build graph
  - id: benchmark-protocol
    resource: /bench/README.md
    title: benchmark setup and measurements
---

# Ordered slices

| Slice | State | Evidence |
|---|---|---|
| Ninja-built OCaml 5 effect/IR prototype | implemented | `ninja test` passes |
| Python Dynamo/FX manifest exporter | implemented | Python unittest passes |
| OCaml FX importer and effect planner | implemented | static linear manifest plans |
| LLVM textual emitter | implemented | `clang -x ir` accepts the linear smoke |
| Metal source emitter | implemented | Xcode `metal` accepts the linear smoke |
| Direct FX GraphModule MPS callable returned to PyTorch | implemented | fixed direct-forward logits match eager MPS exactly; generation routing is now explicit |
| LFM2.5 short-convolution lowering | open | executed as opaque FX nodes through PyTorch MPS |
| LFM2.5 GQA/KV-cache lowering | open | executed as opaque FX nodes through PyTorch MPS |
| model weight loading for the MPS probe | implemented | Transformers checkpoint loads on MPS |
| end-to-end PyTorch MPS comparison | implemented | short smoke proves routed generation; semantic 5x3 result has exact fixed-forward digest and exact generated-token parity |
| ERS trace/report benchsuite | implemented; 2.6B baseline recorded | racebench score math, reference-style HTTP runner, shape-matched semantic 5x3 and full 70x6 profiles, distinct warmup, isolated reports, exact token-ID parity, and `/bench/results/lfm25-racebench-baseline.json` with `engine_pass: true`, eager ERS `0.0`, and 15/15 successful requests per candidate |
| LFM2.5-350M memory-safe benchmark path | implemented; engine pass and baseline recorded | `bench-suite-350m` completed 15/15 warmup and scored requests per candidate, exact token/digest parity, eager ERS `0.0003597708408867709`; the 2.6B result is recorded separately in the ERS benchsuite slice |
| Q8 weight-only linear optimizer/codegen | implemented; 350M Q8 fallback run recorded | `Lfm25.Config.default` and model-level runners select Q8 weight-only linear lowering; CPU reference, Q8 IR, Python model rewrite, FX boundary, Metal `char` emitter, LLVM `i8` emitter, `ninja -f ninja.build q8-smoke`, and the historical dequantizing MPS callable probe pass; the bounded Q8 350M run records 15/15 requests per candidate, exact digest/token parity, and `0/6` needle retrieval at `/bench/results/lfm25-350m-q8-racebench-baseline.json` |
| generated Q8 Metal runtime loading and dispatch | implemented; model integration parity observation recorded | Ninja builds the PyTorch MPS C++ bridge, links the generated `.metallib`, and the Python FX backend selects half/float32 Q8 entry points; the corrected non-aligned `M=3,N=29,K=37` probe passes, while the bounded 350M FX integration reports `max_abs=0.03515625` and `mean_abs=0.004932403564453125` under the existing exact-logit check and writes no ERS result |
| natural needle-in-a-haystack validation | implemented | semantic 5x3 run records `0/6` for both candidates at 2,048/4,096-token contexts and 10/50/90 placement |

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
- What prompt/template and response budget should the LFM2.5-2.6B needle probe
  use so semantic retrieval is measured independently from explanatory output
  formatting?
- How much repeat/counterbalance sampling should be used when comparing MPS
  latency distributions after the isolated profile is recorded?
- What host-memory reservation or process-level model lifecycle is needed to
  run the full 2.6B racebench-shaped profile without driving unified-memory
  pressure down during the measurement?
- Which LFM2.5 linear subgraphs can use the generated Q8 callable without
  falling back to PyTorch dequantization?
- How should model-level Q8 correctness be reported when a generated float32
  kernel has small logit drift but may preserve generated token IDs?
