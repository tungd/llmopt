---
type: Research Tracking
title: 'llmopt research register'
description: 'The ordered compiler slices, evidence state, and unresolved integration questions.'
tags: [tracking, research, roadmap, evidence]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-24T00:03:24Z' }
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
| Dynamo-to-OCaml compiler transport | implemented | default capture writes `LLMOPTFX` ABI-v1 `graph.llmopt`; OCaml parses manifest-v2 typed fields and rejects malformed/truncated/trailing data. The preserved prefill/decode graphs round-trip exactly at 253,354/259,928 bytes versus 776,844/796,970 JSON bytes; JSON emission is opt-in |
| OCaml FX importer and effect planner | implemented for the captured prefill and one-token decode vocabularies | typed arguments, ellipsis, N-dimensional shapes, pointwise/reduction/cast/movement primitives, cache crop/fill/copy, roll, functional slice update, sum, deferred chunk/getitem slices, concat identities, RMSNorm, position/mask construction, and telemetry elision survive real v2 captures and a schedule-v8 round trip |
| LLVM textual emitter | implemented | `clang -x ir` accepts the linear smoke |
| Metal source emitter | implemented | Xcode `metal` accepts the linear smoke |
| Fused LFM RMSNorm pass and Metal kernel | implemented; real-model count pending | synthetic LFM chain fuses from 10 commands to four; float32-to-float16 and float16 kernels pass `rms-norm-smoke` |
| Direct FX GraphModule MPS callable returned to PyTorch | implemented | fixed direct-forward logits match eager MPS exactly; generation routing is now explicit |
| LFM2.5 short-convolution lowering | typed, compiled, and native-dispatched | all ten saved prefill `conv1d` nodes lower to ShortConv commands; the shared native probe executes the same kernel ABI and matches the 12-element fixture output exactly |
| LFM2.5 GQA/KV-cache lowering | fixed prefill/decode integrated | all six saved SDPA nodes lower to masked-attention commands; one full Q8 run packed six prefill positions, unpacked them for decode, and appended only position six into a new radix-owned slot |
| LFM2.5 token embedding lowering | typed, compiled, and native-dispatched | the int64-to-float16 lookup lowers to a validated command and the shared native probe gathers four float16 elements exactly |
| LFM2.5 position and mask lowering | typed, compiled, and native-dispatched | five aranges, prepended diff, bool-to-int64 cumsum, scalar bool fill, and two broadcast gathers lower through schedule v7; exact CPU references and the shared native probe pass, while the exact unused PyTorch telemetry call is elided |
| model weight loading for the MPS probe | implemented | Transformers checkpoint loads on MPS |
| end-to-end PyTorch MPS comparison | implemented | short smoke proves routed generation; semantic 5x3 result has exact fixed-forward digest and exact generated-token parity |
| ERS trace/report benchsuite | implemented; 350M baseline recorded | racebench score math, reference-style HTTP runner, shape-matched semantic 5x3 and full 70x6 profiles, distinct warmup, isolated reports, exact token-ID parity, and `/bench/results/lfm25-350m-racebench-baseline.json` with `engine_pass: true`, eager ERS `0.0003597708408867709`, and 15/15 successful requests per candidate |
| LFM2.5-350M memory-safe benchmark path | implemented; engine pass and baseline recorded | `bench-suite` completed 15/15 warmup and scored requests per candidate, exact token/digest parity, eager ERS `0.0003597708408867709` |
| Q8 weight-only linear optimizer/codegen | implemented; 350M Q8 fallback run recorded | `Lfm25.Config.default` and model-level runners select Q8 weight-only linear lowering; CPU reference, Q8 IR, Python model rewrite, FX boundary, Metal `char` emitter, LLVM `i8` emitter, and `ninja -f ninja.build q8-smoke` pass; the bounded Q8 result has exact digest/token parity, and its saved outputs prove 6/6 control-code retrieval with 0/6 exact-only formatting |
| generated Q8 Metal runtime loading and dispatch | implemented; exact model path verified; native numerical parity remains open | Ninja builds the PyTorch MPS C++ bridge, links the generated `.metallib`, and the Python FX backend selects generated exact dequantization or Phase 2 native Q8 entry points. The combined 350M differential probe records 92 exact-mode generated dispatches with `max_abs=0`, `mean_abs=0`, and 92 native Phase 2 dispatches with `max_abs=0.078125`, `mean_abs=0.00713115930557251`; no ERS result was written |
| OCaml serving radix/KV cache | fixed model integration implemented; request integration open | the full Q8 run recorded one six-token radix hit, seven cached physical slots, and two recurrent checkpoints after prefill plus decode; ownership and reinsertion rollback tests pass |
| Versioned generated package ABI | implemented for fixed ABI-v8 pair | Package ABI v8 retains ABI-v2 through ABI-v7 reads and adds sliced cache writes. Binary-input replanning writes 872-command/46-entry prefill and 926-command/44-entry decode packages with zero opaque operations and 241 validated bindings each |
| OCaml tensor-store ownership | partial; shared real JSON-free archive validated | Dynamo streams static inputs into a versioned binary index plus 256-byte-aligned payloads. A capture session now seals one 422,137,216-byte `weights.llmopt`, canonicalizes aliases by tensor storage identity, and hard-links that archive across prefill and decode graph directories; both packages validate every dtype/shape binding |
| OCaml Metal serving loader and dispatch | complete fixed Q8 schedules execute; batching and parity open | one shared-context run maps the archive once by inode, dispatches 522 prefill and 544 decode commands, returns two sampled token IDs, and leaves consistent radix/KV ownership |
| Complete 350M operation schedule | fixed prefill and one-token decode native-executed | ABI-v8 replans retain zero opaque commands and 241 bindings. One full run uses the 1,153,792/271,360-byte workspaces and completes in 0.432272/0.119545 seconds; exact PyTorch output comparison is not yet recorded |
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
- Which vocabulary-projection tile and reduction order best balance prefill and
  one-token decode before batched command-buffer submission?
- How should fixed six-token prefill and one-token decode specializations become
  variable-length, persistent generation without recapturing every sequence length?
