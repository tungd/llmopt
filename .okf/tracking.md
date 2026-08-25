---
type: Research Tracking
title: 'llmopt research register'
description: 'The ordered compiler slices, evidence state, and unresolved integration questions.'
tags: [tracking, research, roadmap, evidence]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T07:07:41Z' }
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
| OCaml FX importer and effect planner | implemented for captured prefill/decode templates | typed operations survive schedule-v11 round trips; the LFM specialization pass rewrites sequence-dependent dimensions and scalars, then re-infers SSA shapes while preserving static archive tensors |
| LLVM textual emitter | implemented | `clang -x ir` accepts the linear smoke |
| Metal source emitter | implemented | Xcode `metal` accepts the linear smoke |
| Fused LFM RMSNorm pass and Metal kernel | implemented; real-model count pending | synthetic LFM chain fuses from 10 commands to four; float32-to-float16 and float16 kernels pass `rms-norm-smoke` |
| Direct FX GraphModule MPS callable returned to PyTorch | implemented | fixed direct-forward logits match eager MPS exactly; generation routing is now explicit |
| LFM2.5 short-convolution lowering | typed, compiled, and native-dispatched | all ten saved prefill `conv1d` nodes lower to ShortConv commands; the shared native probe executes the same kernel ABI and matches the 12-element fixture output exactly |
| LFM2.5 GQA/KV-cache lowering | variable decode integrated | all six saved SDPA nodes lower to masked-attention commands; three consecutive Q8 decode steps unpacked matched prefixes 6/7/8 and appended one radix-owned position per step |
| LFM2.5 token embedding lowering | typed, compiled, and native-dispatched | the int64-to-float16 lookup lowers to a validated command and the shared native probe gathers four float16 elements exactly |
| LFM2.5 position and mask lowering | typed, compiled, and native-dispatched | five aranges, prepended diff, bool-to-int64 cumsum, scalar bool fill, and two broadcast gathers lower through schedule v7; exact CPU references and the shared native probe pass, while the exact unused PyTorch telemetry call is elided |
| model weight loading for the MPS probe | implemented | Transformers checkpoint loads on MPS |
| end-to-end PyTorch MPS comparison | implemented | short smoke proves routed generation; semantic 5x3 result has exact fixed-forward digest and exact generated-token parity |
| ERS trace/report benchsuite | implemented; 350M baseline recorded | racebench score math, reference-style HTTP runner, shape-matched semantic 5x3 and full 70x6 profiles, distinct warmup, isolated reports, exact token-ID parity, and `/bench/results/lfm25-350m-racebench-baseline.json` with `engine_pass: true`, eager ERS `0.0003597708408867709`, and 15/15 successful requests per candidate |
| LFM2.5-350M memory-safe benchmark path | implemented; engine pass and baseline recorded | `bench-suite` completed 15/15 warmup and scored requests per candidate, exact token/digest parity, eager ERS `0.0003597708408867709` |
| Q8 weight-only linear optimizer/codegen | implemented; 350M Q8 fallback run recorded | `Lfm25.Config.default` and model-level runners select Q8 weight-only linear lowering; CPU reference, Q8 IR, Python model rewrite, FX boundary, Metal `char` emitter, LLVM `i8` emitter, and `ninja -f ninja.build q8-smoke` pass; the bounded Q8 result has exact digest/token parity, and its saved outputs prove 6/6 control-code retrieval with 0/6 exact-only formatting |
| Q8 linear-SiLU epilogue fusion | implemented and compiled for captured templates; model measurement open | the alias-safe pass fuses 16 sole-consumer pairs per stage, reducing prefill/decode to 856/910 commands. Schedule/package ABI v9 round-trips the operation; both model metallibs compile. One 41-output fixture selected fused GEMV exactly, while the final explicit-half-rounding 42-output comparison remains static-only |
| Q8 linear-residual epilogue fusion | implemented and compiled for captured templates; device/model measurement open | the alias- and shape-safe pass fuses all 32 same-shape pairs per stage, reducing prefill/decode to 824/878 commands while retaining 15 unrelated adds. Schedule/package ABI v10, MSL, LLVM inspection, runtime binding, package validation, and the static materialized-reference fixture pass |
| Q8 multiplied-input down-projection fusion | implemented and compiled for captured templates; device/model measurement open | the alias-safe pass absorbs all 16 sole-consumer SwiGLU products per stage, reducing prefill/decode to 808/862 commands and captured-template workspace to 1,098,496/262,144 bytes. Schedule/package ABI v11, MSL, LLVM inspection, runtime buffer order, package validation, and the unlaunched materialized-reference fixture pass |
| SIMD-group Q8 decode GEMV | implemented and compiled; device/model measurement open | each 32-lane SIMD group reduces one output channel, with eight channels per 256-thread group across all four Q8 families and both dtypes. ABI-v11 model packages compile with the new names; runtime lookup retains scalar fallback for older packages. The changed reduction order has not run on device |
| generated Q8 Metal runtime loading and dispatch | implemented; exact model path verified; native numerical parity remains open | Ninja builds the PyTorch MPS C++ bridge, links the generated `.metallib`, and the Python FX backend selects generated exact dequantization or Phase 2 native Q8 entry points. The combined 350M differential probe records 92 exact-mode generated dispatches with `max_abs=0`, `mean_abs=0`, and 92 native Phase 2 dispatches with `max_abs=0.078125`, `mean_abs=0.00713115930557251`; no ERS result was written |
| OCaml serving radix/KV cache | serial multi-turn request integration implemented; batched suffix static-only | the warmed HTTP smoke reused 42/61 and 38/59 second-turn prompt tokens, reported 80/194 total cache reuse, and retained four exact eager-Q8 output sequences. The dependent suffix implementation reserves one checkpoint per token, but its first model attempt failed before scoring and the corrected path has not run on device |
| Versioned generated package ABI | implemented for captured ABI-v11 templates | Package ABI v11 retains ABI-v2 through ABI-v10 reads and adds typed Q8 multiplied-input/residual dispatch. Binary-input replanning writes 808-command/60-entry prefill and 862-command/58-entry decode templates with zero opaque operations and 241 validated bindings each |
| OCaml tensor-store ownership | partial; shared real JSON-free archive validated | Dynamo streams static inputs into a versioned binary index plus 256-byte-aligned payloads. A capture session now seals one 422,137,216-byte `weights.llmopt`, canonicalizes aliases by tensor storage identity, and hard-links that archive across prefill and decode graph directories; both packages validate every dtype/shape binding |
| OCaml Metal serving loader and dispatch | batched schedules/cache phases and shape-selected Q8 decode execute; SIMD and dependent suffix static-only; exact logits open | the latest successful decode uses three ordered submissions with the earlier scalar GEMV. ABI-v11 packages now prefer SIMD-group GEMV and fall back to scalar entries for older packages, but the new reduction order is unlaunched. The corrected suffix implementation and per-token cache writes remain static-only; Q8/FP16 cache bytes remain exact and the latest valid HTTP ERS is `0.11381808711306604` |
| Complete 350M operation schedule | captured templates specialize and repeated decode has pre-fusion token parity | ABI-v11 replans retain zero opaque commands and 241 bindings while removing 16 Q8/SiLU, 32 Q8/residual, and 16 multiplied-input dispatch boundaries from each stage. Offline planning covers prefill 13/128/4,096 and decode-past 1/127/4,095; the latest model run predates all three fusions and executes three growing decode lengths with four-token eager parity |
| native tokenization, LFM chat, generation, and HTTP/SSE | implemented for serial smoke requests | `LLMOPTTK` and typed chat feed one persistent `llmopt-serve` engine. Incremental UTF-8 events carry exact token IDs; the latest valid 4/4 warmed scored trace has pinned outputs, exact eager-Q8 parity, ERS `0.11381808711306604`, and 80/194 cached prompt tokens. The later suffix attempt stopped after 2/4 warmup requests and has no score |
| natural needle-in-a-haystack validation | native retrieval implemented; fixed-output native parity open | eager/direct-FX 2,048/4,096-token contexts retrieve `RAVEN-4271` in 6/6 while fixed 12-token outputs append text. Native stop-on-EOS requests retrieve exact text in 6/6 and match the first seven eager IDs; the runner now pins 12 tokens by default but that corrected long matrix was not rerun |

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
- Which symbolic-dimension representation should replace the current
  LFM-specific captured-template substitution when the compiler expands beyond this target?
