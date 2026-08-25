---
type: Research Tracking
title: 'llmopt research register'
description: 'The ordered compiler slices, evidence state, and unresolved integration questions.'
tags: [tracking, research, roadmap, evidence]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T08:40:41Z' }
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
| Executable LFM feed-forward shape | corrected and covered | the checkpoint declaration remains 6656 while `block_auto_adjust_ff_dim` produces 4608; all captured projections and OCaml model-shaped fixtures now agree on 4608 |
| Fused LFM RMSNorm pass and Metal kernel | implemented and model-executed in aggregate | synthetic LFM chain fuses from 10 commands to four; preserved prefill/decode templates each contain 45 fused commands. Both dtype variants assign one SIMD group per row and retain scalar fallback; the combined optimized run preserves exact model tokens without isolating RMSNorm |
| Direct FX GraphModule MPS callable returned to PyTorch | implemented | fixed direct-forward logits match eager MPS exactly; generation routing is now explicit |
| LFM2.5 short-convolution lowering | typed, compiled, and native-dispatched | all ten saved prefill `conv1d` nodes lower to ShortConv commands; the shared native probe executes the same kernel ABI and matches the 12-element fixture output exactly |
| LFM2.5 GQA/KV-cache lowering | variable decode integrated and model-executed | all six saved SDPA nodes lower to masked-attention commands; one SIMD group computes each width-64 query row with single-pass online softmax. The combined optimized run executes it across 4/4 exact scored requests and retains 80/194 radix reuse; the aggregate measurement does not isolate attention |
| LFM2.5 token embedding lowering | typed, compiled, and native-dispatched | the int64-to-float16 lookup lowers to a validated command and the shared native probe gathers four float16 elements exactly |
| LFM2.5 position and mask lowering | typed, compiled, and native-dispatched | five aranges, prepended diff, bool-to-int64 cumsum, scalar bool fill, and two broadcast gathers lower through schedule v7; exact CPU references and the shared native probe pass, while the exact unused PyTorch telemetry call is elided |
| model weight loading for the MPS probe | implemented | Transformers checkpoint loads on MPS |
| end-to-end PyTorch MPS comparison | implemented | short smoke proves routed generation; semantic 5x3 result has exact fixed-forward digest and exact generated-token parity |
| ERS trace/report benchsuite | implemented; 350M baseline recorded | racebench score math, reference-style HTTP runner, shape-matched semantic 5x3 and full 70x6 profiles, distinct warmup, isolated reports, exact token-ID parity, and `/bench/results/lfm25-350m-racebench-baseline.json` with `engine_pass: true`, eager ERS `0.0003597708408867709`, and 15/15 successful requests per candidate |
| LFM2.5-350M memory-safe benchmark path | implemented; engine pass and baseline recorded | `bench-suite` completed 15/15 warmup and scored requests per candidate, exact token/digest parity, eager ERS `0.0003597708408867709` |
| Q8 weight-only linear optimizer/codegen | implemented; 350M Q8 fallback run recorded | `Lfm25.Config.default` and model-level runners select Q8 weight-only linear lowering; CPU reference, Q8 IR, Python model rewrite, FX boundary, Metal `char` emitter, LLVM `i8` emitter, and `ninja -f ninja.build q8-smoke` pass; the bounded Q8 result has exact digest/token parity, and its saved outputs prove 6/6 control-code retrieval with 0/6 exact-only formatting |
| Q8 linear-SiLU epilogue fusion | implemented and model-executed in aggregate | the alias-safe pass fuses 16 sole-consumer pairs per stage, reducing prefill/decode to 856/910 commands. Schedule/package ABI v9 and the exact small fixture pass; the combined optimized model run preserves exact tokens without isolating this fusion |
| Q8 linear-residual epilogue fusion | implemented and model-executed in aggregate | the alias- and shape-safe pass fuses all 32 same-shape pairs per stage, reducing prefill/decode to 824/878 commands while retaining 15 unrelated adds. ABI v10 and static materialized-reference checks pass; the combined optimized model run preserves exact tokens without isolating this fusion |
| Q8 multiplied-input down-projection fusion | implemented and model-executed in aggregate | the alias-safe pass absorbs all 16 sole-consumer SwiGLU products per stage, reducing prefill/decode to 808/862 commands and captured-template workspace to 1,098,496/262,144 bytes. ABI v11 and static buffer-order/reference checks pass; the combined optimized model run preserves exact tokens without isolating this fusion |
| SIMD-group Q8 decode GEMV | implemented and model-executed in aggregate | each 32-lane SIMD group reduces one output channel, with eight channels per 256-thread group across all four Q8 families and both dtypes. The combined optimized run retains 4/4 exact scored tokens; scalar fallback remains available, and the aggregate measurement does not isolate GEMV |
| Packed SIMD Q8 lane loads | implemented, compiled, and model-executed | all eight decode variants load activation4/char4 vectors and use one float4 dot per lane iteration, reducing loop iterations by four for k=1024/4608. One bounded 350M run preserves 4/4 scored eager-Q8 token sequences and 80/194 reuse while observing ERS `0.3253700872862615`, median TTFT `95.601 ms`, and median TPOT `7.933 ms` |
| Vector-staged Q8 prefill | implemented, compiled, and model-executed | the 16 by 16 output tile stages 64 reduction elements as activation4 and dequantized-weight4 vectors. Emitted barriers per k=1024/4608 tile change from 128/576 to 32/144; one bounded 350M run preserves 4/4 scored eager-Q8 sequences and 80/194 reuse while observing ERS `0.3377415731686302`, median TTFT `93.156 ms`, and median TPOT `7.948 ms` |
| Serving-only last-token vocabulary projection | implemented and statically validated | the typed LFM prefill specializer recognizes the sole-consumer identity-index plus LM-head chain and projects `[1,1,65536]` instead of `[1,tokens,65536]`. The real 350M package reports one output row at 13/128/4,096 tokens; the 4,096-token output allocation is 131,072 bytes and total workspace is 184,680,448 bytes. No model run has measured the latency delta yet |
| SIMD-group RMSNorm | implemented and model-executed in aggregate | one 32-lane SIMD group reduces and writes each row, with eight rows per 256-thread group for both dtype variants. The 45 commands in each 350M stage execute in the combined 4/4 exact scored run; scalar fallback remains available, and the aggregate measurement does not isolate RMSNorm |
| Single-pass SIMD attention | implemented and model-executed in aggregate | one 32-lane SIMD group computes each width-64 query row, reducing each query-key score once and accumulating softmax-weighted values online. All six commands per stage select it; the combined optimized run retains exact scored tokens, and scalar fallback remains declared |
| generated Q8 Metal runtime loading and dispatch | implemented; exact model path verified; native numerical parity remains open | Ninja builds the PyTorch MPS C++ bridge, links the generated `.metallib`, and the Python FX backend selects generated exact dequantization or Phase 2 native Q8 entry points. The combined 350M differential probe records 92 exact-mode generated dispatches with `max_abs=0`, `mean_abs=0`, and 92 native Phase 2 dispatches with `max_abs=0.078125`, `mean_abs=0.00713115930557251`; no ERS result was written |
| OCaml serving radix/KV cache | dependent cached-suffix batching implemented and measured | the corrected runtime reserves one checkpoint per suffix token and completes 4/4 scored requests with 42/61 and 38/59 second-turn reuse, 80/194 total cache reuse, and exact eager-Q8 output sequences |
| Versioned generated package ABI | implemented for captured ABI-v11 templates | Package ABI v11 retains ABI-v2 through ABI-v10 reads and adds typed Q8 multiplied-input/residual dispatch. Binary-input replanning writes 808-command/61-entry prefill and 862-command/59-entry decode templates with zero opaque operations and 241 validated bindings each; the extra entry is scalar attention fallback |
| OCaml tensor-store ownership | partial; shared real JSON-free archive validated | Dynamo streams static inputs into a versioned binary index plus 256-byte-aligned payloads. A capture session now seals one 422,137,216-byte `weights.llmopt`, canonicalizes aliases by tensor storage identity, and hard-links that archive across prefill and decode graph directories; both packages validate every dtype/shape binding |
| OCaml Metal serving loader and dispatch | vector-prefill plus packed fused/SIMD schedules and dependent suffix batches model-executed; native logits measured | ABI-v11 packages prefer vector-staged Q8 prefill, packed SIMD-group Q8 decode, RMSNorm, and attention with scalar fallback. One bounded run completes 4/4 warmup and scored requests, preserves exact tokens and Q8 radix counts, and measures native ERS `0.3377415731686302`; a separate six-token prefill comparison preserves eager-Q8 argmax `19130` with `max_abs=0.078125` and `mean_abs=0.014548537321388721` |
| Complete 350M operation schedule | captured templates specialize and current optimized schedule has token parity | ABI-v11 replans retain zero opaque commands and 241 bindings while removing 16 Q8/SiLU, 32 Q8/residual, and 16 multiplied-input dispatch boundaries per stage. The 808/862-command pair executes 4/4 scored requests with exact eager-Q8 tokens; offline planning covers prefill 13/128/4,096 and decode-past 1/127/4,095, with prefill vocabulary projection reduced to one output row at every checked length |
| native tokenization, LFM chat, generation, and HTTP/SSE | implemented for serial smoke requests | `LLMOPTTK` and typed chat feed one persistent `llmopt-serve` engine. The current 4/4 warmup and scored trace has pinned outputs, exact eager-Q8 parity, ERS `0.3377415731686302`, and 80/194 cached prompt tokens |
| natural needle-in-a-haystack validation | native fixed-output retrieval and token parity measured | the optimized native 2,048/4,096-token matrix retrieves `RAVEN-4271` in 6/6 and exactly matches all 12 eager-Q8 IDs in every request. Exact-only formatting is 0/6 because the fixed output decodes as `RAVEN-4271Lottery`; median TTFT/TPOT is 2,859.034/34.252 ms and 6,540.756/65.629 ms respectively |

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
