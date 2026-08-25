---
type: Research Tracking
title: 'llmopt research register'
description: 'The ordered compiler slices, evidence state, and unresolved integration questions.'
tags: [tracking, research, roadmap, evidence]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T10:28:14Z' }
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
| Fused RMSNorm and RoPE | implemented, compiled, and model-executed | `Passes.fuse_rms_rope` folds twelve 10-command RMSNorm-RoPE subgraphs into single `Rms_rope` IR ops and `llmopt_rms_rope_f16_simd_h64` SIMD Metal kernels. Prefill commands reduce from 810 to 702 (74 package kernels), decode from 864 to 756 (72 package kernels), and specialized decode to 696. Bounded 350M execution preserves 4/4 eager tokens and 80/194 reuse while observing ERS `0.4122601696838274`, median TTFT `73.706 ms`, and median TPOT `7.060 ms` |
| Fused Short-Convolution block | implemented, compiled, and model-executed | `Passes.fuse_short_conv` folds 13-command prefill and 16-command decode conv subgraphs across all 10 conv layers into single `Short_conv_prefill` and `Short_conv_step` ops lowering to `llmopt_short_conv_prefill_f16` and `llmopt_short_conv_step_f16` SIMD Metal kernels. Prefill commands reduce from 702 to 592 (-110 commands, 74 kernels), decode commands reduce from 756 to 606 (-150 commands, 71 kernels), specialized decode reduces to 546 commands, and decode memory allocations drop from 390 to 262. Native M4 Pro execution preserves 4/4 eager short-trace tokens and 6/6 needle retrieval while observing ERS `0.510917` (+0.098657), median TTFT `56.511 ms` (-17.195 ms), and median TPOT `5.554 ms` (-1.506 ms) |
| DAG concurrency analysis & co-scheduling | implemented and compiled | `Pass_co_schedule` analyzes SSA graph DAG and CPM critical-path heights, partitions independent operations into concurrent stages, inserts stage barriers (`Barrier_wait 0`), and dispatches concurrently via `MTLDispatchTypeConcurrent` with concurrency-safe memory lifetime management |
| Modular compiler pass architecture | implemented | Decoupled passes into individual modules conforming to unified `PASS` signature, `Pass.t` record, and declarative `Pass.Pipeline.t` manager |
| SRPT & Continuous Serving Scheduler | implemented and verified | Replaced blocking FCFS server with M/G/1-FB continuous batching queue using SRPT priority scoring, starvation-free aging, and M/G/1/K watermark admission control. Tested with synthetic Poisson arrival traffic ($\lambda = 5.0\text{ req/s}$) yielding $P50\text{ TTFT} = 28.60\text{ ms}$, $P95\text{ TTFT} = 51.35\text{ ms}$, and $100\%$ exact token parity across concurrent multi-turn streams |
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
| Paired SIMD Q8 output channels | implemented, compiled, and model-executed | each SIMD group computes two adjacent channels and shares one packed activation load, changing the vocabulary projection from 8,192 to 4,096 threadgroups. The bounded 350M trace preserves exact tokens and 80/194 reuse while observing ERS `0.40701575836615456`, median TTFT `75.225 ms`, and median TPOT `6.937 ms`; versus the single-channel observation, those values change by `+0.016120`, `+3.748 ms`, and `-0.374 ms` |
| SIMD-group Q8 cache packing | implemented, compiled, and model-executed | one 32-lane SIMD group quantizes each attention or recurrent group, with eight groups per threadgroup and typed scalar fallback for older packages. Full-Q8 replanning emits 70/68-entry, zero-opaque packages and the small Metal invocation is exact. The bounded 350M trace preserves 4/4 eager tokens and 80/194 reuse while observing ERS `0.4021550914067862`, median TTFT `73.132 ms`, and median TPOT `7.308 ms`; versus the preceding paired-Q8 observation, those values change by `-0.004861`, `-2.093 ms`, and `+0.371 ms` |
| Vectorized Q8 cache unpack | implemented, compiled, and model-executed | each thread restores four adjacent int8 values with one char4 load, one shared FP16 scale load, and one half4 store. The bounded 350M trace preserves 4/4 eager tokens and 80/194 reuse while observing ERS `0.41665989463124997`, median TTFT `68.590 ms`, and median TPOT `6.900 ms`; versus SIMD-pack Q8, those values change by `+0.014505`, `-4.542 ms`, and `-0.408 ms`. The separate long matrix remains 6/6 exact but observes TTFT/TPOT increases at both lengths |
| Direct paged-Q8 decode attention | implemented, compiled, and model-executed | Q8 specialization replaces six materialized attention paths and twelve past-K/V inputs with six typed paged-attention commands, one physical token pool, and one slot map. The bounded 350M trace preserves 4/4 eager tokens and 80/194 reuse while observing ERS `0.38326789681891504`, median TTFT `72.550 ms`, and median TPOT `8.035 ms`; the separate 2K/4K needle matrix remains 6/6 retrieval/parity while median TPOT changes by `-10.083/-22.551 ms` versus vector unpack. The selectable FP16 path remains materialized |
| Vector-staged Q8 prefill | implemented, compiled, and model-executed | the 16 by 16 output tile stages 64 reduction elements as activation4 and dequantized-weight4 vectors. Emitted barriers per k=1024/4608 tile change from 128/576 to 32/144; one bounded 350M run preserves 4/4 scored eager-Q8 sequences and 80/194 reuse while observing ERS `0.3377415731686302`, median TTFT `93.156 ms`, and median TPOT `7.948 ms` |
| Serving-only last-token vocabulary projection | implemented and model-executed | the typed LFM prefill specializer recognizes the sole-consumer identity-index plus LM-head chain and projects `[1,1,65536]` instead of `[1,tokens,65536]`. One bounded 350M run preserves exact tokens and 80/194 reuse while observing ERS `0.3588470515801844`, median TTFT `79.157 ms`, and median TPOT `8.300 ms`; the preceding native medians were `93.156/7.948 ms` |
| Q8-default vocabulary projection | implemented, captured, and model-executed | all 93 linear modules including `lm_head` use per-output-channel Q8; the token embedding remains FP16. The real final projection specializes to one row and selects packed SIMD GEMV. One bounded native run preserves 4/4 full-Q8 eager tokens and 80/194 reuse while observing ERS `0.3908962321067631`, median TTFT `71.477 ms`, and median TPOT `7.311 ms` |
| SIMD-group RMSNorm | implemented and model-executed in aggregate | one 32-lane SIMD group reduces and writes each row, with eight rows per 256-thread group for both dtype variants. The 45 commands in each 350M stage execute in the combined 4/4 exact scored run; scalar fallback remains available, and the aggregate measurement does not isolate RMSNorm |
| Single-pass SIMD attention | implemented and model-executed in aggregate | one 32-lane SIMD group computes each width-64 query row, reducing each query-key score once and accumulating softmax-weighted values online. All six commands per stage select it; the combined optimized run retains exact scored tokens, and scalar fallback remains declared |
| generated Q8 Metal runtime loading and dispatch | implemented; exact model path verified; native numerical parity remains open | Ninja builds the PyTorch MPS C++ bridge, links the generated `.metallib`, and the Python FX backend selects generated exact dequantization or Phase 2 native Q8 entry points. The combined 350M differential probe records 92 exact-mode generated dispatches with `max_abs=0`, `mean_abs=0`, and 92 native Phase 2 dispatches with `max_abs=0.078125`, `mean_abs=0.00713115930557251`; no ERS result was written |
| OCaml serving radix/KV cache | dependent cached-suffix batching implemented and measured | the corrected runtime reserves one checkpoint per suffix token and completes 4/4 scored requests with 42/61 and 38/59 second-turn reuse, 80/194 total cache reuse, and exact eager-Q8 output sequences |
| Selectable FP16 KV/recurrent storage | implemented and model-executed; Q8 remains default | the paired full-Q8 package completes 4/4 warmup and scored requests with `--kv fp16`, exact eager and Q8-cache token IDs, and unchanged 80/194 radix reuse. The separate FP16 observation records ERS `0.4297032150753201` and median TTFT/TPOT `69.163/6.698 ms` |
| Versioned generated package ABI | implemented for captured ABI-v12 full-Q8 templates | Package ABI v12 retains ABI-v2 through ABI-v11 reads and typed fused Rms_rope dispatch. The current replan writes 702-command/74-entry prefill and 756-command/72-entry decode templates with zero opaque operations and 243 validated bindings each; direct paged-Q8 attention is available to runtime-specialized decode (696 commands) |
| OCaml tensor-store ownership | shared real JSON-free full-Q8 archive validated | Dynamo streams static inputs into a versioned binary index plus 256-byte-aligned payloads. The full-Q8 capture seals one 489,377,152-byte `weights.llmopt`, canonicalizes aliases, and hard-links it across prefill/decode; its 243 tensors include the FP16 embedding plus separate Q8 head weight/scale |
| OCaml Metal serving loader and dispatch | direct paged-Q8 attention plus packed fused/SIMD schedules and dependent suffix batches model-executed | ABI-v12 packages prefer fused RMSNorm-RoPE, vector-staged Q8 prefill, paired packed SIMD-group Q8 decode including the head, and direct radix-slot attention with materialized FP16 fallback. The bounded model run preserves exact full-Q8 eager tokens and 80/194 reuse, measuring native ERS `0.4122601696838274` |
| Complete 350M operation schedule | full-Q8 captured templates specialize and execute | The 702/756-command ABI-v12 pair has zero opaque commands and 243 bindings. Typed specialization reduces specialized decode to 696 commands with 6 direct paged attentions |
| native tokenization, LFM chat, generation, and HTTP/SSE | implemented for serial smoke requests | `LLMOPTTK` and typed chat feed one persistent `llmopt-serve` engine. The current 4/4 warmup and scored trace has pinned outputs, exact eager parity, ERS `0.4122601696838274`, and 80/194 cached prompt tokens |
| natural needle-in-a-haystack validation | current fused RMSNorm-RoPE native retrieval and token parity measured | the direct paged-Q8 2,048/4,096-token matrix retrieves `RAVEN-4271` in 6/6 and matches all 12 established eager-Q8 IDs. Exact-only formatting is 0/6 because fixed output is `RAVEN-4271Lottery`; median TTFT/TPOT is 1,241.091/25.417 ms and 3,036.156/39.670 ms |

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
