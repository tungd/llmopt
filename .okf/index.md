---
okf_version: '0.2'
---

# llmopt: effect-driven tile compiler research

The executable project contract is W4A16 weights plus grouped-Q8 KV/recurrent
state. Q8-weight and FP16-KV material below is historical evidence, not an
available compiler, package, runtime, CLI, or benchmark configuration.

## Orientation

* [Architecture](architecture.md) - current Dynamo/FX to OCaml planning pipeline.
* [Complete OCaml serving goal](goal-serving-runtime.md) - requirement-by-requirement end state and evidence map.
* [Target model](target-lfm25.md) - LFM2.5-350M facts and target-specific constants.
* [Research tracking](tracking.md) - open questions, slices, and evidence status.

## Decisions

* [FX backend boundary](decisions/fx-backend.md) - why PyTorch owns graph capture.
* [Effect planning](decisions/effect-planning.md) - why OCaml effects own staging.
* [Metal backend boundary](decisions/backend-boundary.md) - MSL execution and LLVM inspection.
* [LFM2.5-350M ESR bandwidth target](decisions/target-lfm25-350m-bandwidth.md) - memory bandwidth physics and ERS feasibility on Apple Silicon.
* [llama.cpp performance target](decisions/llama-cpp-target.md) - llama.cpp Q4_0 as the W4 parity comparison, with MPS retained as reference.

## Experiments and benchmarks

* [FX linear smoke](experiments/exp-0001-fx-linear.md) - first cross-language probe.
* [Cross-runtime benchmark protocol](benchmarks/benchmark-protocol.md) - llama.cpp target, native llmopt side comparison, and MPS reference record format.
* [llama.cpp comparison benchmark](benchmarks/llama-cpp.md) - native llama-bench, llama-server ERS, and optional llmopt side replay.
* [MPS ERS benchsuite](benchmarks/bench-suite.md) - retained PyTorch reference trace scoring, warmup artifacts, and needle validation.
* [Semantic 5x3 result](experiments/exp-0004-lfm25-semantic-5x3.md) - isolated long-context comparison and raw latency evidence.
* [Viettel AI Race benchsuite implementation](experiments/exp-0005-viettel-racebench-implementation.md) - runner adoption and current baseline status.
* [Executable W4A16 SwiGLU rule](experiments/exp-0092-w4a16-swiglu-rule-2026-08-27.md) - deletion audit, rule-engine rewrite, parallel Metal lowering, package inventory, and Q4 comparison.
* [Restored W4 SIMD corrective rerun](experiments/exp-0093-restored-w4-simd-rerun-2026-08-27.md) - restored kernel execution strategy, latest Q4 comparison, and postmortem of the destructive canonicalization.
* [W4A16 RMSNorm/SwiGLU cast absorption](experiments/exp-0094-w4a16-rms-cast-absorption-2026-08-27.md) - rule-engine removal of the 33 single-use f16-to-f32 widening casts from each preserved graph and the resulting static package inventory.
* [W4A16 decode RoPE table elision](experiments/exp-0095-w4a16-rope-table-elision-2026-08-27.md) - CPU-precomputed position rows, direct runtime binding, scalar-branch pruning, and a fresh native comparison.
* [LFM2.5-350M baseline](experiments/exp-0006-lfm25-350m-probe.md) - memory-safe engine-pass and baseline observation.
* [Q8 weight-only linear pass](experiments/exp-0007-q8-linear-pass.md) - first quantized optimizer/codegen slice with Metal and LLVM smoke validation.
* [Generated Q8 Metal runtime](experiments/exp-0008-metal-runtime-q8.md) - metallib loading, MPS dispatch bridge, tiled launch correction, and the captured non-model probe.
* [OCaml radix and KV serving cache](experiments/exp-0009-ocaml-radix-kv-cache.md) - mandatory prefix caching, hybrid checkpoints, owned allocation, and selectable FP16/Q8 layout.
* [Pre-integration ERS and needle correction](experiments/exp-0010-preintegration-ers.md) - zero-cache baseline and corrected retrieval evidence.
* [Compiled-graph serving package](experiments/exp-0011-compiled-graph-package.md) - versioned OCaml package ABI and Ninja artifact validation.
* [Native OCaml Metal dispatch](experiments/exp-0012-native-ocaml-metal.md) - direct package loading, shared buffers, and Q8 command submission without PyTorch.
* [Safetensors mapped Metal dispatch](experiments/exp-0013-safetensors-metal-mapping.md) - one binary tensor archive parsed, mapped, and bound directly from OCaml.
* [LFM2.5-350M Q8 tensor export](experiments/exp-0014-lfm25-350m-tensor-export.md) - all captured model state bound into one validated serving archive.
* [Binary serving schedule](experiments/exp-0015-binary-serving-schedule.md) - rank-aware typed commands loaded without FX JSON or a textual plan.
* [Rank-aware primitives and fused RMSNorm](experiments/exp-0016-rank-primitives-rmsnorm.md) - typed N-D FX lowering and the first fused LFM normalization kernel.
* [Static indexing, chunk elimination, and typed concat](experiments/exp-0017-index-chunk-concat.md) - compact typed movement commands for the next measured LFM operator family.
* [LFM2.5-350M Q8 manifest-v2 recapture](experiments/exp-0018-lfm25-v2-recapture.md) - real-model operator inventory and exact direct-FX parity after the rank-aware lowering slices.
* [Qualified FX target matching](experiments/exp-0019-qualified-fx-targets.md) - qualification-aware operator matching with a measured 14-command recovery.
* [Typed LFM ShortConv lowering and Metal kernel](experiments/exp-0020-lfm-short-conv.md) - ten real conv1d nodes moved into a typed, compiled operation.
* [Typed LFM masked attention and Metal kernel](experiments/exp-0021-lfm-attention.md) - captured prefill GQA boundary moved into typed IR and compiled fused MSL.
* [Typed LFM token embedding and Metal gather](experiments/exp-0022-lfm-embedding.md) - int64 token lookup moved into typed IR and a compiled float16 gather kernel.
* [JSON-free binary weight archive](experiments/exp-0023-binary-weight-archive.md) - package ABI v2 and a typed, aligned, mmap-ready weight archive without a JSON index.
* [Typed LFM position and mask construction](experiments/exp-0024-lfm-mask-position.md) - zero-opaque saved prefill through schedule-v7 primitives and compiled Metal kernels.
* [Shared prefill/decode capture and recurrent state](experiments/exp-0025-lfm-prefill-decode.md) - one binary weight archive, typed schedule-v8 cache operations, and zero-opaque preserved model packages.
* [Native binary schedule execution](experiments/exp-0026-native-schedule-execution.md) - OCaml resolves package values and dispatches Q8 from the command stream against the binary archive.
* [Native built-in kernel dispatch](experiments/exp-0027-native-builtin-dispatch.md) - one typed package executes every currently emitted Metal family with 12 exact outputs.
* [Native typed cast dispatch](experiments/exp-0028-native-cast-dispatch.md) - package ABI v3 adds and executes the three cast directions used by LFM2.5.
* [Native LFM pointwise dispatch](experiments/exp-0029-native-pointwise-dispatch.md) - package ABI v4 executes the nine pointwise forms required by the preserved plans.
* [Binary Dynamo/FX compiler transport](experiments/exp-0030-binary-fx-transport.md) - default capture and offline replanning now cross a versioned binary graph boundary.
* [Native LFM movement dispatch](experiments/exp-0031-native-movement-dispatch.md) - package ABI v5 materializes every movement form observed in the preserved plans.
* [Native recurrent compute dispatch](experiments/exp-0032-native-recurrent-dispatch.md) - package ABI v6 executes decode sum and functional cache update.
* [Native float16 vocabulary projection](experiments/exp-0033-native-f16-linear.md) - both fixed model schedules now declare every observed kernel family.
* [Liveness-planned Metal workspace](experiments/exp-0034-liveness-workspace.md) - one retained workspace replaces per-intermediate Metal buffers using typed schedule liveness.
* [Physical Q8 and FP16 Metal KV cache](experiments/exp-0035-physical-kv-cache.md) - native Q8/FP16 token and checkpoint pools with exact Metal pack/unpack evidence.
* [Native LFM serving engine](experiments/exp-0036-native-serving-engine.md) - complete fixed Q8 prefill/decode schedules coordinated with physical radix-owned state.
* [Binary LFM tokenizer and typed chat encoding](experiments/exp-0037-native-tokenizer-chat.md) - versioned tokenizer archive, native byte-level BPE, and typed LFM chat parity.
* [Variable-length LFM schedules and repeated native decode](experiments/exp-0038-variable-schedule-decode.md) - request-length specialization, growing radix/KV state, and four-token eager-Q8 parity.
* [Native OCaml chat generation](experiments/exp-0039-native-chat-generation.md) - real chat text crosses native tokenization, dynamic Metal execution, and greedy decoding with eager-Q8 parity.
* [Native OCaml HTTP serving and token-level ERS](experiments/exp-0040-native-http-serving.md) - the persistent native endpoint records real cached-prefix usage, corrected token timings, ERS, and exact eager-Q8 output IDs.
* [Native long-context needle retrieval with EOS boundary](experiments/exp-0041-native-long-needle.md) - six native 2,048/4,096-token prompts retrieve exactly, with normal-EOS versus pinned-output semantics recorded separately.
* [Batched native Metal schedule submission](experiments/exp-0042-batched-metal-command.md) - schedule-wide compute/blit encoding preserves parity while reducing TTFT and TPOT.
* [Q8 decode-specialized Metal GEMV](experiments/exp-0043-q8-decode-gemv.md) - one-row vectorized Q8 execution preserves parity and records its matched TTFT, TPOT, and ERS deltas.
* [Native cache-submission batching](experiments/exp-0044-cache-submission-batching.md) - physical KV and recurrent cache phases use ordered submissions with exact byte and token evidence.
* [Cached-suffix command-buffer batching attempt](experiments/exp-0045-cached-suffix-batching-attempt.md) - one dependent replay batch preserves per-token checkpoint structure; the first model attempt failed and the correction remains unmeasured.
* [Q8 linear-SiLU epilogue fusion](experiments/exp-0046-q8-linear-silu-fusion.md) - the first real-model epilogue pass removes sixteen workspace round trips and dispatches from each stage.
* [Q8 linear-residual epilogue fusion](experiments/exp-0047-q8-linear-add-fusion.md) - the second epilogue pass removes 32 residual workspace round trips and dispatches from each stage.
* [Q8 multiplied-input down-projection fusion](experiments/exp-0048-q8-multiplied-input-fusion.md) - each feed-forward down projection absorbs its sole-consumer SwiGLU product.
* [SIMD-group Q8 decode GEMV](experiments/exp-0049-simdgroup-q8-gemv.md) - 32 lanes cooperate on each decode output channel with legacy scalar fallback.
* [SIMD-group RMSNorm](experiments/exp-0050-simdgroup-rmsnorm.md) - one SIMD group reduces and writes each normalization row.
* [Single-pass SIMD attention](experiments/exp-0051-online-softmax-attention.md) - each query-key score is computed once with online softmax accumulation.
* [Optimized native stack measurement](experiments/exp-0052-optimized-native-stack.md) - corrected suffix replay and the fused/SIMD stack measured on 350M.
* [Effective feed-forward shape](experiments/exp-0053-effective-ff-shape.md) - executable SwiGLU projections use the auto-adjusted width 4,608.
* [Packed SIMD Q8 GEMV](experiments/exp-0054-packed-simd-q8-gemv.md) - vectorized activation and int8 weight loads across all decode epilogues.
* [Packed SIMD Q8 measurement](experiments/exp-0055-packed-simd-q8-gemv-measurement.md) - bounded 350M token, radix, latency, and ERS evidence.
* [Vector-staged Q8 prefill](experiments/exp-0056-vector-staged-q8-prefill.md) - 64-wide vector staging inside the tiled Q8 prefill kernel.
* [Vector-prefill measurement](experiments/exp-0057-vector-prefill-measurement.md) - bounded 350M evidence for vector-staged prefill.
* [Native FP16 logit export](experiments/exp-0058-native-logit-export.md) - raw binary vocabulary-row comparison with eager Q8.
* [Fixed-output native needle](experiments/exp-0059-fixed12-native-needle.md) - 2,048/4,096-token retrieval and complete eager-token parity.
* [Last-token vocabulary projection](experiments/exp-0060-last-token-vocab-projection.md) - serving prefill projects only the final vocabulary row.
* [Last-token projection measurement](experiments/exp-0061-last-token-projection-measurement.md) - bounded 350M latency, parity, radix, and ERS evidence.
* [Q8-default LM head](experiments/exp-0062-q8-lm-head-compiler.md) - all linear modules, including the vocabulary projection, default to Q8.
* [Full-Q8 LM-head measurement](experiments/exp-0063-q8-lm-head-measurement.md) - recaptured 350M native trace and long-context evidence.
* [Paired SIMD-group Q8 GEMV](experiments/exp-0064-paired-simd-q8-gemv.md) - adjacent output channels share each packed activation load.
* [Paired SIMD measurement](experiments/exp-0065-paired-simd-measurement.md) - bounded 350M trace and needle matrix for paired decode.
* [Selectable FP16 KV execution](experiments/exp-0066-fp16-kv-model-execution.md) - model-scale FP16 cache execution while Q8 remains default.
* [SIMD-group Q8 cache packing](experiments/exp-0067-simd-q8-cache-pack.md) - attention and recurrent quantization use one SIMD group per Q8 group.
* [SIMD Q8 cache-pack measurement](experiments/exp-0068-simd-q8-cache-pack-measurement.md) - bounded 350M token, radix, latency, and ERS evidence.
* [Vectorized Q8 cache unpack](experiments/exp-0069-vector-q8-cache-unpack.md) - vec4 attention and recurrent restoration with scalar fallback.
* [Vector Q8 cache-unpack measurement](experiments/exp-0070-vector-q8-cache-unpack-measurement.md) - bounded 350M trace and long-context evidence.
* [Direct paged-Q8 decode attention](experiments/exp-0071-paged-q8-attention.md) - generated Metal reads radix-owned Q8 K/V slots without materializing past tensors.
* [Direct paged-Q8 attention measurement](experiments/exp-0072-paged-q8-attention-measurement.md) - bounded 350M short and long-context evidence with exact tokens and mixed latency deltas.
* [Fused RMSNorm and RoPE](experiments/exp-0073-rmsnorm-rope-fusion.md) - twelve 10-command query/key chains across 6 attention layers fuse into single SIMD Metal kernels.
* [Fused RMSNorm-RoPE measurement](experiments/exp-0074-rmsnorm-rope-measurement.md) - bounded 350M trace and needle matrix for the fused RMSNorm-RoPE package.
