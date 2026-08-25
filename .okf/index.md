---
okf_version: '0.2'
---

# llmopt: effect-driven tile compiler research

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

## Experiments and benchmarks

* [FX linear smoke](experiments/exp-0001-fx-linear.md) - first cross-language probe.
* [Benchmark protocol](benchmarks/benchmark-protocol.md) - eager PyTorch MPS comparison record format.
* [MPS ERS benchsuite](benchmarks/bench-suite.md) - trace scoring, warmup artifacts, and needle validation.
* [Semantic 5x3 result](experiments/exp-0004-lfm25-semantic-5x3.md) - isolated long-context comparison and raw latency evidence.
* [Viettel AI Race benchsuite implementation](experiments/exp-0005-viettel-racebench-implementation.md) - runner adoption and current baseline status.
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
