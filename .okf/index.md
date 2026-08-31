---
okf_version: '0.2'
---

# llmopt: effect-driven tile compiler research

The product boundary is a model-neutral PyTorch/Dynamo capture and a declared
`model.llmopt` execution contract. The current Metal backend uses W4A16 weights
plus grouped-Q8 KV/recurrent state; those formats are backend capabilities, not
a built-in model profile or the target weight policy. GGUF with Unsloth Dynamic
mixed quantization is the target weight-distribution path; FX capture, not a
GGUF architecture identifier, defines executable topology. Q8-weight and FP16-KV material below is historical
evidence, not an available compiler, package, runtime, CLI, or benchmark
configuration.

## Orientation

* [Architecture](architecture.md) - current Dynamo/FX to OCaml planning pipeline.
* [Complete OCaml serving goal](goal-serving-runtime.md) - requirement-by-requirement end state and evidence map.
* [Probe models](probe-models.md) - model coverage without product defaults.
* [LFM2.5 probe](target-lfm25.md) - hybrid probe facts and probe-only constants.
* [Research tracking](tracking.md) - open questions, slices, and evidence status.

## Decisions

* [FX backend boundary](decisions/fx-backend.md) - why PyTorch owns graph capture.
* [Effect planning](decisions/effect-planning.md) - why OCaml effects own staging.
* [Metal backend boundary](decisions/backend-boundary.md) - MSL execution and LLVM inspection.
* [LFM2.5-350M ESR bandwidth target](decisions/target-lfm25-350m-bandwidth.md) - memory bandwidth physics and ERS feasibility on Apple Silicon.
* [llama.cpp performance target](decisions/llama-cpp-target.md) - llama.cpp Q4_0 as the W4 parity comparison, with MPS retained as reference.
* [Model Program boundary](decisions/model-program-boundary.md) - root execution contract joining PyTorch capture, compiled entrypoints, state, and serving assets.
* [SRPT and Queueing-Theoretic Serving Scheduler](decisions/srpt-queueing-serving-scheduler.md) - continuous batching, age-weighted SRPT queue optimization, admission watermarks, and decode-first scheduling.
* [XGBoost / GBDT Kernel Cost Model](decisions/xgboost-kernel-cost-model.md) - learned GBDT cost model trained on profiling sweeps and transpiled to pure OCaml AST.
* [Hardware-Aware AOT Compilation & Microarchitectural Discovery](decisions/hardware-aware-aot-compilation-and-discovery.md) - discovery of SIMD width, SRAM banks, bank conflicts, and analytical prefill roofline modeling.
* [DAG Concurrency Analysis & Co-Scheduling](decisions/dag-co-scheduling-optimizer-pass.md) - SSA dependency DAG analysis, ready antichain concurrency, and barrier placement.
* [Fewest-Hops Whole-Block Megakernel Compiler](decisions/fewest-hops-megakernel-compiler.md) - fusing multi-layer blocks into ~40 hops with zero DRAM activation roundtrips.
* [Macro-Operator Fusions](decisions/macro-operator-fusions.md) - high-impact fusions for SwiGLU, RMSNorm-RoPE, QKV, ShortConv, and on-GPU Argmax.
* [AOT Decode Solidification & Zero-JIT Serving](decisions/aot-decode-solidification-zero-jit-serving.md) - zero-overhead serving with prebaked Indirect Command Buffers (ICB).
* [GGUF and Unsloth Dynamic (UD) Quantization Support](decisions/gguf-unsloth-dynamic-quantization.md) - GGUF ingestion, dequantization, and AOT code specialization.
* [Hardware Bitfield Extraction, Speculative Multi-Token Verification, and Queue-Coordinated Pipelining](decisions/speculative-pipelining-hardware-acceleration.md) - single-cycle bitfield extraction, Split-K reductions, and speculative draft verification pipelined through SRPT continuous batching.

## Prior art

* [TensorRT-LLM](prior-art/tensorrt-llm.md) - the same AOT-engine bet, industrialized: build-time measured autotuning, AOT shape profiles, and hand-written per-architecture hot kernels.
* [Modular MAX and Mojo](prior-art/modular-max-mojo.md) - the same portability goal solved in the language: parametric kernel bodies specialized at compile time across CUDA, ROCm, and Metal.

## Experiments and benchmarks

* [FX linear smoke](experiments/exp-0001-fx-linear.md) - first cross-language probe.
* [Cross-runtime benchmark protocol](benchmarks/benchmark-protocol.md) - llama.cpp target, native llmopt side comparison, and MPS reference record format.
* [llama.cpp comparison benchmark](benchmarks/llama-cpp.md) - native llama-bench, llama-server ERS, and optional llmopt side replay.
* [MPS ERS benchsuite](benchmarks/bench-suite.md) - retained PyTorch reference trace scoring, warmup artifacts, and needle validation.
* [Semantic 5x3 result](experiments/exp-0004-lfm25-semantic-5x3.md) - isolated long-context comparison and raw latency evidence.
* [Superseded SmolLM2 cross-model claim](experiments/exp-0096-cross-model-smollm2-validation.md) - retracts unsupported full-model and performance claims and points to reproducible replacement evidence.
* [Architecture-neutral GGUF FX native Linear parity](experiments/exp-0099-gguf-fx-native-linear-parity-2026-08-29.md) - bounded SmolLM, Qwen, and Gemma GGUF-backed Linear execution plus full-capture inventories.
* [Current LFM Model Program comparison with llama.cpp](experiments/exp-0100-model-program-v2-llama-cpp-comparison-2026-08-29.md) - four paired same-text runs for the current ABI-v2 LFM serving engine and llama.cpp Q4_0.
* [Captured full-model GGUF comparison](experiments/exp-0101-gguf-full-model-comparison-2026-08-29.md) - zero-opaque native SmolLM, Qwen, and Gemma forwards, Qwen gated-delta numerical evidence, Gemma RMSNorm fusion, and llama.cpp timing context.
* [Captured-shape SIMD attention specialization](experiments/exp-0102-captured-attention-specialization-2026-08-29.md) - FX-shape-selected h256/h512 Metal kernels and controlled Gemma/Qwen full-forward timing.
* [Quantization-neutral Linear region discovery](experiments/exp-0103-semantic-linear-regions-2026-08-29.md) - semantic Linear fusion regions across mixed GGUF layouts plus fresh Qwen/Gemma comparison timing.
* [Tensor-layout quantization and Linear storage classification](experiments/exp-0104-tensor-layout-capabilities-2026-08-29.md) - shared quant layout geometry and Linear storage facts for compiler passes and later tactics.
* [Typed Scan regions and Qwen recurrence recovery](experiments/exp-0105-typed-scan-recovery-2026-08-29.md) - typed carried state plus structural recovery of the 18 unrolled Qwen recurrences.
* [Registered Metal tactics and paired-row Q4_K Linear](experiments/exp-0106-registered-metal-tactics-2026-08-29.md) - shape/layout/dtype/target-selected Metal implementations plus fresh Qwen/Gemma comparison timing.
* [Captured triangular recurrence fusion](experiments/exp-0107-triangular-recurrence-fusion-2026-08-30.md) - graph-topology-selected recurrence lowering, semantic kernel ABI repair, typed S-expression dumps, and fresh Qwen/Gemma comparison timing.
* [Paired-row mixed-quant Linear tactics](experiments/exp-0108-paired-row-mixed-quant-linear-2026-08-30.md) - paired weight-decode reuse across every supported GGUF quant layout with fresh Qwen/Gemma comparison timing.
* [SIMD-group batched matmul tactic](experiments/exp-0109-simdgroup-batched-matmul-2026-08-30.md) - broadcast-aware SIMD matrix tiles selected from captured batched-matmul geometry.
* [Sub-SIMD short-row K-quant Linear tactics](experiments/exp-0110-subsimd-short-row-kquant-linear-2026-08-30.md) - four-column SIMD execution for captured two-row Q4_K/Q5_K/Q6_K Linear.
* [Graph-recovered zero-state gated-delta execution](experiments/exp-0111-graph-recovered-gated-delta-2026-08-30.md) - recover a semantic recurrence from captured topology and lower it directly to SIMD Metal.
* [Full-SIMD two-column Q4_K Linear tactic](experiments/exp-0112-full-simd-q4-k-linear-2026-08-30.md) - use full SIMD lanes for captured two-row Q4_K Linear without architecture identifiers.
* [Occupancy-selected full-SIMD Q5_K Linear tactic](experiments/exp-0113-occupancy-selected-q5-k-linear-2026-08-30.md) - select full-lane Q5_K execution from captured superblock occupancy.
* [Topology-fused L2 normalization](experiments/exp-0114-topology-fused-l2-normalization-2026-08-30.md) - replace captured final-axis normalization chains with one semantic SIMD dispatch.
* [Attention-linked token-major RMSNorm-RoPE fusion](experiments/exp-0115-attention-linked-rms-rope-2026-08-30.md) - pair captured rotary Q/K branches through their shared semantic Attention.
* [Single-consumer activation-product fusion](experiments/exp-0116-activation-product-fusion-2026-08-30.md) - combine captured SiLU, GELU, and sigmoid products into one typed pointwise dispatch.
* [Single-consumer RMSNorm-residual fusion](experiments/exp-0117-rmsnorm-residual-fusion-2026-08-30.md) - combine captured RMSNorm and residual add into one SIMD dispatch.
* [Shape-selected wide-row RMSNorm](experiments/exp-0118-wide-row-rmsnorm-2026-08-30.md) - assign a whole threadgroup to low-row, wide captured tensors without architecture IDs.
* [Topology-fused Q4_K gated Linear](experiments/exp-0119-q4-k-gated-linear-2026-08-30.md) - fuse same-layout gate/up branches and their sole activation-product consumer.
* [Gated Linear layout extension](experiments/exp-0120-gated-linear-layout-extension-2026-08-30.md) - extend the same semantic fusion through Q5_K and IQ4_XS tactics.
* [Token-major Gated Delta input absorption](experiments/exp-0121-token-major-gated-delta-input-absorption-2026-08-30.md) - remove exclusive transpose/contiguous/cast chains around the semantic recurrence.
* [Q4_K two-token two-output Linear tactic](experiments/exp-0122-q4-k-two-token-two-output-tactic-2026-08-30.md) - reuse two quantized output columns across both captured token rows.
* [Float32-weight two-token two-output Linear tactic](experiments/exp-0123-f32-weight-two-token-two-output-tactic-2026-08-30.md) - reuse both captured token rows across paired dense output columns.
* [Q5_K paired-token weight-reuse tactic](experiments/exp-0124-q5-k-paired-token-weight-reuse-2026-08-30.md) - load each Q5_K block once while recording the corrected compiler/runtime site inventory.
* [Shared shape-aware Linear tactic registry](experiments/exp-0125-shared-shape-aware-linear-tactic-registry-2026-08-30.md) - bind each captured Linear command through the same typed tactic policy used by compiler emission.
* [Graph-relational GQA and attention layout fusion](experiments/exp-0126-graph-relational-attention-layout-fusion-2026-08-30.md) - eliminate grouped-head expansion and token-first movement from captured topology and dimension relations.
* [Prepared steady-state package execution](experiments/exp-0127-prepared-steady-state-package-execution-2026-08-30.md) - reuse typed memory preparation and exclude validation readback from synchronized execution timing.
* [Graph-relational token-major ShortConv-SiLU fusion](experiments/exp-0128-graph-relational-short-conv-silu-fusion-2026-08-30.md) - fuse captured depthwise convolution trim, activation, and layout movement from topology and shape relations.
* [Packed last-axis L2 normalization](experiments/exp-0129-packed-last-axis-l2-normalization-2026-08-30.md) - normalize captured contiguous packed slices directly without materialized indexing.
* [Attention-linked token-major rotary QK fusion](experiments/exp-0130-attention-linked-token-major-rotary-qk-2026-08-30.md) - fuse paired captured Q/K half-rotation topology into one layout-aware semantic kernel.
* [Q5_0 gated Linear fusion](experiments/exp-0131-q5-0-gated-linear-2026-08-30.md) - extend topology-selected gated Linear execution to captured Q5_0 weights.
* [Token-major Attention value absorption](experiments/exp-0132-token-major-attention-value-2026-08-30.md) - let captured Attention consume exclusive token-major value tensors directly.
* [Topology-fused quantized Linear residuals](experiments/exp-0133-quantized-linear-residual-2026-08-30.md) - fuse exclusive block-quantized Linear plus residual topology through shared tactics.
* [Hardware-derived prefill cost model](experiments/exp-0097-hardware-derived-prefill-cost-model.md) - analytical roofline knee, core saturation limits, SLA-bounded chunk budget, and serving queue scheduler integration.
* [Zero-overhead parameterized sampling engine](experiments/exp-0098-parameterized-sampling-engine.md) - zero-allocation streaming min-heap ARM NEON sampler with temperature, top-k, top-p, min-p, and seeded PRNG.
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
