# Experiments

* [FX linear smoke](exp-0001-fx-linear.md) - first Python-to-OCaml code-generation probe.
* [LFM2.5 naive MPS execution](exp-0002-lfm25-mps.md) - complete model forward against eager MPS.
* [LFM2.5 MPS ERS benchsuite](exp-0003-lfm25-benchsuite.md) - first scored trace and needle probe.
* [LFM2.5 isolated semantic 5x3 comparison](exp-0004-lfm25-semantic-5x3.md) - long-context raw latency, exact token parity, and ERS dynamic-range observation.
* [Viettel AI Race benchsuite implementation](exp-0005-viettel-racebench-implementation.md) - adopted runner contracts, full trace shape, target adaptation, and the interrupted memory-safe probe.
* [LFM2.5-350M smaller-model probe](exp-0006-lfm25-350m-probe.md) - separate Ninja target for obtaining a memory-safe engine-pass and baseline observation.
* [Q8 weight-only linear lowering](exp-0007-q8-linear-pass.md) - first quantized optimizer and source-emission slice.
* [Generated Q8 Metal runtime](exp-0008-metal-runtime-q8.md) - metallib loading, MPS dispatch, tiled launch correction, and a bounded non-model probe.
* [OCaml radix and KV serving cache](exp-0009-ocaml-radix-kv-cache.md) - SGLang-derived compressed radix semantics with hybrid checkpoints and configurable KV storage.
* [Pre-integration ERS and needle correction](exp-0010-preintegration-ers.md) - isolated Q8 result before cache integration and corrected retrieval grading.
* [Compiled-graph serving package](exp-0011-compiled-graph-package.md) - versioned package ABI, typed kernel entries, and Ninja validation of referenced artifacts.
* [Native OCaml Metal dispatch](exp-0012-native-ocaml-metal.md) - standalone OCaml package loading and exact Q8 fixture execution.
* [Safetensors mapped Metal dispatch](exp-0013-safetensors-metal-mapping.md) - one binary tensor archive mapped and bound directly by the OCaml runtime.
* [LFM2.5-350M Q8 tensor export](exp-0014-lfm25-350m-tensor-export.md) - complete captured static state in one validated serving archive.
* [Binary serving schedule](exp-0015-binary-serving-schedule.md) - versioned rank-aware command stream and JSON-free native startup.
* [Rank-aware primitives and fused RMSNorm](exp-0016-rank-primitives-rmsnorm.md) - Dynamo metadata correction, typed N-D commands, RMSNorm fusion, and Metal compilation.
* [Static indexing, chunk elimination, and typed concat](exp-0017-index-chunk-concat.md) - normalized tensor indexing, compile-time chunk/getitem fusion, and binary concat commands.
* [LFM2.5-350M Q8 manifest-v2 recapture](exp-0018-lfm25-v2-recapture.md) - one memory-bounded real capture with 835 commands, 42 opaque commands, and exact direct-FX logits.
* [Qualified FX target matching](exp-0019-qualified-fx-targets.md) - fixes the expand/logical-and suffix collision and lowers the saved manifest to 28 opaque commands.
* [Typed LFM ShortConv lowering and Metal kernel](exp-0020-lfm-short-conv.md) - exact depthwise prefill convolution lowering, schedule-v4 encoding, CPU reference, and compiled MSL.
* [Typed LFM masked attention and Metal kernel](exp-0021-lfm-attention.md) - six SDPA nodes lowered through schedule v5 with CPU and Xcode Metal evidence.
* [Typed LFM token embedding and Metal gather](exp-0022-lfm-embedding.md) - model lookup moved into schedule v6 with exact CPU and Xcode Metal evidence.
* [JSON-free binary weight archive](exp-0023-binary-weight-archive.md) - replaces the safetensors serving boundary with a versioned binary index and aligned payloads.
* [Typed LFM position and mask construction](exp-0024-lfm-mask-position.md) - schedule-v7 lowering and compiled MSL remove the final opaque commands from saved no-cache prefill.
* [Shared prefill/decode capture and recurrent state](exp-0025-lfm-prefill-decode.md) - one shared binary archive and schedule-v8 lowering produce zero-opaque prefill/decode packages.
* [Native binary schedule execution](exp-0026-native-schedule-execution.md) - automatic OCaml input binding, Q8 dispatch, and output resolution against `weights.llmopt`.
* [Native built-in kernel dispatch](exp-0027-native-builtin-dispatch.md) - generic typed dispatch and one 12-output Metal execution probe.
* [Native typed cast dispatch](exp-0028-native-cast-dispatch.md) - ABI-v3 cast kernels and one exact 15-output execution probe.
* [Native LFM pointwise dispatch](exp-0029-native-pointwise-dispatch.md) - ABI-v4 broadcast/scalar kernels and one exact 24-output execution probe.
* [Binary Dynamo/FX compiler transport](exp-0030-binary-fx-transport.md) - `LLMOPTFX` replaces JSON on the default Python-to-OCaml compile path.
* [Native LFM movement dispatch](exp-0031-native-movement-dispatch.md) - ABI-v5 movement kernels and one exact 36-output execution probe.
* [Native recurrent compute dispatch](exp-0032-native-recurrent-dispatch.md) - ABI-v6 sum and slice-update kernels with one exact 38-output probe.
* [Native float16 vocabulary projection](exp-0033-native-f16-linear.md) - SIMD-reduced final projection with one exact 39-output probe and complete fixed-graph metallibs.
* [Liveness-planned Metal workspace](exp-0034-liveness-workspace.md) - alias-aware interval allocation reduces the offline prefill/decode workspace while preserving one exact 39-output device probe.
* [Physical Q8 and FP16 Metal KV cache](exp-0035-physical-kv-cache.md) - package-declared native pools and cache kernels round-trip attention and recurrent state exactly.
* [Native LFM serving engine](exp-0036-native-serving-engine.md) - one complete fixed Q8 prefill/decode run binds physical state to radix ownership.
* [Binary LFM tokenizer and typed chat encoding](exp-0037-native-tokenizer-chat.md) - native OCaml text-to-token parity without JSON, Jinja, Python, or model loading.
* [Variable-length LFM schedules and repeated native decode](exp-0038-variable-schedule-decode.md) - captured templates specialize to request lengths and preserve four eager-Q8 greedy tokens across three native decode steps.
* [Native OCaml chat generation](exp-0039-native-chat-generation.md) - binary tokenization, typed chat, dynamic prefill, greedy decode, and text output execute as one native flow.
* [Native OCaml HTTP serving and token-level ERS](exp-0040-native-http-serving.md) - persistent HTTP/SSE serving proves request-level radix reuse, corrected token timing, and four exact eager-Q8 token sequences.
* [Native long-context needle retrieval with EOS boundary](exp-0041-native-long-needle.md) - the 2,048/4,096-token matrix retrieves 6/6 and records the fixed-output contract correction without rerunning it.
* [Batched native Metal schedule submission](exp-0042-batched-metal-command.md) - one ordered command buffer per schedule preserves exact outputs and raises warmed native ERS from 0.061695 to 0.110586.
* [Q8 decode-specialized Metal GEMV](exp-0043-q8-decode-gemv.md) - one-row Q8 dispatch preserves exact tokens and lowers every measured TPOT while the one-trace ERS delta is recorded exactly.
* [Native cache-submission batching](exp-0044-cache-submission-batching.md) - ordered unpack and pack phases reduce decode submission waits while preserving exact cache bytes and tokens.
* [Cached-suffix command-buffer batching attempt](exp-0045-cached-suffix-batching-attempt.md) - dependent decode and per-token cache writes share one planned batch; the first model attempt failed before scoring and the correction has static evidence only.
* [Q8 linear-SiLU epilogue fusion](exp-0046-q8-linear-silu-fusion.md) - sixteen feed-forward pairs per stage become typed GEMM/GEMV epilogues with alias-safe rewriting and preserved float16 rounding.
* [Q8 linear-residual epilogue fusion](exp-0047-q8-linear-add-fusion.md) - all 32 same-shape residual boundaries per stage become typed GEMM/GEMV epilogues without absorbing broadcast adds.
* [Q8 multiplied-input down-projection fusion](exp-0048-q8-multiplied-input-fusion.md) - sixteen materialized SwiGLU products per stage move into typed Q8 down-projection loads.
* [SIMD-group Q8 decode GEMV](exp-0049-simdgroup-q8-gemv.md) - one SIMD group reduces each Q8 output channel while old packages retain scalar dispatch.
* [SIMD-group RMSNorm rows](exp-0050-simdgroup-rmsnorm.md) - one SIMD group reduces and writes each row while older packages retain scalar dispatch.
* [Single-pass SIMD online-softmax attention](exp-0051-online-softmax-attention.md) - each LFM query-key score is computed once while scalar fallback remains available.
* [Optimized native stack measurement](exp-0052-optimized-native-stack.md) - corrected suffix replay and the combined fused/SIMD stack retain exact tokens and measure native ERS 0.236555.
* [Effective LFM feed-forward shape correction](exp-0053-effective-ff-shape.md) - distinguishes declared 6656 from the auto-adjusted 4608-wide executable projections.
* [Packed SIMD-group Q8 GEMV](exp-0054-packed-simd-q8-gemv.md) - four activations and int8 weights per lane iteration across all fused decode families.
* [Packed SIMD Q8 GEMV model measurement](exp-0055-packed-simd-q8-gemv-measurement.md) - one bounded 350M run with exact tokens, unchanged radix reuse, and ERS 0.325370.
* [Vector-staged Q8 prefill kernel](exp-0056-vector-staged-q8-prefill.md) - 64-wide activation4/weight4 staging with fourfold fewer reduction barriers.
* [Vector-staged Q8 prefill model measurement](exp-0057-vector-prefill-measurement.md) - one bounded 350M run with exact tokens, unchanged radix reuse, and ERS 0.337742.
* [Native FP16 logit comparison](exp-0058-native-logit-export.md) - raw OCaml Metal vocabulary-row export and one bounded eager-Q8 numeric comparison without JSON tensor transport.
* [Native fixed-12-token long-context needle parity](exp-0059-fixed12-native-needle.md) - 6/6 retrieval and complete eager-Q8 token parity through the optimized OCaml Metal server.
* [Serving-only last-token vocabulary projection](exp-0060-last-token-vocab-projection.md) - typed prefill specialization projects one FP16 vocabulary row and avoids the full-sequence logits allocation.
* [Last-token vocabulary projection model measurement](exp-0061-last-token-projection-measurement.md) - one bounded 350M run preserves exact tokens and 80/194 reuse while observing ERS 0.358847.
* [Q8-default LM-head compiler boundary](exp-0062-q8-lm-head-compiler.md) - the frontend quantizes the vocabulary projection and typed prefill specialization reduces it to one Q8 row.
* [Full-Q8 LM-head capture and native measurement](exp-0063-q8-lm-head-measurement.md) - bounded 350M capture, exact short-trace parity, ERS 0.390896, and 6/6 long-context retrieval/token parity.
* [Paired SIMD-group Q8 decode GEMV](exp-0064-paired-simd-q8-gemv.md) - two output channels share each activation vector load, halving one-token threadgroup scheduling with exact synthetic Metal outputs.
* [Paired SIMD-group Q8 model measurement](exp-0065-paired-simd-measurement.md) - exact short and needle tokens with observed ERS 0.407016 and mixed latency deltas.
* [Model-scale selectable FP16 KV execution](exp-0066-fp16-kv-model-execution.md) - the paired 350M package executes FP16 KV/recurrent storage with exact tokens and unchanged radix reuse while Q8 remains default.
* [SIMD-group Q8 cache packing](exp-0067-simd-q8-cache-pack.md) - attention and recurrent quantization use one SIMD group per Q8 group.
* [SIMD Q8 cache-pack measurement](exp-0068-simd-q8-cache-pack-measurement.md) - bounded 350M token, radix, latency, and ERS evidence.
* [Vectorized Q8 cache unpack](exp-0069-vector-q8-cache-unpack.md) - vec4 attention and recurrent restoration with scalar fallback.
* [Vector Q8 cache-unpack measurement](exp-0070-vector-q8-cache-unpack-measurement.md) - bounded 350M trace and long-context evidence.
* [Direct paged-Q8 decode attention](exp-0071-paged-q8-attention.md) - the decode schedule and Metal kernel consume radix-owned Q8 K/V slots directly.
* [Direct paged-Q8 attention model measurement](exp-0072-paged-q8-attention-measurement.md) - exact bounded 350M tokens with lower long-context TPOT and lower short-trace ERS.
* [Fused RMSNorm and RoPE](exp-0073-rmsnorm-rope-fusion.md) - twelve 10-command query/key chains across 6 attention layers fuse into single SIMD Metal kernels, reducing prefill/decode plans by 108 commands.
* [Fused RMSNorm-RoPE model measurement](exp-0074-rmsnorm-rope-measurement.md) - exact bounded 350M tokens with improved short-trace ERS (0.412260) and lower TPOT.
* [Macro-operator fusion compiler boundary](exp-0076-macro-operator-fusions-compiler.md) - fresh full-Q8 prefill/decode package compilation with dual-linear, QKV, and decode ShortConv-step matches.
* [XGBoost cost-model differential evidence](exp-0077-xgboost-cost-model-differential-2026-08-26.md) - fresh Q8 differential, exact four-request token parity, and isolated raw timing observations with the runner's comparison boundary retained.
* [XGBoost kernel cost-model device sweep](exp-0078-xgboost-kernel-cost-model-device-sweep-2026-08-26.md) - parameterized native Q8 tile dispatch, broad device measurements, transpilation validation, and the recorded learned-selection boundary.
* [Macro fusion model integration](exp-0079-macro-fusion-model-integration-2026-08-26.md) - validated current full-model Q8 packages with dual-linear, QKV, and ShortConv-step fusions plus exact cross-process parity.
* [XGBoost cost-model repair audit](exp-0080-xgboost-kernel-cost-model-repair-2026-08-26.md) - independent selector holdout and measured-oracle boundary after the broad Q8 tile sweep.
* [XGBoost static-versus-dynamic audit](exp-0081-xgboost-static-dynamic-audit-2026-08-26.md) - paired-median comparison showing the existing artifact-selected tile policy against fixed `16x16x64`.
* [Macro-fusion full-package audit](exp-0082-macro-fusion-full-package-audit-2026-08-26.md) - fresh all-five Q8 package inventory, validation, and static command delta.
* [XGBoost relative-target selector repair](exp-0083-xgboost-relative-target-repair-2026-08-26.md) - grouped holdout/cross-validation evidence for the repaired target and fresh static package replan.
* [Macro-fusion relative-model replan](exp-0084-macro-fusion-relative-model-replan-2026-08-26.md) - fresh all-five package inventory and serving-pair validation after installing the repaired selector.
* [Q8 residual/output metadata validation repair](exp-0085-q8-residual-metadata-validator-2026-08-26.md) - schedule-level rejection of runtime-invalid residual metadata and rebuilt v3 package evidence.
* [Q8 LM-head kernel-manifest repair](exp-0086-q8-lm-head-manifest-repair-2026-08-26.md) - one bounded native failure, the mixed-fusion ABI repair, and corrected v4 static package evidence.
* [Q8 v4 macro-fusion native execution](exp-0087-q8-macro-probe-v4-execution.md) - successful Apple M4 Pro supervised native execution with 4/4 warmup, 4/4 scored parity, and zero token mismatches.
* [Q8 serving record breakthrough](exp-0088-q8-serving-record-breakthrough-2026-08-26.md) - decode-first scheduling fix and SIMD-pair kernels breaking repository all-time record with ERS 0.4290 (TPOT 6.81 ms).
* [llama.cpp target benchmark setup](exp-0089-llama-cpp-target-2026-08-26.md) - native llama-bench throughput, same-trace llama-server ERS, and optional llmopt side comparison.
* [Prebaked decode dispatch and Q4 target replay](exp-0090-prebaked-decode-q4-comparison-2026-08-27.md) - retained Metal decode records, refreshed Q8 side evidence, and the requested llama.cpp Q4_0 timing comparison.
* [Canonical W4A16/KVQ8 pipeline and Q4 comparison](exp-0091-canonical-w4a16-kvq8-pipeline-2026-08-27.md) - W4 LM-head repair, removal of superseded paths, ABI-v16 engine generation, and one native Q4_0 shared-trace measurement.
* [Executable W4A16 SwiGLU fusion rule](exp-0092-w4a16-swiglu-rule-2026-08-27.md) - restores rule-driven FFN execution, generic regression coverage, package auditing, and measured parallel W4 lowering.
* [Restored W4 SIMD execution and corrective rerun](exp-0093-restored-w4-simd-rerun-2026-08-27.md) - ports the retained SIMD/vector execution strategy into canonical W4 kernels and records the cleanup failure and corrected Q4 comparison.
* [W4A16 RMSNorm/SwiGLU cast absorption](exp-0094-w4a16-rms-cast-absorption-2026-08-27.md) - removes the 33 single-use f16-to-f32 widening casts from each preserved graph and records the static package delta.
* [W4A16 decode RoPE table elision](exp-0095-w4a16-rope-table-elision-2026-08-27.md) - precomputes all position rows, binds canonical runtime cosine/sine inputs, prunes the scalar branch, and records the native dispatch/timing comparison.
* [Superseded SmolLM2 cross-model validation claim](exp-0096-cross-model-smollm2-validation.md) - retracts unsupported full-model and benchmark numbers and points to the reproducible replacement.
* [Hardware-derived prefill cost model](exp-0097-hardware-derived-prefill-cost-model.md) - records the target-hardware and cost-model work retained by the current compiler.
* [Parameterized sampling engine](exp-0098-parameterized-sampling-engine.md) - records the generic generation sampling implementation.
* [Architecture-neutral GGUF FX native linear parity](exp-0099-gguf-fx-native-linear-parity-2026-08-29.md) - one real SmolLM, Qwen, and Gemma GGUF tensor captured through torch.compile and executed by the native Metal runtime with exact measured deltas.
* [Current LFM Model Program comparison with llama.cpp](exp-0100-model-program-v2-llama-cpp-comparison-2026-08-29.md) - four paired same-text runs comparing the current ABI-v2 LFM serving engine with llama.cpp Q4_0 while retaining the quantization and tokenization differences.
* [Captured full-model GGUF comparison](exp-0101-gguf-full-model-comparison-2026-08-29.md) - zero-opaque native SmolLM, Qwen, and Gemma forwards, Qwen gated-delta numerical evidence, Gemma RMSNorm fusion, and llama.cpp timing context.
* [Captured-shape SIMD attention specialization](exp-0102-captured-attention-specialization-2026-08-29.md) - graph-driven h256/h512 kernels cut Gemma and Qwen full-forward latency while preserving baseline argmax IDs.
* [Quantization-neutral Linear region discovery](exp-0103-semantic-linear-regions-2026-08-29.md) - semantic Linear fusion recovery across mixed GGUF layouts with fresh Qwen/Gemma llama.cpp context.
* [Tensor-layout quantization and Linear storage classification](exp-0104-tensor-layout-capabilities-2026-08-29.md) - one physical-layout authority and closed Linear storage representation across archive, IR, pass, and planner seams.
* [Typed Scan regions and Qwen recurrence recovery](exp-0105-typed-scan-recovery-2026-08-29.md) - graph-structural recovery of all 18 Qwen 63-step carried-state regions without architecture identifiers.
* [Registered Metal tactics and paired-row Q4_K Linear](exp-0106-registered-metal-tactics-2026-08-29.md) - compiler-selected Metal implementations and paired-row Q4_K execution without model identifiers.
* [Captured triangular recurrence fusion](exp-0107-triangular-recurrence-fusion-2026-08-30.md) - structural carried-state lowering and semantic kernel dispatch with fresh Qwen/Gemma llama.cpp context.
* [Paired-row mixed-quant Linear tactics](exp-0108-paired-row-mixed-quant-linear-2026-08-30.md) - shape/layout/dtype-selected paired execution across all supported GGUF quant layouts.
* [SIMD-group batched matmul tactic](exp-0109-simdgroup-batched-matmul-2026-08-30.md) - aligned SIMD matrix execution plus a tiled fallback selected from captured shape.
* [Sub-SIMD short-row K-quant Linear tactics](exp-0110-subsimd-short-row-kquant-linear-2026-08-30.md) - four-column vectorized Q4_K/Q5_K/Q6_K execution selected from captured shape and layout.
* [Graph-recovered zero-state gated-delta execution](exp-0111-graph-recovered-gated-delta-2026-08-30.md) - captured-topology semantic replacement and direct SIMD recurrence execution.
* [Full-SIMD two-column Q4_K Linear tactic](exp-0112-full-simd-q4-k-linear-2026-08-30.md) - full-lane packed Q4_K execution selected from captured geometry and physical layout.
* [Occupancy-selected full-SIMD Q5_K Linear tactic](exp-0113-occupancy-selected-q5-k-linear-2026-08-30.md) - four-superblock Q5_K execution selected from captured occupancy geometry.
