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
