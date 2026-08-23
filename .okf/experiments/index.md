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
