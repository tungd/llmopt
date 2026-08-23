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
