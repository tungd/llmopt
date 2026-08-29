---
type: Prior Art
title: 'TensorRT-LLM: the same AOT-engine bet, industrialized'
description: 'NVIDIA compiles a model into a hardware-specific, non-portable engine with build-time autotuning and AOT shape profiles, but still hand-writes the hot kernels per GPU architecture.'
tags: [prior-art, aot, compiler, tensorrt, nvidia, autotuning, shape-profiles, cuda-graphs]
status: stable
stale_after: '2027-02-28'
generated: { by: claude/opus-5, at: '2026-08-29T06:01:18Z' }
sources:
  - id: trt-deploy
    resource: https://www.spheron.network/blog/tensorrt-llm-production-deployment-guide/
    title: TensorRT-LLM production deployment - engine build, multi-GPU serving, in-flight batching (2026)
  - id: trt-arch
    resource: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/architecture/architecture-overview.html
    title: NVIDIA TensorRT architecture overview
  - id: compiler-study
    resource: https://link.springer.com/article/10.1007/s11227-026-08559-6
    title: Characterization of machine learning compilers for LLM inference on NVIDIA GPUs
  - id: local-aot
    resource: /decisions/aot-decode-solidification-zero-jit-serving.md
    title: llmopt AOT decode solidification decision
  - id: local-cost-model
    resource: /decisions/xgboost-kernel-cost-model.md
    title: llmopt learned kernel cost model decision
---

# What it is

TensorRT-LLM is a Python library plus CUDA runtime built on TensorRT. It ingests a
Hugging Face checkpoint, applies graph fusion and FP8 / INT4-AWQ quantization, and
emits a `.plan` engine: a pre-fused, quantized, hardware-specific execution graph.
Engines are bound to one GPU architecture and one dtype combination and are not
portable between them.[^trt-deploy][^trt-arch]

This is structurally the same bet as llmopt's: all specialization happens offline,
the runtime is a thin executor over a solidified artifact.[^local-aot] The
`.plan` + CUDA-runtime pair corresponds to llmopt's `model.llmopt` +
`kernel.metallib` pair.

# Three points of comparison

| Dimension | TensorRT-LLM | llmopt |
| :--- | :--- | :--- |
| Artifact | `.plan` engine, one GPU arch + dtype | `model.llmopt` + `kernel.metallib`, one probed Apple target |
| Kernel selection | Candidate tactics **benchmarked on the real device at build time** | Learned GBDT cost model predicting tile geometry |
| Dynamic shapes | Optimization profiles (min/opt/max) baked AOT | Prefill template buckets; decode still specialized per sequence length at runtime |
| Hot kernels | Hand-written CUDA C++ per architecture (attention, MoE, quant-dequant) | Generated MSL from fusion passes |
| Command replay | CUDA Graphs | Prebaked `MTLIndirectCommandBuffer` |
| Batching | In-flight (continuous) batching in the runtime | Queue exists; decode still executes one sequence per dispatch |

# Why it matters for llmopt

**Measured autotuning versus predicted cost.** TensorRT selects kernel tactics by
timing candidates on the actual device during the build. llmopt predicts with a
learned model instead.[^local-cost-model] Since llmopt already emits variant
families (`_m4`, `_m8` and per-quant-scheme dequantizers), those variants form a
tactic set that could be timed directly at package-build time — removing the need
for the cost model to generalize, and removing the per-target retraining cost that
any second hardware backend would otherwise incur.

**Shape profiles, not runtime specialization.** TensorRT's answer to variable
sequence length is a small set of shape profiles chosen at build time. llmopt's
`Target_hardware.Prefill_cost_model` already emits `template_buckets`, which is the
same construct; extending it to decode is the standing fix for runtime schedule
specialization.[^local-aot]

**The uncomfortable one.** NVIDIA, with the most mature AOT inference compiler in
production, still hand-writes attention, MoE, and quantization kernels in CUDA C++
for each new architecture, and lets the compiler do graph-level fusion around
them.[^trt-deploy] Peak performance on state-of-the-art models is reported to
require exactly this architecture-specific tooling.[^compiler-study] That is
evidence against the strong form of llmopt's "no bespoke kernels" claim, and the
choice it forces is explicit: either accept a curated kernel library that grows per
architecture, or pursue parametric kernel bodies as in
[Modular MAX / Mojo](modular-max-mojo.md).

[^trt-deploy]: TensorRT-LLM production deployment guide (2026).
[^trt-arch]: NVIDIA TensorRT architecture overview.
[^compiler-study]: Characterization of ML compilers for LLM inference on NVIDIA GPUs.
[^local-aot]: llmopt decision - AOT decode solidification and zero-JIT serving.
[^local-cost-model]: llmopt decision - XGBoost / GBDT kernel cost model.
