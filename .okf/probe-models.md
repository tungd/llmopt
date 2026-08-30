---
type: Reference
title: 'Compiler and runtime probe models'
description: 'Models used to expose architecture coverage without becoming compiler or runtime defaults.'
tags: [models, probes, compiler, runtime, portability]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T12:18:06+07:00' }
sources:
  - id: profile
    resource: /lib/model_profile.ml
    title: Explicit model metadata contract
  - id: linker
    resource: /lib/model_program_linker.ml
    title: Generic Model Program linker
  - id: lfm-probe
    resource: /lib/lfm25_probe.ml
    title: LFM2.5-specific probe fixture
  - id: pipeline
    resource: /bin/pipeline.ml
    title: Generic low-level pipeline
  - id: smollm-experiment
    resource: /.okf/experiments/exp-0096-cross-model-smollm2-validation.md
    title: SmolLM2 cross-model experiment
  - id: gguf-probes
    resource: /.okf/experiments/exp-0099-gguf-fx-native-linear-parity-2026-08-29.md
    title: SmolLM Qwen Gemma GGUF FX probe
  - id: lfm-performance
    resource: /.okf/experiments/exp-0100-model-program-v2-llama-cpp-comparison-2026-08-29.md
    title: Current LFM Model Program comparison with llama.cpp
  - id: full-model-comparison
    resource: /.okf/experiments/exp-0101-gguf-full-model-comparison-2026-08-29.md
    title: Captured full-model GGUF comparison
  - id: recurrence-fusion
    resource: /.okf/experiments/exp-0107-triangular-recurrence-fusion-2026-08-30.md
    title: Captured triangular recurrence fusion
  - id: mixed-quant-tactics
    resource: /.okf/experiments/exp-0108-paired-row-mixed-quant-linear-2026-08-30.md
    title: Paired-row mixed-quant Linear tactics
  - id: simd-batched-matmul
    resource: /.okf/experiments/exp-0109-simdgroup-batched-matmul-2026-08-30.md
    title: SIMD-group batched matmul tactic
  - id: short-row-kquant
    resource: /.okf/experiments/exp-0110-subsimd-short-row-kquant-linear-2026-08-30.md
    title: Sub-SIMD short-row K-quant Linear tactics
  - id: gated-delta
    resource: /.okf/experiments/exp-0111-graph-recovered-gated-delta-2026-08-30.md
    title: Graph-recovered zero-state gated-delta execution
  - id: full-simd-q4-k
    resource: /.okf/experiments/exp-0112-full-simd-q4-k-linear-2026-08-30.md
    title: Full-SIMD two-column Q4_K Linear tactic
  - id: occupancy-q5-k
    resource: /.okf/experiments/exp-0113-occupancy-selected-q5-k-linear-2026-08-30.md
    title: Occupancy-selected full-SIMD Q5_K Linear tactic
  - id: l2-normalization
    resource: /.okf/experiments/exp-0114-topology-fused-l2-normalization-2026-08-30.md
    title: Topology-fused L2 normalization
  - id: attention-linked-rms-rope
    resource: /.okf/experiments/exp-0115-attention-linked-rms-rope-2026-08-30.md
    title: Attention-linked token-major RMSNorm-RoPE fusion
  - id: activation-product
    resource: /.okf/experiments/exp-0116-activation-product-fusion-2026-08-30.md
    title: Single-consumer activation-product fusion
  - id: rmsnorm-residual
    resource: /.okf/experiments/exp-0117-rmsnorm-residual-fusion-2026-08-30.md
    title: Single-consumer RMSNorm-residual fusion
  - id: quant-linear-residual
    resource: /.okf/experiments/exp-0133-quantized-linear-residual-2026-08-30.md
    title: Topology-fused quantized Linear residuals
---

# Boundary

Probe models provide concrete graphs, state shapes, tokens, and measurement
receipts. They do not define a product model, architecture selector, generation
fallback, or success threshold.

The generic linker requires explicit model identity, generation/chat metadata,
and persistent-state roles, then validates entrypoint and state dimensions
against captured packages.
Generic serving tools load only a complete `model.llmopt` contract. The default
Ninja `all` target excludes model-specific diagnostic executables.

# Current probes

| Probe | Measured or captured coverage | Current boundary |
|---|---|---|
| `LiquidAI/LFM2.5-350M` | Complete W4A16/Q8-KV prefill, decode, cache, tokenizer, Model Program ABI v2 serving, and a four-repeat same-text comparison with llama.cpp Q4_0 | Only end-to-end serving probe. The comparison is not GGUF/UD weight parity and remains owned by `Lfm25_probe`, probe-only diagnostics, and LFM receipts. |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | Q4_K_M GGUF: 2,131-node capture, 273 statics, and a 1,054-command/340-dispatch zero-opaque native forward after topology-selected rotary, attention-layout, gated-Linear, and 60 quantized-Linear residual fusions | Latest two-token no-cache observation is `3.266931 ms` versus llama.cpp `3.291720 ms` (`0.992469x`). Both row argmax IDs remain `198,198`; the residual epilogues have mean/max logit drift `0.037337277/0.21875` from slice 30. No cached decode, Model Program, or HTTP serving run. |
| `unsloth/Qwen3.5-0.8B` | UD-Q4_K_XL GGUF: 14,219-node capture, all 321 statics resolved, and a 1,655-command/702-dispatch zero-opaque native forward after graph-recovered gated-delta, packed L2 normalization, ShortConv-SiLU, and 48 quantized-Linear residual fusions | Latest two-token no-cache observation is `7.966518 ms` versus llama.cpp `7.953770 ms` (`1.001603x`). Graph-general tactics preserve argmax rows `760,16`; full Torch parity remains absent against corrected reference `198,16`. No cached decode, Model Program, or HTTP serving run. |
| `unsloth/gemma-4-E2B-it` | UD-Q4_K_XL GGUF: 4,399-node capture and a 2,414-command/786-dispatch zero-opaque native forward; graph-general RMSNorm, attention-linked rotary, activation-product, RMSNorm-residual, and attention-value layout fusions execute independently of weight layout | Latest two-token no-cache observation is `18.554449 ms` versus llama.cpp `17.499137 ms` (`1.060307x`). Graph-derived execution preserves row argmax IDs `84904,148465`; quantized Linear residual and packed-L2 passes match zero sites. No cached decode, Model Program, or HTTP serving run. |

# Adding a probe

A new probe supplies captured packages, explicit `Model_profile` metadata, and
state-role bindings when the graph has persistent state.
Model-specific fixtures belong in a probe module or experiment harness. If a
new topology exposes missing IR, ABI, specialization, or runtime vocabulary,
that capability is generalized at its owning layer; the model name is not
added as a dispatch condition.
