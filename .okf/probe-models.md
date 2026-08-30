---
type: Reference
title: 'Compiler and runtime probe models'
description: 'Models used to expose architecture coverage without becoming compiler or runtime defaults.'
tags: [models, probes, compiler, runtime, portability]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T12:05:08+07:00' }
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
| `HuggingFaceTB/SmolLM2-135M-Instruct` | Q4_K_M GGUF: 2,131-node capture, 273 statics, 2,461-command zero-opaque native full forward; both token argmax IDs match the same-GGUF Transformers reference | Two-token no-cache median is `10.454059 ms` versus llama.cpp `3.7798125 ms` (`2.7658x`). No LLMOpt cached decode, Model Program, or HTTP serving run. |
| `unsloth/Qwen3.5-0.8B` | UD-Q4_K_XL GGUF: 14,219-node capture, all 321 statics resolved, and a 2,507-command/1,128-dispatch zero-opaque native full forward after graph-recovered gated-delta and final-axis L2-normalization fusion, including direct `IQ4_XS` Linear execution | Two-token no-cache median is `12.558937 ms` versus llama.cpp `7.931209 ms` (`1.583483x`). Graph-general Q4_K/Q5_K tactics and 36 topology-fused L2 normalizations preserve argmax rows `760,16`; full Torch parity remains absent against corrected reference `198,16`. No cached decode, Model Program, or HTTP serving run. |
| `unsloth/gemma-4-E2B-it` | UD-Q4_K_XL GGUF: 4,399-node capture and zero-opaque native full forward; graph-general RMSNorm fusion plus output liveness produce 3,997 commands/1,635 dispatches, while full-SIMD Q4_K execution preserves both token argmax IDs | Latest two-token no-cache observation is `28.671980 ms` versus llama.cpp `17.3268545 ms` (`1.654771x`). The L2-normalization and Q5_K occupancy selections match zero sites and output remains byte exact with slice 11. No LLMOpt cached decode, Model Program, or HTTP serving run. |

# Adding a probe

A new probe supplies captured packages, explicit `Model_profile` metadata, and
state-role bindings when the graph has persistent state.
Model-specific fixtures belong in a probe module or experiment harness. If a
new topology exposes missing IR, ABI, specialization, or runtime vocabulary,
that capability is generalized at its owning layer; the model name is not
added as a dispatch condition.
