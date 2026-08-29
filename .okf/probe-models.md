---
type: Reference
title: 'Compiler and runtime probe models'
description: 'Models used to expose architecture coverage without becoming compiler or runtime defaults.'
tags: [models, probes, compiler, runtime, portability]
status: stable
generated: { by: codex/gpt-5, at: '2026-08-29T13:54:08+07:00' }
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
| `HuggingFaceTB/SmolLM2-135M-Instruct` | One real Q5_0 GGUF Linear executed 576/576 exact; full CPU FX capture returned 274 placeholders and mapped 273/273 state entries | No complete native package, generation run, or llama.cpp performance receipt. The earlier unsupported full-model/performance claim is deprecated. |
| `unsloth/Qwen3.5-0.8B` | One real UD Q4_K GGUF Linear executed 3581/3584 exact with max abs `0.000030517578125`; full meta capture contained 14,219 nodes | 303/321 static entries map through the generic name table; derived buffers, `ssm_dt` naming, and IQ4_XS execution remain outside the recorded package. |
| `unsloth/gemma-4-E2B-it` | One real UD Q4_K GGUF Linear executed 2048/2048 exact; full meta capture contained 4,399 nodes and mapped 541/541 state entries | No complete native package or generation run; the GGUF carries additional PLE tensors and IQ4_XS tensors not executed by the recorded probe. |

# Adding a probe

A new probe supplies captured packages, explicit `Model_profile` metadata, and
state-role bindings when the graph has persistent state.
Model-specific fixtures belong in a probe module or experiment harness. If a
new topology exposes missing IR, ABI, specialization, or runtime vocabulary,
that capability is generalized at its owning layer; the model name is not
added as a dispatch condition.
